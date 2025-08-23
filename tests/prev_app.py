# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional
# import asyncio
# import logging
# import json
# import time

# from controllers.insurance_recommender_chat import InsuranceAgent, UserProfile
# from clients.gemini_client import ChatMessage
# from controllers.insurance_recommender_keywords import PolicyRecommender, UserProfile as KeywordUserProfile, PolicyRecommendation
# from controllers.select_policy_chat import PolicyDocumentChat
# from controllers.policy_analyzer import PolicyScenarioAnalyzer

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create FastAPI app
# app = FastAPI(
#     title="AI Insurance Agent API",
#     description="API for AI-powered insurance policy recommendations and chat",
#     version="1.0.0"
# )

# # Global agent instances
# agent = None
# policy_recommender = None
# policy_chat_instances = {}  # Store policy chat instances by session

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the insurance agent and policy recommender on startup."""
#     global agent, policy_recommender
#     try:
#         agent = InsuranceAgent()
#         policy_recommender = PolicyRecommender()
#         logger.info("Insurance agent and policy recommender initialized successfully")
#     except Exception as e:
#         logger.error(f"Failed to initialize services: {e}")
#         raise

# # Pydantic models for request/response
# class ChatMessage(BaseModel):
#     """Individual chat message."""
#     role: str  # "user" or "assistant"
#     content: str

# class UserProfileResponse(BaseModel):
#     """User profile information."""
#     age: Optional[int] = None
#     gender: Optional[str] = None
#     occupation: Optional[str] = None
#     income_range: Optional[str] = None
#     family_size: Optional[int] = None
#     existing_conditions: List[str] = []
#     current_insurance: Optional[str] = None
#     budget: Optional[str] = None
#     coverage_preferences: List[str] = []
#     location: Optional[str] = None
#     lifestyle: Optional[str] = None
#     profile_summary: str

# class PolicyInfo(BaseModel):
#     """Enhanced policy information from search results."""
#     company_name: str
#     title: str
#     type: str
#     sub_type: str
#     pdf_link: str
#     keywords: List[str]
#     text_snippet: str
#     relevance_score: float
#     chunk_count: int = 1

# # New models for policy recommendation endpoint
# class KeywordUserProfileRequest(BaseModel):
#     """User profile for policy recommendations."""
#     age: int
#     gender: str
#     location: str
#     income_range: str
#     family_status: str
#     dependents: int
#     health_conditions: List[str] = []
#     occupation: str
#     lifestyle: str
#     risk_tolerance: str
#     budget_range: str
#     coverage_preferences: List[str]
#     additional_notes: str = ""

# class PolicyRecommendationResponse(BaseModel):
#     """Enhanced policy recommendation response."""
#     policy_content: str
#     company_name: str
#     policy_type: str
#     policy_sub_type: str
#     policy_title: str
#     pdf_link: str
#     keywords: List[str]
#     relevance_score: float
#     gemini_reasoning: str
#     qdrant_score: float
#     final_rank: int

# class RecommendationsRequest(BaseModel):
#     """Request model for policy recommendations."""
#     user_profile: KeywordUserProfileRequest
#     top_k: int = 20
#     top_n: int = 5
#     score_threshold: float = 0.5

# class RecommendationsResponse(BaseModel):
#     """Response model for policy recommendations."""
#     recommendations: List[PolicyRecommendationResponse]
#     total_policies_found: int
#     search_query_used: str
#     processing_time_seconds: Optional[float] = None

# class ChatRequest(BaseModel):
#     """Request model for chat endpoint."""
#     message: str
#     conversation_history: List[ChatMessage] = []
#     user_profile: Optional[Dict[str, Any]] = None

# class ChatResponse(BaseModel):
#     """Response model for chat endpoint."""
#     response: str
#     updated_profile: UserProfileResponse
#     search_performed: bool
#     search_reasoning: Optional[str] = None
#     policies_found: List[PolicyInfo] = []
#     conversation_id: Optional[str] = None

# # New models for policy chat endpoint
# class PolicyChatRequest(BaseModel):
#     """Request model for policy chat endpoint."""
#     pdf_url: str
#     message: str
#     session_id: Optional[str] = None
#     collection_name: str = "policies"

# class PolicyDocumentInfo(BaseModel):
#     """Policy document information."""
#     title: str
#     url: str
#     total_chunks: int
#     total_tokens: int
#     average_tokens_per_chunk: int
#     chat_session_messages: int

# class PolicyChatResponse(BaseModel):
#     """Response model for policy chat endpoint."""
#     response: str
#     document_info: PolicyDocumentInfo
#     session_id: str
#     token_usage: Dict[str, Any]

# class PolicyAnalyzerRequest(BaseModel):
#     """Request model for policy analyzer endpoint."""
#     pdf_url: str
#     collection_name: str = "policies"

# class PolicyAnalyzerResponse(BaseModel):
#     """Response model for policy analyzer endpoint."""
#     analysis: str
#     document_title: str
#     document_id: str
#     total_chunks: int
#     total_tokens: int
#     processing_time_seconds: float
#     token_usage: Dict[str, Any]
#     pdf_url: str

# def convert_profile_to_response(profile: UserProfile) -> UserProfileResponse:
#     """Convert UserProfile to UserProfileResponse."""
#     return UserProfileResponse(
#         age=profile.age,
#         gender=profile.gender,
#         occupation=profile.occupation,
#         income_range=profile.income_range,
#         family_size=profile.family_size,
#         existing_conditions=profile.existing_conditions or [],
#         current_insurance=profile.current_insurance,
#         budget=profile.budget,
#         coverage_preferences=profile.coverage_preferences or [],
#         location=profile.location,
#         lifestyle=profile.lifestyle,
#         profile_summary=profile.get_profile_summary()
#     )

# def convert_policies_to_response(policies: List[Dict[str, Any]]) -> List[PolicyInfo]:
#     """Convert policy search results to PolicyInfo list with grouping by title."""
#     if not policies:
#         return []
    
#     # Group policies by title and company to avoid duplicates
#     policy_groups = {}
#     for policy in policies:
#         title = policy.get("title", "Unknown Policy")
#         company_name = policy.get("company_name", "Unknown Company")
#         pdf_link = policy.get("pdf_link", "")
        
#         key = f"{company_name}_{title}_{pdf_link}"
        
#         if key not in policy_groups:
#             policy_groups[key] = {
#                 "company_name": company_name,
#                 "title": title,
#                 "type": policy.get("type", ""),
#                 "sub_type": policy.get("sub_type", ""),
#                 "pdf_link": pdf_link,
#                 "all_keywords": set(),
#                 "all_text": [],
#                 "best_score": 0,
#                 "chunk_count": 0
#             }
        
#         # Accumulate data from all chunks
#         policy_groups[key]["all_keywords"].update(policy.get("keywords", []))
#         policy_groups[key]["all_text"].append(policy.get("text", ""))
#         policy_groups[key]["best_score"] = max(policy_groups[key]["best_score"], policy.get("score", 0))
#         policy_groups[key]["chunk_count"] += 1
    
#     # Convert to PolicyInfo objects
#     policy_list = []
#     for group_data in policy_groups.values():
#         # Combine text from all chunks, taking first 300 chars
#         combined_text = " ".join(group_data["all_text"])
#         text_snippet = combined_text[:300] + "..." if len(combined_text) > 300 else combined_text
        
#         policy_info = PolicyInfo(
#             company_name=group_data["company_name"],
#             title=group_data["title"],
#             type=group_data["type"],
#             sub_type=group_data["sub_type"],
#             pdf_link=group_data["pdf_link"],
#             keywords=list(group_data["all_keywords"])[:10],  # Limit to top 10 keywords
#             text_snippet=text_snippet,
#             relevance_score=group_data["best_score"],
#             chunk_count=group_data["chunk_count"]
#         )
#         policy_list.append(policy_info)
    
#     # Sort by relevance score
#     policy_list.sort(key=lambda x: x.relevance_score, reverse=True)
#     return policy_list

# def convert_keyword_recommendations_to_response(recommendations: List[PolicyRecommendation]) -> List[PolicyRecommendationResponse]:
#     """Convert PolicyRecommendation objects to PolicyRecommendationResponse."""
#     return [
#         PolicyRecommendationResponse(
#             policy_content=rec.policy_content,
#             company_name=rec.company_name,
#             policy_type=rec.policy_type,
#             policy_sub_type=rec.policy_sub_type,
#             policy_title=rec.policy_title,
#             pdf_link=rec.pdf_link,
#             keywords=rec.keywords,
#             relevance_score=rec.relevance_score,
#             gemini_reasoning=rec.gemini_reasoning,
#             qdrant_score=rec.qdrant_score,
#             final_rank=rec.final_rank
#         )
#         for rec in recommendations
#     ]

# def get_document_url(document: Dict[str, Any]) -> str:
#     """Extract URL from document, checking multiple possible field names."""
#     url_fields = ["url", "pdf_url", "pdf_link", "link"]
#     for field in url_fields:
#         url = document.get(field, "")
#         if url:
#             return url
#     return ""

# async def get_or_create_policy_chat(session_id: str, pdf_url: str, collection_name: str) -> PolicyDocumentChat:
#     """Get existing policy chat instance or create new one."""
#     global policy_chat_instances
    
#     if session_id not in policy_chat_instances:
#         # Create new policy chat instance
#         policy_chat = PolicyDocumentChat(collection_name=collection_name)
#         await policy_chat.initialize()
        
#         # Search for document
#         document = policy_chat.search_document_by_url(pdf_url)
#         if not document:
#             raise HTTPException(status_code=404, detail=f"Policy document not found for URL: {pdf_url}")
        
#         # Update the document metadata to include the correct URL
#         document_url = get_document_url(document)
#         if not document_url:
#             # If no URL found in document, use the search URL
#             document_url = pdf_url
        
#         # Prepare document chunks
#         if not policy_chat.prepare_document_chunks(document):
#             raise HTTPException(status_code=500, detail="Failed to process policy document content")
        
#         # Override the URL in document metadata to ensure it's correct
#         policy_chat.document_metadata["url"] = document_url
        
#         policy_chat_instances[session_id] = policy_chat
#         logger.info(f"Created new policy chat session: {session_id}")
    
#     return policy_chat_instances[session_id]

# @app.post("/chat-with-policy", response_model=PolicyChatResponse)
# async def chat_with_policy(request: PolicyChatRequest):
#     """
#     Chat with a specific insurance policy document.
    
#     **Input:**
#     - `pdf_url` (string): URL of the PDF policy document to chat with
#     - `message` (string): User's question about the policy
#     - `session_id` (string, optional): Session ID for maintaining conversation context
#     - `collection_name` (string, optional): MongoDB collection name (default: "policies")
    
#     **Output:**
#     - `response` (string): AI assistant's response about the policy
#     - `document_info` (object): Information about the loaded policy document
#         - `title` (string): Policy document title
#         - `url` (string): Policy document URL
#         - `total_chunks` (int): Number of text chunks created from document
#         - `total_tokens` (int): Total tokens in the document
#         - `average_tokens_per_chunk` (int): Average tokens per chunk
#         - `chat_session_messages` (int): Number of messages in current session
#     - `session_id` (string): Session ID for this conversation
#     - `token_usage` (object): Token usage and cost information
    
#     **Example Request:**
#     ```json
#     {
#         "pdf_url": "https://example.com/health-policy.pdf",
#         "message": "What are the key benefits of this policy?",
#         "session_id": "user123_policy456",
#         "collection_name": "policies"
#     }
#     ```
    
#     **Example Response:**
#     ```json
#     {
#         "response": "This health insurance policy offers comprehensive coverage including...",
#         "document_info": {
#             "title": "Comprehensive Health Insurance Policy",
#             "url": "https://example.com/health-policy.pdf",
#             "total_chunks": 15,
#             "total_tokens": 45000,
#             "average_tokens_per_chunk": 3000,
#             "chat_session_messages": 2
#         },
#         "session_id": "user123_policy456",
#         "token_usage": {
#             "total_cost": 0.0234,
#             "totals": {
#                 "total_tokens": 1250
#             }
#         }
#     }
#     ```
#     """
#     try:
#         import uuid
#         import time
        
#         # Generate session ID if not provided
#         session_id = request.session_id or f"policy_chat_{uuid.uuid4().hex[:8]}"
        
#         # Get or create policy chat instance
#         policy_chat = await get_or_create_policy_chat(
#             session_id, request.pdf_url, request.collection_name
#         )
        
#         # Process user message
#         start_time = time.time()
#         response_text = await policy_chat.chat(request.message)
#         processing_time = time.time() - start_time
        
#         # Get document info
#         doc_info_dict = policy_chat.get_document_info()
        
#         document_url = doc_info_dict.get("url", "")
#         if not document_url:
#             document_url = request.pdf_url
        
#         document_info = PolicyDocumentInfo(
#             title=doc_info_dict.get("title", "Unknown Policy"),
#             url=document_url,
#             total_chunks=doc_info_dict.get("total_chunks", 0),
#             total_tokens=doc_info_dict.get("total_tokens", 0),
#             average_tokens_per_chunk=doc_info_dict.get("average_tokens_per_chunk", 0),
#             chat_session_messages=doc_info_dict.get("chat_session_messages", 0)
#         )
        
#         # Get token usage
#         token_usage = policy_chat.get_token_usage_summary()
        
#         # Prepare response
#         chat_response = PolicyChatResponse(
#             response=response_text,
#             document_info=document_info,
#             session_id=session_id,
#             token_usage=token_usage
#         )
        
#         logger.info(f"Policy chat completed in {processing_time:.2f}s - Session: {session_id}")
#         return chat_response
        
#     except HTTPException:
#         # Re-raise HTTP exceptions
#         raise
#     except Exception as e:
#         logger.error(f"Error in policy chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to process policy chat: {str(e)}")

# @app.delete("/chat-with-policy/{session_id}")
# async def close_policy_chat_session(session_id: str):
#     """
#     Close a policy chat session and clean up resources.
    
#     **Input:**
#     - `session_id` (string): Session ID to close
    
#     **Output:**
#     - `message` (string): Confirmation message
#     - `session_closed` (boolean): Whether session was successfully closed
#     """
#     global policy_chat_instances
    
#     if session_id in policy_chat_instances:
#         try:
#             policy_chat_instances[session_id].close()
#             del policy_chat_instances[session_id]
#             logger.info(f"Closed policy chat session: {session_id}")
#             return {
#                 "message": f"Policy chat session {session_id} closed successfully",
#                 "session_closed": True
#             }
#         except Exception as e:
#             logger.error(f"Error closing policy chat session {session_id}: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to close session: {str(e)}")
#     else:
#         return {
#             "message": f"Policy chat session {session_id} not found",
#             "session_closed": False
#         }

# @app.get("/chat-with-policy/sessions")
# async def list_policy_chat_sessions():
#     """
#     List all active policy chat sessions.
    
#     **Output:**
#     - `active_sessions` (list): List of active session IDs
#     - `total_sessions` (int): Total number of active sessions
#     """
#     global policy_chat_instances
    
#     sessions = []
#     for session_id, policy_chat in policy_chat_instances.items():
#         doc_info = policy_chat.get_document_info()
#         sessions.append({
#             "session_id": session_id,
#             "document_title": doc_info.get("title", "Unknown"),
#             "document_url": doc_info.get("url", ""),
#             "messages_count": doc_info.get("chat_session_messages", 0)
#         })
    
#     return {
#         "active_sessions": sessions,
#         "total_sessions": len(sessions)
#     }

# @app.post("/recommend-policies", response_model=RecommendationsResponse)
# async def recommend_policies(request: RecommendationsRequest):
#     """
#     Get AI-powered policy recommendations based on user profile.
    
#     **Input:**
#     - `user_profile` (object): Comprehensive user profile information
#         - `age` (int): User's age
#         - `gender` (string): User's gender
#         - `location` (string): User's location
#         - `income_range` (string): User's income range
#         - `family_status` (string): Single, married, divorced, etc.
#         - `dependents` (int): Number of dependents
#         - `health_conditions` (list): List of health conditions
#         - `occupation` (string): User's occupation
#         - `lifestyle` (string): Active, sedentary, etc.
#         - `risk_tolerance` (string): Low, medium, high
#         - `budget_range` (string): Budget range for insurance
#         - `coverage_preferences` (list): Types of insurance interested in
#         - `additional_notes` (string): Any additional information
#     - `top_k` (int, optional): Number of policies to retrieve from vector DB (default: 20)
#     - `top_n` (int, optional): Number of final recommendations to return (default: 5)
#     - `score_threshold` (float, optional): Minimum similarity score (default: 0.7)
    
#     **Output:**
#     - `recommendations` (list): List of top policy recommendations
#         - Each recommendation includes company, title, type, keywords, relevance score, reasoning, etc.
#     - `total_policies_found` (int): Total number of policies found in search
#     - `search_query_used` (string): The optimized search query generated by AI
#     - `processing_time_seconds` (float): Time taken to process the request
    
#     **Example Request:**
#     ```json
#     {
#         "user_profile": {
#             "age": 35,
#             "gender": "male",
#             "location": "Delhi, India",
#             "income_range": "20-22 LPA",
#             "family_status": "married",
#             "dependents": 2,
#             "health_conditions": ["diabetes"],
#             "occupation": "software engineer",
#             "lifestyle": "sedentary",
#             "risk_tolerance": "medium",
#             "budget_range": "30-40K per month",
#             "coverage_preferences": ["health", "life"],
#             "additional_notes": "Looking for family coverage"
#         },
#         "top_k": 15,
#         "top_n": 3,
#         "score_threshold": 0.5
#     }
#     ```
    
#     **Example Response:**
#     ```json
#     {
#         "recommendations": [
#             {
#                 "policy_content": "Comprehensive health coverage with diabetes management...",
#                 "company_name": "HDFC ERGO",
#                 "policy_type": "health",
#                 "policy_sub_type": "family",
#                 "policy_title": "Optima Secure Health Policy",
#                 "pdf_link": "https://example.com/policy.pdf",
#                 "keywords": ["health", "diabetes", "family"],
#                 "relevance_score": 8.5,
#                 "gemini_reasoning": "This policy is highly suitable for a 35-year-old software engineer...",
#                 "qdrant_score": 0.92,
#                 "final_rank": 1
#             }
#         ],
#         "total_policies_found": 15,
#         "search_query_used": "35-year-old male software engineer Delhi diabetes family health insurance...",
#         "processing_time_seconds": 12.5
#     }
#     ```
#     """
#     global policy_recommender
    
#     if not policy_recommender:
#         raise HTTPException(status_code=500, detail="Policy recommender not initialized")
    
#     try:
#         import time
#         start_time = time.time()
        
#         # Convert request profile to KeywordUserProfile
#         user_profile = KeywordUserProfile(
#             age=request.user_profile.age,
#             gender=request.user_profile.gender,
#             location=request.user_profile.location,
#             income_range=request.user_profile.income_range,
#             family_status=request.user_profile.family_status,
#             dependents=request.user_profile.dependents,
#             health_conditions=request.user_profile.health_conditions,
#             occupation=request.user_profile.occupation,
#             lifestyle=request.user_profile.lifestyle,
#             risk_tolerance=request.user_profile.risk_tolerance,
#             budget_range=request.user_profile.budget_range,
#             coverage_preferences=request.user_profile.coverage_preferences,
#             additional_notes=request.user_profile.additional_notes
#         )
        
#         # Generate optimized search query first to include in response
#         search_query = await policy_recommender.generate_optimized_search_query(user_profile)
        
#         # Get policy recommendations
#         recommendations = await policy_recommender.recommend_policies(
#             user_profile=user_profile,
#             top_k=request.top_k,
#             top_n=request.top_n,
#             score_threshold=request.score_threshold
#         )
        
#         # Calculate processing time
#         processing_time = time.time() - start_time
        
#         # Convert recommendations to response format
#         response_recommendations = convert_keyword_recommendations_to_response(recommendations)
        
#         # Prepare response
#         response = RecommendationsResponse(
#             recommendations=response_recommendations,
#             total_policies_found=len(recommendations),
#             search_query_used=search_query,
#             processing_time_seconds=round(processing_time, 2)
#         )
        
#         logger.info(f"Policy recommendations completed in {processing_time:.2f}s - Returned {len(recommendations)} recommendations")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in policy recommendation endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

# @app.post("/chat", response_model=ChatResponse)
# async def chat_with_agent(request: ChatRequest):
#     """
#     Chat with the AI Insurance Agent.
    
#     **Input:**
#     - `message` (string): User's message/question
#     - `conversation_history` (list, optional): Previous messages in the conversation
#         - Each message has `role` ("user" or "assistant") and `content` (string)
#     - `user_profile` (dict, optional): Current user profile information
#         - Contains fields like age, gender, occupation, existing_conditions, etc.
    
#     **Output:**
#     - `response` (string): Agent's response to the user
#     - `updated_profile` (object): Updated user profile after processing the message
#         - Contains all profile fields plus a `profile_summary` string
#     - `search_performed` (boolean): Whether policy database search was performed
#     - `search_reasoning` (string, optional): Explanation for search decision
#     - `policies_found` (list): List of relevant policies if search was performed
#         - Each policy contains company_name, title, type, keywords, pdf_link, text_snippet, relevance_score, and chunk_count
#     - `conversation_id` (string, optional): Identifier for conversation tracking
    
#     **Example Request:**
#     ```json
#     {
#         "message": "I'm 30 years old and need health insurance",
#         "conversation_history": [
#             {"role": "assistant", "content": "Hello! How can I help you with insurance?"},
#             {"role": "user", "content": "Hi, I need insurance advice"}
#         ],
#         "user_profile": {
#             "age": null,
#             "occupation": "software engineer"
#         }
#     }
#     ```
    
#     **Example Response:**
#     ```json
#     {
#         "response": "Great! As a 30-year-old software engineer, I can help you find suitable health insurance...",
#         "updated_profile": {
#             "age": 30,
#             "occupation": "software engineer",
#             "existing_conditions": [],
#             "profile_summary": "Age: 30; Occupation: software engineer"
#         },
#         "search_performed": true,
#         "search_reasoning": "User provided age and requested health insurance recommendations",
#         "policies_found": [
#             {
#                 "company_name": "HDFC ERGO",
#                 "title": "Optima Secure Health Policy",
#                 "type": "health",
#                 "sub_type": "individual",
#                 "pdf_link": "https://example.com/policy.pdf",
#                 "keywords": ["health", "young adults", "comprehensive"],
#                 "text_snippet": "This policy offers comprehensive health coverage...",
#                 "relevance_score": 0.85,
#                 "chunk_count": 3
#             }
#         ]
#     }
#     ```
#     """
#     global agent
    
#     if not agent:
#         raise HTTPException(status_code=500, detail="Insurance agent not initialized")
    
#     try:
#         # Initialize user profile
#         if request.user_profile:
#             user_profile = UserProfile(**request.user_profile)
#         else:
#             user_profile = UserProfile()
        
#         # Convert conversation history
#         conversation_history = [msg.content for msg in request.conversation_history]
        
#         # Update user profile using AI
#         user_profile = await agent.update_user_profile(
#             request.message, conversation_history, user_profile
#         )
        
#         # Let AI decide if we should search policies
#         search_decision = await agent.should_search_policies(
#             request.message, conversation_history, user_profile
#         )
        
#         # Search policies if needed
#         policies = []
#         search_performed = search_decision.get("should_search", False)
#         search_reasoning = search_decision.get("reasoning", None)
        
#         if search_performed:
#             search_query = search_decision.get("search_query", request.message)
#             policies = await agent.search_policies(search_query, top_k=15)  # Get more chunks for better grouping
#             logger.info(f"Found {len(policies)} policy chunks for query: {search_query}")
        
#         # Get enhanced message with proper context
#         enhanced_message = await agent.get_enhanced_response(
#             request.message, user_profile, conversation_history, policies
#         )
        
#         # Generate response using AI
#         try:
#             response = await agent.gemini_client.generate_with_thinking(
#                 prompt=enhanced_message,
#                 model=agent.model,
#                 thinking_budget=agent.thinking_budget,
#                 include_thoughts=False,
#                 temperature=0.7
#             )
            
#             # Check if response is valid
#             if not response or not response.text:
#                 logger.error("Received empty response from Gemini")
#                 raise HTTPException(status_code=500, detail="Failed to generate response")
        
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
        
#         # Prepare response
#         chat_response = ChatResponse(
#             response=response.text,
#             updated_profile=convert_profile_to_response(user_profile),
#             search_performed=search_performed,
#             search_reasoning=search_reasoning,
#             policies_found=convert_policies_to_response(policies)
#         )
        
#         logger.info(f"Chat completed - Profile: {user_profile.get_profile_summary()}")
#         logger.info(f"Found {len(chat_response.policies_found)} unique policies from {len(policies)} chunks")
#         return chat_response
        
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/policy-analyzer", response_model=PolicyAnalyzerResponse)
# async def analyze_policy(request: PolicyAnalyzerRequest):
#     """
#     Analyze an insurance policy document against predefined scenarios.
    
#     **Input:**
#     - `pdf_url` (string): URL of the PDF policy document to analyze
#     - `collection_name` (string, optional): MongoDB collection name (default: "policies")
    
#     **Output:**
#     - `analysis` (string): Comprehensive scenario-based analysis of the policy
#     - `document_title` (string): Title of the analyzed policy document
#     - `document_id` (string): MongoDB document ID
#     - `total_chunks` (int): Number of text chunks created from document
#     - `total_tokens` (int): Total tokens in the document
#     - `processing_time_seconds` (float): Time taken to complete the analysis
#     - `token_usage` (object): Token usage and cost information
#     - `pdf_url` (string): The analyzed policy document URL
    
#     **Example Request:**
#     ```json
#     {
#         "pdf_url": "https://example.com/health-policy.pdf",
#         "collection_name": "policies"
#     }
#     ```
    
#     **Example Response:**
#     ```json
#     {
#         "analysis": "**Scenario**: Emergency Room Visits\\n- _Included?_: Yes - Policy covers emergency medical expenses...",
#         "document_title": "Comprehensive Health Insurance Policy",
#         "document_id": "507f1f77bcf86cd799439011",
#         "total_chunks": 15,
#         "total_tokens": 45000,
#         "processing_time_seconds": 45.2,
#         "token_usage": {
#             "total_cost": 0.0234,
#             "totals": {
#                 "total_tokens": 1250
#             }
#         },
#         "pdf_url": "https://example.com/health-policy.pdf"
#     }
#     ```
#     """
#     try:
#         start_time = time.time()
        
#         # Initialize the policy analyzer
#         analyzer = PolicyScenarioAnalyzer(collection_name=request.collection_name)
        
#         try:
#             # Initialize clients
#             await analyzer.initialize()
#             logger.info(f"Policy analyzer initialized for URL: {request.pdf_url}")
            
#             # Search for document first to get metadata
#             document = analyzer.search_document_by_url(request.pdf_url)
#             if not document:
#                 raise HTTPException(
#                     status_code=404, 
#                     detail=f"Policy document not found for URL: {request.pdf_url}"
#                 )
            
#             # Extract document metadata
#             doc_title = document.get("title", "Unknown Policy")
#             doc_id = str(document.get("_id", ""))
            
#             # Prepare document chunks to get token count
#             document_chunks = analyzer.prepare_document_chunks(document)
#             if not document_chunks:
#                 raise HTTPException(
#                     status_code=500, 
#                     detail="Failed to process policy document content"
#                 )
            
#             # Calculate total tokens
#             total_tokens = sum(analyzer.chunker.count_tokens(chunk) for chunk in document_chunks)
            
#             logger.info(f"Starting analysis for document: {doc_title} (ID: {doc_id})")
            
#             # Perform the analysis
#             analysis_result = await analyzer.analyze_policy(request.pdf_url)
            
#             if not analysis_result:
#                 raise HTTPException(
#                     status_code=500, 
#                     detail="Failed to generate policy analysis. Check server logs for details."
#                 )
            
#             # Calculate processing time
#             processing_time = time.time() - start_time
            
#             # Get token usage summary
#             token_usage = analyzer.get_token_usage_summary()
            
#             # Prepare response
#             response = PolicyAnalyzerResponse(
#                 analysis=analysis_result,
#                 document_title=doc_title,
#                 document_id=doc_id,
#                 total_chunks=len(document_chunks),
#                 total_tokens=total_tokens,
#                 processing_time_seconds=round(processing_time, 2),
#                 token_usage=token_usage,
#                 pdf_url=request.pdf_url
#             )
            
#             logger.info(f"Policy analysis completed in {processing_time:.2f}s for {doc_title}")
#             return response
            
#         finally:
#             # Always clean up resources
#             analyzer.close()
            
#     except HTTPException:
#         # Re-raise HTTP exceptions
#         raise
#     except Exception as e:
#         logger.error(f"Error in policy analyzer endpoint: {e}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Failed to analyze policy: {str(e)}"
#         )

# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint.
    
#     **Output:**
#     - `status` (string): "healthy" if service is running
#     - `agent_initialized` (boolean): Whether the insurance agent is ready
#     - `policy_recommender_initialized` (boolean): Whether the policy recommender is ready
#     - `active_policy_chat_sessions` (int): Number of active policy chat sessions
#     """
#     global agent, policy_recommender, policy_chat_instances
#     return {
#         "status": "healthy",
#         "agent_initialized": agent is not None,
#         "policy_recommender_initialized": policy_recommender is not None,
#         "active_policy_chat_sessions": len(policy_chat_instances)
#     }

# @app.get("/")
# async def root():
#     """
#     Root endpoint with API information.
    
#     **Output:**
#     - Basic API information and available endpoints
#     """
#     return {
#         "message": "AI Insurance Agent API",
#         "version": "1.0.0",
#         "endpoints": {
#             "/chat": "POST - Chat with the insurance agent",
#             "/chat-with-policy": "POST - Chat with a specific policy document",
#             "/recommend-policies": "POST - Get AI-powered policy recommendations",
#             "/policy-analyzer": "POST - Analyze a policy document",
#             "/health": "GET - Health check",
#             "/docs": "GET - API documentation"
#         }
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


