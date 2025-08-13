from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
import json

# Import from insurance_agent.py
from insurance_recommender_chat import InsuranceAgent, UserProfile
from clients.gemini_client import ChatMessage

# Import from insurance_recommender_keywords.py
from insurance_recommender_keywords import PolicyRecommender, UserProfile as KeywordUserProfile, PolicyRecommendation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Insurance Agent API",
    description="API for AI-powered insurance policy recommendations",
    version="1.0.0"
)

# Global agent instances
agent = None
policy_recommender = None

@app.on_event("startup")
async def startup_event():
    """Initialize the insurance agent and policy recommender on startup."""
    global agent, policy_recommender
    try:
        agent = InsuranceAgent()
        policy_recommender = PolicyRecommender()
        logger.info("Insurance agent and policy recommender initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str  # "user" or "assistant"
    content: str

class UserProfileResponse(BaseModel):
    """User profile information."""
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    income_range: Optional[str] = None
    family_size: Optional[int] = None
    existing_conditions: List[str] = []
    current_insurance: Optional[str] = None
    budget: Optional[str] = None
    coverage_preferences: List[str] = []
    location: Optional[str] = None
    lifestyle: Optional[str] = None
    profile_summary: str

class PolicyInfo(BaseModel):
    """Enhanced policy information from search results."""
    company_name: str
    title: str
    type: str
    sub_type: str
    pdf_link: str
    keywords: List[str]
    text_snippet: str
    relevance_score: float
    chunk_count: int = 1

# New models for policy recommendation endpoint
class KeywordUserProfileRequest(BaseModel):
    """User profile for policy recommendations."""
    age: int
    gender: str
    location: str
    income_range: str
    family_status: str
    dependents: int
    health_conditions: List[str] = []
    occupation: str
    lifestyle: str
    risk_tolerance: str
    budget_range: str
    coverage_preferences: List[str]
    additional_notes: str = ""

class PolicyRecommendationResponse(BaseModel):
    """Enhanced policy recommendation response."""
    policy_content: str
    company_name: str
    policy_type: str
    policy_sub_type: str
    policy_title: str
    pdf_link: str
    keywords: List[str]
    relevance_score: float
    gemini_reasoning: str
    qdrant_score: float
    final_rank: int

class RecommendationsRequest(BaseModel):
    """Request model for policy recommendations."""
    user_profile: KeywordUserProfileRequest
    top_k: int = 20
    top_n: int = 5
    score_threshold: float = 0.7

class RecommendationsResponse(BaseModel):
    """Response model for policy recommendations."""
    recommendations: List[PolicyRecommendationResponse]
    total_policies_found: int
    search_query_used: str
    processing_time_seconds: Optional[float] = None

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    conversation_history: List[ChatMessage] = []
    user_profile: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    updated_profile: UserProfileResponse
    search_performed: bool
    search_reasoning: Optional[str] = None
    policies_found: List[PolicyInfo] = []
    conversation_id: Optional[str] = None

def convert_profile_to_response(profile: UserProfile) -> UserProfileResponse:
    """Convert UserProfile to UserProfileResponse."""
    return UserProfileResponse(
        age=profile.age,
        gender=profile.gender,
        occupation=profile.occupation,
        income_range=profile.income_range,
        family_size=profile.family_size,
        existing_conditions=profile.existing_conditions or [],
        current_insurance=profile.current_insurance,
        budget=profile.budget,
        coverage_preferences=profile.coverage_preferences or [],
        location=profile.location,
        lifestyle=profile.lifestyle,
        profile_summary=profile.get_profile_summary()
    )

def convert_policies_to_response(policies: List[Dict[str, Any]]) -> List[PolicyInfo]:
    """Convert policy search results to PolicyInfo list with grouping by title."""
    if not policies:
        return []
    
    # Group policies by title and company to avoid duplicates
    policy_groups = {}
    for policy in policies:
        title = policy.get("title", "Unknown Policy")
        company_name = policy.get("company_name", "Unknown Company")
        pdf_link = policy.get("pdf_link", "")
        
        key = f"{company_name}_{title}_{pdf_link}"
        
        if key not in policy_groups:
            policy_groups[key] = {
                "company_name": company_name,
                "title": title,
                "type": policy.get("type", ""),
                "sub_type": policy.get("sub_type", ""),
                "pdf_link": pdf_link,
                "all_keywords": set(),
                "all_text": [],
                "best_score": 0,
                "chunk_count": 0
            }
        
        # Accumulate data from all chunks
        policy_groups[key]["all_keywords"].update(policy.get("keywords", []))
        policy_groups[key]["all_text"].append(policy.get("text", ""))
        policy_groups[key]["best_score"] = max(policy_groups[key]["best_score"], policy.get("score", 0))
        policy_groups[key]["chunk_count"] += 1
    
    # Convert to PolicyInfo objects
    policy_list = []
    for group_data in policy_groups.values():
        # Combine text from all chunks, taking first 300 chars
        combined_text = " ".join(group_data["all_text"])
        text_snippet = combined_text[:300] + "..." if len(combined_text) > 300 else combined_text
        
        policy_info = PolicyInfo(
            company_name=group_data["company_name"],
            title=group_data["title"],
            type=group_data["type"],
            sub_type=group_data["sub_type"],
            pdf_link=group_data["pdf_link"],
            keywords=list(group_data["all_keywords"])[:10],  # Limit to top 10 keywords
            text_snippet=text_snippet,
            relevance_score=group_data["best_score"],
            chunk_count=group_data["chunk_count"]
        )
        policy_list.append(policy_info)
    
    # Sort by relevance score
    policy_list.sort(key=lambda x: x.relevance_score, reverse=True)
    return policy_list

def convert_keyword_recommendations_to_response(recommendations: List[PolicyRecommendation]) -> List[PolicyRecommendationResponse]:
    """Convert PolicyRecommendation objects to PolicyRecommendationResponse."""
    return [
        PolicyRecommendationResponse(
            policy_content=rec.policy_content,
            company_name=rec.company_name,
            policy_type=rec.policy_type,
            policy_sub_type=rec.policy_sub_type,
            policy_title=rec.policy_title,
            pdf_link=rec.pdf_link,
            keywords=rec.keywords,
            relevance_score=rec.relevance_score,
            gemini_reasoning=rec.gemini_reasoning,
            qdrant_score=rec.qdrant_score,
            final_rank=rec.final_rank
        )
        for rec in recommendations
    ]

@app.post("/recommend-policies", response_model=RecommendationsResponse)
async def recommend_policies(request: RecommendationsRequest):
    """
    Get AI-powered policy recommendations based on user profile.
    
    **Input:**
    - `user_profile` (object): Comprehensive user profile information
        - `age` (int): User's age
        - `gender` (string): User's gender
        - `location` (string): User's location
        - `income_range` (string): User's income range
        - `family_status` (string): Single, married, divorced, etc.
        - `dependents` (int): Number of dependents
        - `health_conditions` (list): List of health conditions
        - `occupation` (string): User's occupation
        - `lifestyle` (string): Active, sedentary, etc.
        - `risk_tolerance` (string): Low, medium, high
        - `budget_range` (string): Budget range for insurance
        - `coverage_preferences` (list): Types of insurance interested in
        - `additional_notes` (string): Any additional information
    - `top_k` (int, optional): Number of policies to retrieve from vector DB (default: 20)
    - `top_n` (int, optional): Number of final recommendations to return (default: 5)
    - `score_threshold` (float, optional): Minimum similarity score (default: 0.7)
    
    **Output:**
    - `recommendations` (list): List of top policy recommendations
        - Each recommendation includes company, title, type, keywords, relevance score, reasoning, etc.
    - `total_policies_found` (int): Total number of policies found in search
    - `search_query_used` (string): The optimized search query generated by AI
    - `processing_time_seconds` (float): Time taken to process the request
    
    **Example Request:**
    ```json
    {
        "user_profile": {
            "age": 35,
            "gender": "male",
            "location": "Delhi, India",
            "income_range": "20-22 LPA",
            "family_status": "married",
            "dependents": 2,
            "health_conditions": ["diabetes"],
            "occupation": "software engineer",
            "lifestyle": "sedentary",
            "risk_tolerance": "medium",
            "budget_range": "30-40K per month",
            "coverage_preferences": ["health", "life"],
            "additional_notes": "Looking for family coverage"
        },
        "top_k": 15,
        "top_n": 3
    }
    ```
    
    **Example Response:**
    ```json
    {
        "recommendations": [
            {
                "policy_content": "Comprehensive health coverage with diabetes management...",
                "company_name": "HDFC ERGO",
                "policy_type": "health",
                "policy_sub_type": "family",
                "policy_title": "Optima Secure Health Policy",
                "pdf_link": "https://example.com/policy.pdf",
                "keywords": ["health", "diabetes", "family"],
                "relevance_score": 8.5,
                "gemini_reasoning": "This policy is highly suitable for a 35-year-old software engineer...",
                "qdrant_score": 0.92,
                "final_rank": 1
            }
        ],
        "total_policies_found": 15,
        "search_query_used": "35-year-old male software engineer Delhi diabetes family health insurance...",
        "processing_time_seconds": 12.5
    }
    ```
    """
    global policy_recommender
    
    if not policy_recommender:
        raise HTTPException(status_code=500, detail="Policy recommender not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Convert request profile to KeywordUserProfile
        user_profile = KeywordUserProfile(
            age=request.user_profile.age,
            gender=request.user_profile.gender,
            location=request.user_profile.location,
            income_range=request.user_profile.income_range,
            family_status=request.user_profile.family_status,
            dependents=request.user_profile.dependents,
            health_conditions=request.user_profile.health_conditions,
            occupation=request.user_profile.occupation,
            lifestyle=request.user_profile.lifestyle,
            risk_tolerance=request.user_profile.risk_tolerance,
            budget_range=request.user_profile.budget_range,
            coverage_preferences=request.user_profile.coverage_preferences,
            additional_notes=request.user_profile.additional_notes
        )
        
        # Generate optimized search query first to include in response
        search_query = await policy_recommender.generate_optimized_search_query(user_profile)
        
        # Get policy recommendations
        recommendations = await policy_recommender.recommend_policies(
            user_profile=user_profile,
            top_k=request.top_k,
            top_n=request.top_n,
            score_threshold=request.score_threshold
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Convert recommendations to response format
        response_recommendations = convert_keyword_recommendations_to_response(recommendations)
        
        # Prepare response
        response = RecommendationsResponse(
            recommendations=response_recommendations,
            total_policies_found=len(recommendations),
            search_query_used=search_query,
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.info(f"Policy recommendations completed in {processing_time:.2f}s - Returned {len(recommendations)} recommendations")
        return response
        
    except Exception as e:
        logger.error(f"Error in policy recommendation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the AI Insurance Agent.
    
    **Input:**
    - `message` (string): User's message/question
    - `conversation_history` (list, optional): Previous messages in the conversation
        - Each message has `role` ("user" or "assistant") and `content` (string)
    - `user_profile` (dict, optional): Current user profile information
        - Contains fields like age, gender, occupation, existing_conditions, etc.
    
    **Output:**
    - `response` (string): Agent's response to the user
    - `updated_profile` (object): Updated user profile after processing the message
        - Contains all profile fields plus a `profile_summary` string
    - `search_performed` (boolean): Whether policy database search was performed
    - `search_reasoning` (string, optional): Explanation for search decision
    - `policies_found` (list): List of relevant policies if search was performed
        - Each policy contains company_name, title, type, keywords, pdf_link, text_snippet, relevance_score, and chunk_count
    - `conversation_id` (string, optional): Identifier for conversation tracking
    
    **Example Request:**
    ```json
    {
        "message": "I'm 30 years old and need health insurance",
        "conversation_history": [
            {"role": "assistant", "content": "Hello! How can I help you with insurance?"},
            {"role": "user", "content": "Hi, I need insurance advice"}
        ],
        "user_profile": {
            "age": null,
            "occupation": "software engineer"
        }
    }
    ```
    
    **Example Response:**
    ```json
    {
        "response": "Great! As a 30-year-old software engineer, I can help you find suitable health insurance...",
        "updated_profile": {
            "age": 30,
            "occupation": "software engineer",
            "existing_conditions": [],
            "profile_summary": "Age: 30; Occupation: software engineer"
        },
        "search_performed": true,
        "search_reasoning": "User provided age and requested health insurance recommendations",
        "policies_found": [
            {
                "company_name": "HDFC ERGO",
                "title": "Optima Secure Health Policy",
                "type": "health",
                "sub_type": "individual",
                "pdf_link": "https://example.com/policy.pdf",
                "keywords": ["health", "young adults", "comprehensive"],
                "text_snippet": "This policy offers comprehensive health coverage...",
                "relevance_score": 0.85,
                "chunk_count": 3
            }
        ]
    }
    ```
    """
    global agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Insurance agent not initialized")
    
    try:
        # Initialize user profile
        if request.user_profile:
            user_profile = UserProfile(**request.user_profile)
        else:
            user_profile = UserProfile()
        
        # Convert conversation history
        conversation_history = [msg.content for msg in request.conversation_history]
        
        # Update user profile using AI
        user_profile = await agent.update_user_profile(
            request.message, conversation_history, user_profile
        )
        
        # Let AI decide if we should search policies
        search_decision = await agent.should_search_policies(
            request.message, conversation_history, user_profile
        )
        
        # Search policies if needed
        policies = []
        search_performed = search_decision.get("should_search", False)
        search_reasoning = search_decision.get("reasoning", None)
        
        if search_performed:
            search_query = search_decision.get("search_query", request.message)
            policies = await agent.search_policies(search_query, top_k=15)  # Get more chunks for better grouping
            logger.info(f"Found {len(policies)} policy chunks for query: {search_query}")
        
        # Get enhanced message with proper context
        enhanced_message = await agent.get_enhanced_response(
            request.message, user_profile, conversation_history, policies
        )
        
        # Generate response using AI
        try:
            response = await agent.gemini_client.generate_with_thinking(
                prompt=enhanced_message,
                model=agent.model,
                thinking_budget=agent.thinking_budget,
                include_thoughts=False,
                temperature=0.7
            )
            
            # Check if response is valid
            if not response or not response.text:
                logger.error("Received empty response from Gemini")
                raise HTTPException(status_code=500, detail="Failed to generate response")
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
        
        # Prepare response
        chat_response = ChatResponse(
            response=response.text,
            updated_profile=convert_profile_to_response(user_profile),
            search_performed=search_performed,
            search_reasoning=search_reasoning,
            policies_found=convert_policies_to_response(policies)
        )
        
        logger.info(f"Chat completed - Profile: {user_profile.get_profile_summary()}")
        logger.info(f"Found {len(chat_response.policies_found)} unique policies from {len(policies)} chunks")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    **Output:**
    - `status` (string): "healthy" if service is running
    - `agent_initialized` (boolean): Whether the insurance agent is ready
    - `policy_recommender_initialized` (boolean): Whether the policy recommender is ready
    """
    global agent, policy_recommender
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "policy_recommender_initialized": policy_recommender is not None
    }

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    **Output:**
    - Basic API information and available endpoints
    """
    return {
        "message": "AI Insurance Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Chat with the insurance agent",
            "/recommend-policies": "POST - Get AI-powered policy recommendations",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")