from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
import json

# Import from insurance_agent.py
from insurance_recommender_chat import InsuranceAgent, UserProfile
from clients.gemini_client import ChatMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Insurance Agent API",
    description="API for AI-powered insurance policy recommendations",
    version="1.0.0"
)

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the insurance agent on startup."""
    global agent
    try:
        agent = InsuranceAgent()
        logger.info("Insurance agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize insurance agent: {e}")
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
    """
    global agent
    return {
        "status": "healthy",
        "agent_initialized": agent is not None
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
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")