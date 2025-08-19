import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass, asdict

from clients.gemini_client import GeminiClient, ChatSession, ChatMessage
from clients.openai_client import OpenAIClient
from clients.qdrant_client import get_qdrant_client, search_points_in_collection
from clients.token_tracker import TokenTracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for insurance recommendations."""
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    income_range: Optional[str] = None
    family_size: Optional[int] = None
    existing_conditions: List[str] = None
    current_insurance: Optional[str] = None
    budget: Optional[str] = None
    coverage_preferences: List[str] = None
    location: Optional[str] = None
    lifestyle: Optional[str] = None
    
    def __post_init__(self):
        if self.existing_conditions is None:
            self.existing_conditions = []
        if self.coverage_preferences is None:
            self.coverage_preferences = []
    
    def get_profile_summary(self) -> str:
        """Get a summary of the user profile for context."""
        profile_parts = []
        
        if self.age:
            profile_parts.append(f"Age: {self.age}")
        if self.gender:
            profile_parts.append(f"Gender: {self.gender}")
        if self.occupation:
            profile_parts.append(f"Occupation: {self.occupation}")
        if self.income_range:
            profile_parts.append(f"Income: {self.income_range}")
        if self.family_size:
            profile_parts.append(f"Family size: {self.family_size}")
        if self.existing_conditions:
            profile_parts.append(f"Health conditions: {', '.join(self.existing_conditions)}")
        if self.current_insurance:
            profile_parts.append(f"Current insurance: {self.current_insurance}")
        if self.budget:
            profile_parts.append(f"Budget: {self.budget}")
        if self.coverage_preferences:
            profile_parts.append(f"Coverage preferences: {', '.join(self.coverage_preferences)}")
        if self.location:
            profile_parts.append(f"Location: {self.location}")
        if self.lifestyle:
            profile_parts.append(f"Lifestyle: {self.lifestyle}")
        
        return "; ".join(profile_parts) if profile_parts else "No profile information collected yet"

def clean_json_response(response_text: str) -> str:
    """Clean JSON response by removing markdown formatting."""
    response_text = response_text.strip()
    
    # Remove markdown code blocks
    if response_text.startswith('```json'):
        response_text = response_text.replace('```json', '').replace('```', '').strip()
    elif response_text.startswith('```'):
        response_text = response_text.replace('```', '').strip()
    
    # Remove any leading/trailing whitespace or newlines
    response_text = response_text.strip()
    
    return response_text

class InsuranceAgent:
    """AI Insurance Agent that helps users find the best insurance policies."""
    
    def __init__(self):
        self.token_tracker = TokenTracker()
        self.gemini_client = GeminiClient(self.token_tracker)
        self.openai_client = OpenAIClient(self.token_tracker)
        self.qdrant_client = get_qdrant_client()
        
        # Collections - Updated to match the new collection names
        self.chunk_collection = "policy_chunks2"
        self.keyword_collection = "policy_keywords2"
        
        # Agent configuration
        self.model = "gemini-2.5-flash"
        self.thinking_budget = 2048
        
        # System prompt for the insurance agent
        self.system_prompt = """You are an expert insurance advisor helping users find the best insurance policy for their specific needs. 

CRITICAL RULES:
1. NEVER suggest or recommend any insurance policy that is not from the provided database search results
2. ONLY recommend policies that are retrieved from the policy database search
3. If no relevant policies are found in the database, inform the user and ask for more specific requirements

Your workflow:
1. GATHER INFORMATION: Ask relevant questions to understand the user's profile, needs, and preferences
2. DECIDE TO SEARCH: When you have enough information or when user asks for recommendations, use the SEARCH_POLICIES function
3. ANALYZE DATABASE RESULTS: Analyze ONLY the retrieved policies from the database and recommend the best options
4. UPDATE PROFILE: Use the UPDATE_PROFILE function to maintain accurate user information

Key areas to explore:
- Personal details (age, gender, occupation, income, family size)
- Health status and existing conditions
- Current insurance coverage
- Budget constraints
- Coverage preferences (health, life, accident, critical illness, etc.)
- Lifestyle factors
- Location/state (for regulatory compliance)

Available Functions:
- SEARCH_POLICIES: Use when you need to find relevant insurance policies from the database
- UPDATE_PROFILE: Use to update user profile information based on the conversation

Be conversational, empathetic, and thorough. Always base recommendations on actual policies from the database."""
    
    async def search_policies(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant policies using vector similarity."""
        try:
            # Generate embedding for the query
            query_embedding = await self.openai_client.generate_embedding(
                text=query,
                model="text-embedding-3-large"
            )
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Search in policy chunks collection
            results = search_points_in_collection(
                client=self.qdrant_client,
                collection_name=self.chunk_collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=0.6,
                with_payload=True
            )
            
            # Extract and format results with enhanced policy information
            policies = []
            for result in results:
                policy_info = {
                    "text": result.payload.get("chunk", ""),
                    "keywords": result.payload.get("keywords", []),
                    "company_name": result.payload.get("company_name", ""),
                    "type": result.payload.get("type", ""),
                    "sub_type": result.payload.get("sub_type", ""),
                    "title": result.payload.get("title", ""),
                    "pdf_link": result.payload.get("pdf_link", ""),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "score": result.score
                }
                policies.append(policy_info)
            
            logger.info(f"Found {len(policies)} relevant policy chunks")
            return policies
            
        except Exception as e:
            logger.error(f"Error searching policies: {e}")
            return []
    
    def format_policy_context(self, policies: List[Dict[str, Any]], user_profile: UserProfile) -> str:
        """Format policy search results for the AI context."""
        if not policies:
            return "No relevant policies found in the database."
        
        context_parts = [
            "=== POLICY DATABASE SEARCH RESULTS ===",
            f"User Profile: {user_profile.get_profile_summary()}",
            f"Found {len(policies)} relevant policy options:",
            ""
        ]
        
        # Group policies by title and pdf_link to avoid duplicates
        policy_groups = {}
        for policy in policies:
            title = policy.get('title', 'Unknown Policy')
            pdf_link = policy.get('pdf_link', '')
            key = f"{title}_{pdf_link}"
            
            if key not in policy_groups:
                policy_groups[key] = {
                    'title': title,
                    'company_name': policy.get('company_name', ''),
                    'type': policy.get('type', ''),
                    'sub_type': policy.get('sub_type', ''),
                    'pdf_link': pdf_link,
                    'chunks': [],
                    'all_keywords': set(),
                    'best_score': 0
                }
            
            policy_groups[key]['chunks'].append({
                'text': policy.get('text', ''),
                'keywords': policy.get('keywords', []),
                'score': policy.get('score', 0),
                'chunk_index': policy.get('chunk_index', 0)
            })
            policy_groups[key]['all_keywords'].update(policy.get('keywords', []))
            policy_groups[key]['best_score'] = max(policy_groups[key]['best_score'], policy.get('score', 0))
        
        # Sort by best score
        sorted_policies = sorted(policy_groups.values(), key=lambda x: x['best_score'], reverse=True)
        
        for i, policy_group in enumerate(sorted_policies, 1):
            context_parts.extend([
                f"POLICY {i}: {policy_group['title']}",
                f"Company: {policy_group['company_name']}",
                f"Type: {policy_group['type']}" + (f" - {policy_group['sub_type']}" if policy_group['sub_type'] else ""),
                f"Document Link: {policy_group['pdf_link']}",
                f"Relevance Score: {policy_group['best_score']:.3f}",
                f"Keywords: {', '.join(list(policy_group['all_keywords'])[:10])}",
                "",
                "Relevant Content Sections:"
            ])
            
            # Show top 3 most relevant chunks for this policy
            top_chunks = sorted(policy_group['chunks'], key=lambda x: x['score'], reverse=True)[:3]
            for j, chunk in enumerate(top_chunks, 1):
                content_preview = chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text']
                context_parts.extend([
                    f"  Section {j} (Score: {chunk['score']:.3f}): {content_preview}",
                    ""
                ])
            
            context_parts.append("-" * 60)
        
        return "\n".join(context_parts)
    
    async def should_search_policies(self, user_message: str, conversation_history: List[str], user_profile: UserProfile) -> Dict[str, Any]:
        """Let AI decide if we should search policies and what to search for."""
        try:
            decision_prompt = f"""
Based on the conversation context, determine if we should search the insurance policy database now.

CONVERSATION HISTORY:
{chr(10).join(conversation_history[-6:]) if conversation_history else "No previous conversation"}

CURRENT USER MESSAGE: {user_message}

CURRENT USER PROFILE: {user_profile.get_profile_summary()}

Respond with a JSON object containing:
{{
    "should_search": true/false,
    "search_query": "specific search terms if should_search is true",
    "reasoning": "brief explanation of your decision"
}}

Search when:
- User asks for policy recommendations or comparisons
- User has provided enough information for meaningful recommendations
- User asks about specific coverage types
- You have sufficient profile information to make targeted searches

Don't search when:
- Still gathering basic information
- User is asking general questions about insurance concepts
- Insufficient profile information for meaningful results

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no code blocks, no additional text.
"""

            response = await self.gemini_client.generate_with_thinking(
                prompt=decision_prompt,
                model="gemini-2.5-flash",
                thinking_budget=512,
                include_thoughts=False,
                temperature=0.1
            )
            
            # Check if response is valid
            if not response or not response.text:
                logger.error("Received empty response for search decision")
                return {"should_search": False, "search_query": "", "reasoning": "Empty response"}
            
            # Parse JSON response
            try:
                response_text = clean_json_response(response.text)
                decision = json.loads(response_text)
                logger.info(f"Search decision: {decision}")
                return decision
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse search decision JSON: {response.text}")
                logger.error(f"JSON decode error: {e}")
                return {"should_search": False, "search_query": "", "reasoning": "Failed to parse decision"}
                
        except Exception as e:
            logger.error(f"Error in search decision: {e}")
            # Fallback: simple keyword-based decision
            search_triggers = ["recommend", "suggest", "best policy", "insurance", "coverage", "compare"]
            should_search_fallback = any(trigger in user_message.lower() for trigger in search_triggers)
            return {
                "should_search": should_search_fallback, 
                "search_query": user_message if should_search_fallback else "", 
                "reasoning": f"Fallback decision due to error: {e}"
            }
    
    async def update_user_profile(self, user_message: str, conversation_history: List[str], current_profile: UserProfile) -> UserProfile:
        """Let AI update the user profile based on conversation."""
        try:
            profile_update_prompt = f"""
Based on the conversation, extract and update the user's profile information.

CONVERSATION HISTORY:
{chr(10).join(conversation_history[-10:]) if conversation_history else "No previous conversation"}

CURRENT USER MESSAGE: {user_message}

CURRENT PROFILE: {json.dumps(asdict(current_profile), indent=2)}

Extract any new information and provide an updated profile as a JSON object with these fields:
{{
    "age": integer or null,
    "gender": "Male"/"Female" or null,
    "occupation": string or null,
    "income_range": string or null,
    "family_size": integer or null,
    "existing_conditions": [list of health conditions],
    "current_insurance": string or null,
    "budget": string or null,
    "coverage_preferences": [list of coverage types],
    "location": string or null,
    "lifestyle": string or null
}}

Only update fields where you have clear information. Keep existing values if no new information is provided.
IMPORTANT: Return ONLY the JSON object, no markdown formatting, no code blocks, no additional text.
"""

            response = await self.gemini_client.generate_with_thinking(
                prompt=profile_update_prompt,
                model="gemini-2.5-flash",
                thinking_budget=512,
                include_thoughts=False,
                temperature=0.1
            )
            
            # Check if response is valid
            if not response or not response.text:
                logger.error("Received empty response for profile update")
                return current_profile
            
            # Parse JSON response
            try:
                response_text = clean_json_response(response.text)
                updated_data = json.loads(response_text)
                
                # Update profile with new data
                for key, value in updated_data.items():
                    if hasattr(current_profile, key) and value is not None:
                        setattr(current_profile, key, value)
                
                logger.info(f"Profile updated: {current_profile.get_profile_summary()}")
                return current_profile
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse profile update JSON: {response.text}")
                logger.error(f"JSON decode error: {e}")
                return current_profile
                
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            # Return current profile unchanged as fallback
            return current_profile
    
    async def get_enhanced_response(self, user_message: str, user_profile: UserProfile, conversation_history: List[str], policies: List[Dict[str, Any]] = None) -> str:
        """Get AI response with policy context and strict database-only recommendations."""
        
        # Prepare context
        context_parts = [
            f"USER PROFILE: {user_profile.get_profile_summary()}",
            f"USER MESSAGE: {user_message}",
            ""
        ]
        
        # Add policy context if available
        if policies:
            policy_context = self.format_policy_context(policies, user_profile)
            context_parts.extend([
                "AVAILABLE POLICIES FROM DATABASE:",
                policy_context,
                "",
                "IMPORTANT: Base your recommendations ONLY on the above policies from the database.",
                "When referencing policies, mention the company name, policy title, and type.",
                ""
            ])
        else:
            context_parts.extend([
                "NO POLICIES RETRIEVED FROM DATABASE.",
                "Do not recommend any specific policies. Ask for more information to search the database.",
                ""
            ])
        
        # Add conversation context
        if conversation_history:
            context_parts.extend([
                "RECENT CONVERSATION:",
                "\n".join(conversation_history[-6:]),
                ""
            ])
        
        # Add system prompt
        context_parts.append(self.system_prompt)
        
        enhanced_message = "\n".join(context_parts)
        
        return enhanced_message
    
    async def run_agent(self):
        """Run the interactive insurance agent."""
        print("=" * 60)
        print("WELCOME TO YOUR AI INSURANCE ADVISOR")
        print("=" * 60)
        print("I'm here to help you find the best insurance policy for your needs.")
        print("Let's start by getting to know you better!")
        print("Type 'quit' to exit at any time.")
        print("-" * 60)
        
        # Initialize session and user profile
        session = self.gemini_client.create_chat_session()
        user_profile = UserProfile()
        
        # Initial agent message
        initial_message = f"""Hello! I'm your personal insurance advisor. I'm here to help you find the perfect insurance policy that matches your specific needs and budget.

        To give you the best recommendations, I'll need to understand your situation better. Let's start with some basic information:

        1. What's your age and current occupation?
        2. Are you looking for insurance for yourself or your family?
        3. Do you have any existing health conditions I should know about?
        4. What type of insurance coverage are you most interested in?

        Feel free to share as much or as little as you're comfortable with. We can always discuss more details as we go along!"""
        
        print("Agent:", initial_message)
        print()
        
        # Add initial message to session
        session.messages.append(ChatMessage(
            role="model",
            content=initial_message,
            thought_summary=None,
            thought_tokens=0,
            output_tokens=0
        ))
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nThank you for using the AI Insurance Advisor. Have a great day!")
                    break
                
                if not user_input:
                    continue
                
                # Update user profile using AI
                conversation_history = [msg.content for msg in session.messages]
                user_profile = await self.update_user_profile(user_input, conversation_history, user_profile)
                
                # Let AI decide if we should search policies
                search_decision = await self.should_search_policies(user_input, conversation_history, user_profile)
                
                policies = []
                if search_decision.get("should_search", False):
                    print(f"\n[Searching policy database: {search_decision.get('reasoning', 'No reason provided')}]")
                    search_query = search_decision.get("search_query", user_input)
                    policies = await self.search_policies(search_query, top_k=10)
                    
                    if not policies:
                        print("[No relevant policies found in database]")
                    else:
                        unique_policies = {}
                        for p in policies:
                            key = f"{p.get('title', '')}_{p.get('company_name', '')}"
                            if key not in unique_policies:
                                unique_policies[key] = p
                        print(f"[Found {len(unique_policies)} unique policies from {len(policies)} chunks]")
                
                # Get enhanced message with proper context
                enhanced_message = await self.get_enhanced_response(
                    user_input, user_profile, conversation_history, policies
                )
                
                print("\nAgent: ", end="", flush=True)
                
                # Stream the response
                full_response = ""
                thoughts = ""
                
                async for chunk, is_thought in self.gemini_client.stream_chat_with_thinking(
                    session=session,
                    user_message=enhanced_message,
                    model=self.model,
                    thinking_budget=self.thinking_budget,
                    include_thoughts=False,  # Hide thoughts from user for better UX
                    temperature=0.7
                ):
                    if is_thought:
                        thoughts += chunk
                    else:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                
                print("\n")
                
                # Log thinking for debugging (optional)
                if thoughts:
                    logger.info(f"Agent thinking: {thoughts[:200]}...")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print(f"Sorry, I encountered an error: {e}")
                print("Let's continue our conversation.")
        
        # Print session statistics
        stats = self.gemini_client.get_session_stats(session)
        print(f"\n=== Session Summary ===")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Thought tokens used: {stats['total_thought_tokens']}")
        print(f"Output tokens used: {stats['total_output_tokens']}")
        
        # Print final user profile
        print(f"\n=== Your Profile ===")
        print(user_profile.get_profile_summary())

async def main():
    """Main function to run the insurance agent."""
    try:
        agent = InsuranceAgent()
        await agent.run_agent()
    except Exception as e:
        logger.error(f"Failed to start insurance agent: {e}")
        print(f"Error starting the insurance agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())