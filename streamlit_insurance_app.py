import streamlit as st
import requests
import json
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Insurance Agent",
    page_icon="🏥",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8001"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

if "api_available" not in st.session_state:
    st.session_state.api_available = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_messages": 0,
        "searches_performed": 0,
        "policies_found_total": 0,
        "unique_policies_found": 0
    }

def check_api_health():
    """Check if the API is available and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result.get("agent_initialized", False)
        return False
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False

def send_chat_message(message: str, conversation_history: List[Dict], user_profile: Dict) -> Dict:
    """Send message to the chat API endpoint."""
    try:
        request_data = {
            "message": message,
            "conversation_history": conversation_history,
            "user_profile": user_profile
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            return {
                "response": f"Sorry, I encountered an error (Status: {response.status_code}). Please try again.",
                "updated_profile": {"profile_summary": "Error occurred"},
                "search_performed": False,
                "policies_found": []
            }
            
    except requests.exceptions.Timeout:
        return {
            "response": "Sorry, the request timed out. Please try again with a shorter message.",
            "updated_profile": {"profile_summary": "Request timeout"},
            "search_performed": False,
            "policies_found": []
        }
    except Exception as e:
        logger.error(f"Error sending chat message: {e}")
        return {
            "response": f"Sorry, I encountered an error: {str(e)}. Please try again.",
            "updated_profile": {"profile_summary": "Error occurred"},
            "search_performed": False,
            "policies_found": []
        }

def format_policy_card(policy: Dict, index: int) -> None:
    """Format and display a policy as a card."""
    with st.container():
        # Policy header with company and title
        st.markdown(f"**{index}. {policy['company_name']} - {policy['title']}**")
        
        # Policy details in columns
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Type:** {policy['type']}" + (f" - {policy['sub_type']}" if policy['sub_type'] else ""))
            st.markdown(f"**Relevance Score:** {policy['relevance_score']:.2f}")
            
        with col2:
            st.markdown(f"**Chunks Found:** {policy.get('chunk_count', 1)}")
            
        with col3:
            if policy.get('pdf_link'):
                st.markdown(f"[View Policy Document]({policy['pdf_link']})")
        
        # Keywords
        if policy.get('keywords'):
            keywords_text = ", ".join(policy['keywords'][:8])  # Show first 8 keywords
            if len(policy['keywords']) > 8:
                keywords_text += f" (+{len(policy['keywords']) - 8} more)"
            st.markdown(f"**Keywords:** {keywords_text}")
        
        # Content preview
        if policy.get('text_snippet'):
            with st.expander("Preview Policy Content"):
                st.markdown(policy['text_snippet'])
        
        st.markdown("---")

def main():
    st.title("AI Insurance Agent")
    st.markdown("*Your personal AI-powered insurance advisor*")
    st.markdown("---")
    
    # Check API availability
    if st.session_state.api_available is None:
        with st.spinner("Checking API connection..."):
            st.session_state.api_available = check_api_health()
    
    if not st.session_state.api_available:
        st.error("API Service Unavailable")
        st.markdown("""
        The insurance agent API is not available. Please ensure:
        1. The FastAPI server is running on `http://localhost:8001`
        2. All dependencies are installed
        3. The insurance agent is properly initialized
        """)
        
        if st.button("Retry Connection"):
            st.session_state.api_available = None
            st.rerun()
        return
    
    # Add initial message if empty
    if not st.session_state.messages:
        initial_message = {
            "role": "assistant", 
            "content": """Hello! I'm your personal insurance advisor. I'm here to help you find the perfect insurance policy that matches your specific needs and budget.

To give you the best recommendations, I'll need to understand your situation better. Let's start with some basic information:

1. What's your age and current occupation?
2. Are you looking for insurance for yourself or your family?
3. Do you have any existing health conditions I should know about?
4. What type of insurance coverage are you most interested in?

Feel free to share as much or as little as you're comfortable with. We can always discuss more details as we go along!"""
        }
        st.session_state.messages.append(initial_message)
    
    # Create main chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show additional info for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Show search info
                    if metadata.get("search_performed"):
                        st.info(f"Policy Search Performed: {metadata.get('search_reasoning', 'Policy database searched')}")
                    
                    # Show policies found with enhanced display
                    if metadata.get("policies_found"):
                        policies = metadata["policies_found"]
                        
                        with st.expander(f"Found {len(policies)} Relevant Insurance Policies", expanded=False):
                            st.markdown("**Policies ranked by relevance to your needs:**")
                            st.markdown("")
                            
                            for i, policy in enumerate(policies, 1):
                                format_policy_card(policy, i)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat immediately
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Update conversation history for API
        st.session_state.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your message and searching for relevant policies..."):
                # Send to API
                api_response = send_chat_message(
                    prompt, 
                    st.session_state.conversation_history[:-1],  # Don't include the current message
                    st.session_state.user_profile
                )
                
                # Extract response data
                agent_response = api_response.get("response", "Sorry, I couldn't process your request.")
                updated_profile = api_response.get("updated_profile", {})
                search_performed = api_response.get("search_performed", False)
                search_reasoning = api_response.get("search_reasoning")
                policies_found = api_response.get("policies_found", [])
                
                # Display main response
                st.markdown(agent_response)
                
                # Show search information
                if search_performed:
                    st.info(f"Policy Search Performed: {search_reasoning}")
                
                # Show policies if found with enhanced display
                if policies_found:
                    with st.expander(f"Found {len(policies_found)} Relevant Insurance Policies", expanded=True):
                        st.markdown("**Policies ranked by relevance to your needs:**")
                        st.markdown("")
                        
                        for i, policy in enumerate(policies_found, 1):
                            format_policy_card(policy, i)
                
                # Update session state
                st.session_state.user_profile = {
                    k: v for k, v in updated_profile.items() 
                    if k != 'profile_summary'
                }
                
                # Add assistant response to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": agent_response
                })
                
                # Add to messages with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": agent_response,
                    "metadata": {
                        "search_performed": search_performed,
                        "search_reasoning": search_reasoning,
                        "policies_found": policies_found
                    }
                }
                st.session_state.messages.append(assistant_message)
                
                # Update session stats
                st.session_state.session_stats["total_messages"] += 1
                if search_performed:
                    st.session_state.session_stats["searches_performed"] += 1
                st.session_state.session_stats["policies_found_total"] += sum(
                    policy.get('chunk_count', 1) for policy in policies_found
                )
                st.session_state.session_stats["unique_policies_found"] += len(policies_found)
    
    # Sidebar with profile and stats
    with st.sidebar:
        st.header("Your Profile")
        
        profile_summary = st.session_state.user_profile.get("profile_summary", "")
        if not profile_summary and st.session_state.messages:
            # Try to get from latest updated profile
            try:
                if st.session_state.conversation_history:
                    temp_response = send_chat_message(
                        "Please summarize my current profile", 
                        st.session_state.conversation_history[-2:], 
                        st.session_state.user_profile
                    )
                    profile_summary = temp_response.get("updated_profile", {}).get("profile_summary", "")
            except:
                pass
        
        if profile_summary and profile_summary != "No profile information collected yet":
            st.text_area("Profile Information", profile_summary, height=150, disabled=True)
            
            # Show individual profile fields
            with st.expander("Detailed Profile", expanded=False):
                profile = st.session_state.user_profile
                if profile.get("age"):
                    st.write(f"**Age:** {profile['age']}")
                if profile.get("occupation"):
                    st.write(f"**Occupation:** {profile['occupation']}")
                if profile.get("location"):
                    st.write(f"**Location:** {profile['location']}")
                if profile.get("family_size"):
                    st.write(f"**Family Size:** {profile['family_size']}")
                if profile.get("budget"):
                    st.write(f"**Budget:** {profile['budget']}")
                if profile.get("existing_conditions"):
                    st.write(f"**Health Conditions:** {', '.join(profile['existing_conditions'])}")
        else:
            st.info("Profile information will appear here as we chat")
        
        st.header("Session Statistics")
        stats = st.session_state.session_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats["total_messages"])
            st.metric("Searches", stats["searches_performed"])
        with col2:
            st.metric("Unique Policies", stats["unique_policies_found"])
            st.metric("Total Chunks", stats["policies_found_total"])
        
        if stats["total_messages"] > 0:
            search_rate = (stats["searches_performed"] / stats["total_messages"]) * 100
            st.metric("Search Rate", f"{search_rate:.0f}%")
        
        # API Status
        st.header("Connection Status")
        if st.session_state.api_available:
            st.success("API Connected")
        else:
            st.error("API Disconnected")
        
        # Action buttons
        st.header("Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh API", help="Check API connection"):
                st.session_state.api_available = None
                st.rerun()
        
        with col2:
            if st.button("Clear Chat", help="Start new conversation"):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.session_state.user_profile = {}
                st.session_state.session_stats = {
                    "total_messages": 0,
                    "searches_performed": 0,
                    "policies_found_total": 0,
                    "unique_policies_found": 0
                }
                st.rerun()
        
        # Export conversation
        if st.session_state.messages and st.button("Export Chat", help="Download conversation"):
            chat_export = {
                "conversation": st.session_state.messages,
                "user_profile": st.session_state.user_profile,
                "session_stats": st.session_state.session_stats
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(chat_export, indent=2),
                file_name="insurance_chat_export.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()