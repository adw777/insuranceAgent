import streamlit as st
import requests
import json
from typing import List, Dict, Any
import logging
import uuid

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
if "api_available" not in st.session_state:
    st.session_state.api_available = None

# Chat tab states
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_user_profile" not in st.session_state:
    st.session_state.chat_user_profile = {}
if "chat_conversation_history" not in st.session_state:
    st.session_state.chat_conversation_history = []

# Policy chat tab states
if "policy_chat_sessions" not in st.session_state:
    st.session_state.policy_chat_sessions = {}
if "selected_policy_url" not in st.session_state:
    st.session_state.selected_policy_url = ""

# Form recommendations state
if "form_recommendations" not in st.session_state:
    st.session_state.form_recommendations = []

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
            return {
                "response": f"Error: {response.status_code}. Please try again.",
                "updated_profile": {"profile_summary": "Error occurred"},
                "search_performed": False,
                "policies_found": []
            }
            
    except Exception as e:
        logger.error(f"Error sending chat message: {e}")
        return {
            "response": f"Error: {str(e)}. Please try again.",
            "updated_profile": {"profile_summary": "Error occurred"},
            "search_performed": False,
            "policies_found": []
        }

def send_policy_recommendation_request(user_profile: Dict, top_k: int = 20, top_n: int = 5) -> Dict:
    """Send request to policy recommendation endpoint."""
    try:
        request_data = {
            "user_profile": user_profile,
            "top_k": top_k,
            "top_n": top_n,
            "score_threshold": 0.5
        }
        
        response = requests.post(
            f"{API_BASE_URL}/recommend-policies",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error getting policy recommendations: {e}")
        return {"error": str(e)}

def send_policy_chat_message(pdf_url: str, message: str, session_id: str = None) -> Dict:
    """Send message to policy chat endpoint."""
    try:
        request_data = {
            "pdf_url": pdf_url,
            "message": message,
            "session_id": session_id,
            "collection_name": "policies"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat-with-policy",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error in policy chat: {e}")
        return {"error": str(e)}

def send_policy_analysis_request(pdf_url: str) -> Dict:
    """Send request to policy analyzer endpoint."""
    try:
        request_data = {
            "pdf_url": pdf_url,
            "collection_name": "policies"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/policy-analyzer",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error getting policy analysis: {e}")
        return {"error": str(e)}

def format_policy_card(policy: Dict, index: int, show_actions: bool = False, key_prefix: str = ""):
    """Format and display a policy as a card."""
    with st.container():
        st.markdown(f"**{index}. {policy.get('company_name', 'Unknown')} - {policy.get('title', 'Unknown Policy')}**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Type:** {policy.get('type', 'N/A')}" + (f" - {policy.get('sub_type', '')}" if policy.get('sub_type') else ""))
            score = policy.get('relevance_score', policy.get('qdrant_score', 0))
            st.markdown(f"**Relevance Score:** {score:.2f}")
            
        with col2:
            st.markdown(f"**Chunks:** {policy.get('chunk_count', 1)}")
            
        with col3:
            pdf_link = policy.get('pdf_link', policy.get('pdf_url', ''))
            if pdf_link:
                st.markdown(f"[View Document]({pdf_link})")
        
        if policy.get('keywords'):
            keywords = policy['keywords'][:8]
            keywords_text = ", ".join(keywords)
            if len(policy.get('keywords', [])) > 8:
                keywords_text += f" (+{len(policy['keywords']) - 8} more)"
            st.markdown(f"**Keywords:** {keywords_text}")
        
        if policy.get('text_snippet') or policy.get('policy_content'):
            content = policy.get('text_snippet', policy.get('policy_content', ''))
            if len(content) > 300:
                content = content[:300] + "..."
            with st.expander("Preview Content"):
                st.markdown(content)
        
        if policy.get('gemini_reasoning'):
            with st.expander("AI Reasoning"):
                st.markdown(policy['gemini_reasoning'])
        
        if show_actions:
            col1, col2 = st.columns(2)
            pdf_link = policy.get('pdf_link', policy.get('pdf_url', ''))
            
            with col1:
                if st.button(f"Chat with Policy {index}", key=f"{key_prefix}_chat_{index}_{abs(hash(pdf_link))}"):
                    st.session_state.selected_policy_url = pdf_link
                    st.rerun()
            
            with col2:
                if st.button(f"Analyze Policy {index}", key=f"{key_prefix}_analyze_{index}_{abs(hash(pdf_link))}"):
                    st.session_state.selected_policy_url = pdf_link
                    st.rerun()
        
        st.markdown("---")

def chat_tab():
    """Chat recommendation tab."""
    st.header("Chat with AI Insurance Agent")
    st.markdown("Get personalized recommendations through conversation")
    
    # Add initial message if empty
    if not st.session_state.chat_messages:
        initial_message = {
            "role": "assistant", 
            "content": """Hello! I'm your personal insurance advisor. I can help you find the perfect insurance policy.

To give you the best recommendations, please share:
1. Your age and occupation
2. Family situation (single, married, dependents)
3. Any health conditions
4. Type of insurance you're interested in
5. Your budget range

Feel free to ask any questions about insurance!"""
        }
        st.session_state.chat_messages.append(initial_message)
    
    # Create a container for chat messages with fixed height
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    if metadata.get("search_performed"):
                        st.info(f"Policy Search: {metadata.get('search_reasoning', 'Searched policy database')}")
                    
                    if metadata.get("policies_found"):
                        policies = metadata["policies_found"]
                        with st.expander(f"Found {len(policies)} Relevant Policies", expanded=True):
                            for i, policy in enumerate(policies, 1):
                                format_policy_card(policy, i, show_actions=True, key_prefix="chat")
    
    # Chat input at the bottom
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(user_message)
        
        st.session_state.chat_conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Get AI response
        with st.spinner("Processing..."):
            api_response = send_chat_message(
                prompt, 
                st.session_state.chat_conversation_history[:-1],
                st.session_state.chat_user_profile
            )
            
            agent_response = api_response.get("response", "Sorry, I couldn't process your request.")
            updated_profile = api_response.get("updated_profile", {})
            search_performed = api_response.get("search_performed", False)
            search_reasoning = api_response.get("search_reasoning")
            policies_found = api_response.get("policies_found", [])
            
            # Update session state
            st.session_state.chat_user_profile = {
                k: v for k, v in updated_profile.items() 
                if k != 'profile_summary'
            }
            
            st.session_state.chat_conversation_history.append({
                "role": "assistant",
                "content": agent_response
            })
            
            assistant_message = {
                "role": "assistant",
                "content": agent_response,
                "metadata": {
                    "search_performed": search_performed,
                    "search_reasoning": search_reasoning,
                    "policies_found": policies_found
                }
            }
            st.session_state.chat_messages.append(assistant_message)
        
        # Rerun to display the new messages
        st.rerun()

def recommend_policies_tab():
    """Policy recommendation form tab."""
    st.header("Get Policy Recommendations")
    st.markdown("Fill out this form to get AI-powered policy recommendations")
    
    with st.form("recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=22)
            gender = st.selectbox("Gender", ["male", "female", "other"])
            location = st.text_input("Location", value="Delhi, India")
            income_range = st.selectbox("Income Range", [
                "Below 5 LPA", "5-10 LPA", "10-15 LPA", "15-20 LPA", 
                "20-25 LPA", "25-30 LPA", "Above 30 LPA"
            ])
            family_status = st.selectbox("Family Status", [
                "single", "married", "divorced", "widowed"
            ])
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        
        with col2:
            occupation = st.text_input("Occupation", value="Software Engineer")
            lifestyle = st.selectbox("Lifestyle", ["active", "moderate", "sedentary"])
            risk_tolerance = st.selectbox("Risk Tolerance", ["low", "medium", "high"])
            budget_range = st.text_input("Budget Range", value="10-20K per year")
            
            coverage_preferences = st.multiselect("Coverage Preferences", [
                "health", "life", "motor", "travel", "home", "personal_accident"
            ], default=["health"])
            
            health_conditions = st.multiselect("Health Conditions", [
                "diabetes", "hypertension", "heart_disease", "asthma", "cancer", "none"
            ], default=["none"])
        
        additional_notes = st.text_area("Additional Notes", placeholder="Any specific requirements or preferences...")
        
        col1, col2, col3 = st.columns(3)
        # with col1:
        #     top_k = st.number_input("Policies to Search", min_value=5, max_value=50, value=20)
        # with col2:
        #     top_n = st.number_input("Final Recommendations", min_value=1, max_value=20, value=5)
        top_k=20
        top_n=5
        submitted = st.form_submit_button("Get Recommendations")
    
    # Handle form submission outside the form
    if submitted:
        with st.spinner("Analyzing your profile and finding best policies..."):
            user_profile = {
                "age": age,
                "gender": gender,
                "location": location,
                "income_range": income_range,
                "family_status": family_status,
                "dependents": dependents,
                "health_conditions": health_conditions,
                "occupation": occupation,
                "lifestyle": lifestyle,
                "risk_tolerance": risk_tolerance,
                "budget_range": budget_range,
                "coverage_preferences": coverage_preferences,
                "additional_notes": additional_notes
            }
            
            result = send_policy_recommendation_request(user_profile, top_k, top_n)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                recommendations = result.get("recommendations", [])
                st.session_state.form_recommendations = recommendations
                st.success(f"Found {len(recommendations)} recommendations in {result.get('processing_time_seconds', 0):.1f}s")
                
                st.markdown(f"**Search Query Used:** {result.get('search_query_used', 'N/A')}")
    
    # Display recommendations outside the form
    if st.session_state.form_recommendations:
        st.markdown("### Recommended Policies")
        for i, policy in enumerate(st.session_state.form_recommendations, 1):
            format_policy_card(policy, i, show_actions=True, key_prefix="form")

def policy_chat_tab():
    """Policy chat tab."""
    st.header("Chat with Policy Document")
    
    # Policy URL input with unique key
    pdf_url = st.text_input("Policy Document URL", 
                           value=st.session_state.selected_policy_url, 
                           placeholder="Enter the PDF URL of the policy document",
                           key="policy_chat_url_input")
    
    if not pdf_url:
        st.info("Enter a policy document URL to start chatting")
        return
    
    # Create session ID for this policy
    session_key = f"policy_chat_{abs(hash(pdf_url))}"
    
    if session_key not in st.session_state.policy_chat_sessions:
        st.session_state.policy_chat_sessions[session_key] = {
            "messages": [],
            "session_id": str(uuid.uuid4())[:8],
            "pdf_url": pdf_url
        }
    
    session_data = st.session_state.policy_chat_sessions[session_key]
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in session_data["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # if message["role"] == "assistant" and "document_info" in message:
                #     doc_info = message["document_info"]
                    # with st.expander("Document Info"):
                    #     st.write(f"**Title:** {doc_info['title']}")
                    #     st.write(f"**Chunks:** {doc_info['total_chunks']}")
                    #     st.write(f"**Tokens:** {doc_info['total_tokens']:,}")
    
    # Chat input at the bottom with unique key
    if prompt := st.chat_input("Ask questions about this policy...", key="policy_chat_input"):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        session_data["messages"].append(user_message)
        
        # Get response
        with st.spinner("Analyzing policy document..."):
            response = send_policy_chat_message(pdf_url, prompt, session_data["session_id"])
            
            if "error" in response:
                assistant_message = {
                    "role": "assistant", 
                    "content": f"Error: {response['error']}",
                    "document_info": {}
                }
            else:
                assistant_response = response.get("response", "No response received")
                doc_info = response.get("document_info", {})
                
                # Add to session
                assistant_message = {
                    "role": "assistant", 
                    "content": assistant_response,
                    "document_info": doc_info
                }
            
            session_data["messages"].append(assistant_message)
        
        # Rerun to display the new messages
        st.rerun()

def policy_analysis_tab():
    """Policy analysis tab."""
    st.header("Policy Document Analysis")
    
    # Policy URL input with unique key
    pdf_url = st.text_input("Policy Document URL", 
                           value=st.session_state.selected_policy_url,
                           placeholder="Enter the PDF URL of the policy document",
                           key="policy_analysis_url_input")
    
    if st.button("Analyze Policy", disabled=not pdf_url):
        if pdf_url:
            with st.spinner("Analyzing policy document... This may take a few minutes."):
                result = send_policy_analysis_request(pdf_url)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"Analysis completed in {result.get('processing_time_seconds', 0):.1f}s")
                    
                    # Display document info
                    # col1, col2, col3 = st.columns(3)
                    # with col1:
                    #     st.metric("Document", result.get('document_title', 'Unknown'))
                    # with col2:
                    #     st.metric("Chunks", result.get('total_chunks', 0))
                    # with col3:
                    #     st.metric("Tokens", f"{result.get('total_tokens', 0):,}")
                    
                    # Display analysis
                    st.markdown("### Analysis Results")
                    st.markdown(result.get('analysis', 'No analysis available'))

def main():
    st.title("AI Insurance Agent")
    st.markdown("Your comprehensive insurance advisory platform")
    
    # Check API availability
    if st.session_state.api_available is None:
        with st.spinner("Checking API connection..."):
            st.session_state.api_available = check_api_health()
    
    if not st.session_state.api_available:
        st.error("API Service Unavailable")
        st.markdown("Please ensure the FastAPI server is running on http://localhost:8001")
        if st.button("Retry Connection"):
            st.session_state.api_available = None
            st.rerun()
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Chat Recommendations", "Form Recommendations", "Policy Chat", "Policy Analysis"])
    
    with tab1:
        chat_tab()
    
    with tab2:
        recommend_policies_tab()
    
    with tab3:
        policy_chat_tab()
    
    with tab4:
        policy_analysis_tab()
    
    # Sidebar
    with st.sidebar:
        st.header("Connection Status")
        if st.session_state.api_available:
            st.success("API Connected")
        else:
            st.error("API Disconnected")
        
        if st.button("Refresh API"):
            st.session_state.api_available = None
            st.rerun()
        
        if st.button("Clear All Data"):
            # Clear all session states except api_available
            keys_to_clear = [k for k in st.session_state.keys() if k != 'api_available']
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()