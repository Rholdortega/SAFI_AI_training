"""
Streamlit Frontend for SAFI Chatbot
Clean, fast conversational UI
"""

import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="SAFI Chatbot",
    page_icon="🎍",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Hide all default avatars */
    [data-testid="stChatMessageAvatarContainer"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with large icon and title
st.markdown("""
    <h1 style='font-size: 3.5rem; text-align: center;'>
        <span style='font-size: 5rem;'>🎍</span> SAFI Chatbot
    </h1>
    <p style='text-align: center; color: #8B7355; font-size: 1.2rem;'>Sustainable & Alternative Fibers Initiative</p>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


def send_message(message: str) -> str:
    """Send message to backend API"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "conversation_id": "default", "stream": False},
            timeout=60  # Longer timeout for LLM responses
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Error: No response in data")
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to backend. Ensure FastAPI is running on port 8000."
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The model is taking too long."
    except Exception as e:
        return f"Error: {str(e)}"


# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"✨ {msg['content']}")
    else:
        st.markdown(f"**You:** {msg['content']}")

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**You:** {prompt}")
    
    # Get assistant response
    with st.spinner("Thinking..."):
        response = send_message(prompt)
    st.markdown(f"✨ {response}")
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Sidebar
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Debug section
    st.caption("Debug")
    if st.button("Test Backend"):
        try:
            # Test health
            health = requests.get(f"{API_URL}/health", timeout=5)
            st.write(f"Health: {health.json()}")
            
            # Test chat
            test = requests.post(
                f"{API_URL}/chat",
                json={"message": "Hi", "conversation_id": "test", "stream": False},
                timeout=30
            )
            st.write(f"Chat test: {test.json()}")
        except Exception as e:
            st.error(f"Error: {e}")
