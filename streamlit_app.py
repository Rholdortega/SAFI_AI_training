"""
SAFI Chatbot - Single file version for Streamlit Cloud
"""

import streamlit as st
import google.generativeai as genai

# Page config
st.set_page_config(
    page_title="SAFI Chatbot",
    page_icon="🎍",
    layout="centered"
)

# Get API key from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = None

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None

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


def generate_response(message: str) -> str:
    """Generate response using Gemini API"""
    if not model:
        return "⚠️ API key not configured. Add GEMINI_API_KEY to Streamlit secrets."
    try:
        response = model.generate_content(message)
        return response.text
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
        response = generate_response(prompt)
    st.markdown(f"✨ {response}")
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Sidebar
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
