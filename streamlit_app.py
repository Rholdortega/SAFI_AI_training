"""
SAFI Research Intelligence - Full Context Version
Loads all papers directly into Gemini's context window (no RAG)
"""
import streamlit as st
import google.generativeai as genai
from typing import List, Tuple
import os
import pickle
import glob

# Page config
st.set_page_config(
    page_title="SAFI Research Intelligence",
    page_icon="🎍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS with GREEN theme
st.markdown("""
    <style>
    .main {
        background-color: #f0f5f0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #e8f0e8;
    }
    
    .main .block-container {
        max-width: 48rem;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        margin-bottom: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .user-message {
        background-color: #e3f0e3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        margin-right: 20%;
        border: 1px solid #d0e0d0;
    }
    
    .message-label {
        font-size: 0.75rem;
        color: #4a6b4a;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .stChatInputContainer {
        background-color: #f0f5f0;
    }
    
    .stats-box {
        background-color: #f8faf8;
        border-left: 3px solid #4a6b4a;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #5a7a5a;
    }
    </style>
""", unsafe_allow_html=True)

# Get API key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = None

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None


def load_papers_from_embeddings(embeddings_file: str) -> Tuple[str, List[str], int]:
    """
    Reconstruct full paper texts from the embeddings pickle file.
    Returns: (combined_text, paper_names, estimated_tokens)
    """
    if not os.path.exists(embeddings_file):
        return "", [], 0
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data["knowledge_base"]
    metadata = data["chunk_metadata"]
    
    # Group chunks by source file
    papers = {}
    for chunk, meta in zip(chunks, metadata):
        source = meta.get('source', meta.get('file', 'Unknown'))
        if source not in papers:
            papers[source] = []
        papers[source].append(chunk)
    
    # Combine into structured document
    combined_parts = []
    paper_names = list(papers.keys())
    
    for paper_name in paper_names:
        paper_chunks = papers[paper_name]
        paper_text = "\n".join(paper_chunks)
        combined_parts.append(f"""
================================================================================
PAPER: {paper_name}
================================================================================
{paper_text}
""")
    
    combined_text = "\n".join(combined_parts)
    
    # Estimate tokens (~4 chars per token for English)
    estimated_tokens = len(combined_text) // 4
    
    return combined_text, paper_names, estimated_tokens


def generate_response(message: str, full_context: str, chat_history: List[dict] = None) -> str:
    """Generate response using full paper context"""
    if not model:
        return "⚠️ API key not configured."
    
    if not full_context:
        return "⚠️ No papers loaded. Please check the embeddings file."
    
    try:
        # Build conversation history (last 3 exchanges for continuity)
        conversation_context = ""
        if chat_history and len(chat_history) >= 2:
            recent = chat_history[-6:]  # Last 3 Q&A pairs
            conversation_context = "Recent conversation:\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:800] + "..." if len(msg["content"]) > 800 else msg["content"]
                conversation_context += f"{role}: {content}\n"
            conversation_context += "\n---\n\n"
        
        prompt = f"""You are a research assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

You have access to the complete text of all SAFI research papers below. Use this knowledge to answer questions accurately and comprehensively.

=== SAFI RESEARCH PAPERS ===
{full_context}
=== END OF PAPERS ===

{conversation_context}Current question: {message}

Instructions:
- Answer based on the paper content provided above
- Include specific numerical values with units when available (e.g., "576 kg CO₂-eq/ADt")
- Cite the specific paper when referencing findings
- If comparing across papers, clearly indicate which findings come from which source
- For terms like "carbon footprint," "GWP," and "global warming potential," treat them as equivalent metrics
- If the papers don't contain information to answer the question, say so clearly
- Be thorough but concise

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "⚠️ API quota exceeded. The full-context approach uses more tokens per request. Try again later or switch to the RAG version."
        return f"Error: {error_msg}"


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_context" not in st.session_state:
    st.session_state.full_context = ""
if "paper_names" not in st.session_state:
    st.session_state.paper_names = []
if "token_estimate" not in st.session_state:
    st.session_state.token_estimate = 0
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Load papers on startup
if not st.session_state.initialized and model:
    with st.spinner("Loading SAFI research papers..."):
        try:
            embeddings_file = "safi_embeddings.pkl"
            
            full_context, paper_names, token_estimate = load_papers_from_embeddings(embeddings_file)
            
            if full_context:
                st.session_state.full_context = full_context
                st.session_state.paper_names = paper_names
                st.session_state.token_estimate = token_estimate
                st.session_state.initialized = True
            else:
                st.error(f"Could not load papers from: {embeddings_file}")
                
        except Exception as e:
            st.error(f"Error loading papers: {str(e)}")

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### SAFI Research Intelligence")
    st.caption("Full Context Mode")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats
    if st.session_state.initialized:
        st.metric("Research Papers", len(st.session_state.paper_names))
        st.metric("Est. Context Tokens", f"{st.session_state.token_estimate:,}")
        
        # Show paper list
        with st.expander("📄 Loaded Papers"):
            for name in st.session_state.paper_names:
                st.caption(f"• {name}")
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About"):
        st.write("""
        **Full Context Mode** loads all SAFI papers directly into Gemini's context window.
        
        **Advantages:**
        - No retrieval errors
        - Handles cross-paper questions
        - Better for complex queries
        
        **Trade-off:**
        - Higher token usage per query
        """)

# ============ MAIN CONTENT ============

if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="main-header">
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🎍</div>
            <h1 style='font-size: 2.5rem; margin: 0; font-weight: 400;'>
                SAFI Research Intelligence
            </h1>
            <p style='color: #4a6b4a; font-size: 1rem; margin-top: 0.5rem;'>
                Full Context Mode — All papers loaded
            </p>
        </div>
    """, unsafe_allow_html=True)

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-message">
                <div class="message-label">You</div>
                {msg["content"]}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="assistant-message">
                <div class="message-label">SAFI Research Intelligence</div>
                {msg["content"]}
            </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about SAFI research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing papers..."):
        response = generate_response(
            prompt, 
            st.session_state.full_context,
            st.session_state.messages
        )
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

