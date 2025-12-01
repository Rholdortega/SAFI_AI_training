"""
SAFI Research Intelligence - Clean UI with Green Theme
"""
import streamlit as st
import google.generativeai as genai
from typing import List, Tuple
import numpy as np
import PyPDF2
import requests
from io import BytesIO
import time
import os
import pickle

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
    /* Main background - soft green like Claude's beige */
    .main {
        background-color: #f0f5f0;
    }
    
    /* Sidebar background - slightly darker green */
    [data-testid="stSidebar"] {
        background-color: #e8f0e8;
    }
    
    /* Limit chat width */
    .main .block-container {
        max-width: 48rem;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Clean header */
    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        margin-bottom: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* User message styling - light green */
    .user-message {
        background-color: #e3f0e3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        margin-left: 20%;
    }
    
    /* Assistant message styling - white with subtle green border */
    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        margin-right: 20%;
        border: 1px solid #d0e0d0;
    }
    
    /* Message labels */
    .message-label {
        font-size: 0.75rem;
        color: #4a6b4a;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Chat input background */
    .stChatInputContainer {
        background-color: #f0f5f0;
    }
    
    /* Source citation styling */
    .sources-box {
        background-color: #f8faf8;
        border-left: 3px solid #4a6b4a;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
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
    embedding_model = "models/text-embedding-004"
else:
    model = None
    embedding_model = None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def fetch_pdf_from_doi(doi: str) -> str:
    """Attempt to fetch PDF from DOI"""
    try:
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=research@safi.edu"
        response = requests.get(unpaywall_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('is_oa') and data.get('best_oa_location'):
                pdf_url = data['best_oa_location'].get('url_for_pdf')
                if pdf_url:
                    pdf_response = requests.get(pdf_url, timeout=30)
                    if pdf_response.status_code == 200:
                        pdf_file = BytesIO(pdf_response.content)
                        return extract_text_from_pdf(pdf_file)
        
        return f"Could not access open PDF for DOI: {doi}"
    except Exception as e:
        return f"Error fetching DOI: {str(e)}"

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_context(query: str, top_k: int = 8, min_similarity: float = 0.3) -> Tuple[str, List[dict], List[float]]:
    """Retrieve most relevant chunks with metadata and similarity scores"""
    if not model or "embeddings" not in st.session_state or not st.session_state.embeddings:
        return "", [], []
    
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in st.session_state.embeddings
    ]
    
    # Get top indices sorted by similarity
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Filter by minimum similarity threshold
    filtered_indices = [i for i in top_indices if similarities[i] >= min_similarity]
    
    if not filtered_indices:
        return "", [], []
    
    relevant_chunks = [st.session_state.knowledge_base[i] for i in filtered_indices]
    chunk_sources = [st.session_state.chunk_metadata[i] for i in filtered_indices]
    chunk_scores = [similarities[i] for i in filtered_indices]
    
    context = "\n\n".join(relevant_chunks)
    return context, chunk_sources, chunk_scores


def generate_response(message: str, chat_history: List[dict] = None) -> Tuple[str, List[str]]:
    """Generate response with RAG and return sources"""
    if not model:
        return "⚠️ API key not configured.", []
    
    try:
        # Retrieve relevant context
        context, sources, scores = retrieve_relevant_context(message, top_k=8, min_similarity=0.3)
        
        # Build conversation context from recent history (last 2 exchanges)
        conversation_context = ""
        if chat_history and len(chat_history) >= 2:
            recent = chat_history[-4:]  # Last 2 Q&A pairs
            conversation_context = "Recent conversation:\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                # Truncate long messages
                content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                conversation_context += f"{role}: {content}\n"
            conversation_context += "\n"
        
        if context:
            # Deduplicate sources while preserving order
            seen = set()
            unique_sources = []
            for s in sources:
                if s['source'] not in seen:
                    seen.add(s['source'])
                    unique_sources.append(s['source'])
            
            source_info = "\n".join([f"- {s}" for s in unique_sources])
            
            augmented_prompt = f"""You are a research assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

Answer the question using the retrieved SAFI research content below. Be accurate and base your response on the evidence provided.

{conversation_context}Retrieved from:
{source_info}

---
{context}
---

Current question: {message}

Instructions:
- Answer based on the retrieved content above
- Include specific numerical values with units when available
- If the retrieved content doesn't fully address the question, acknowledge this
- Cite the source when referencing specific findings (e.g., "According to [Author et al.]...")
- For terms like "carbon footprint," "GWP," and "global warming potential," treat them as equivalent metrics
- If this is a follow-up question, use the conversation context appropriately

Answer:"""
            
            response = model.generate_content(augmented_prompt)
            return response.text, unique_sources
        else:
            augmented_prompt = f"""You are a research assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

{conversation_context}No relevant documents were retrieved for this question. Please let the user know and offer to help if they can rephrase or clarify.

Question: {message}

Answer:"""
            
            response = model.generate_content(augmented_prompt)
            return response.text, []
    
    except Exception as e:
        return f"Error: {str(e)}", []

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "chunk_metadata" not in st.session_state:
    st.session_state.chunk_metadata = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Load pre-computed embeddings
if not st.session_state.initialized and model:
    with st.spinner("Loading SAFI knowledge base..."):
        try:
            embeddings_file = "safi_embeddings.pkl"
            
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                
                st.session_state.knowledge_base = data["knowledge_base"]
                st.session_state.embeddings = data["embeddings"]
                st.session_state.chunk_metadata = data["chunk_metadata"]
                
                st.session_state.initialized = True
            else:
                st.error(f"Embeddings file not found: {embeddings_file}")
                
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### SAFI Research Intelligence")
    
    st.divider()
    
    # CLEAR CHAT BUTTON
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats
    if st.session_state.initialized:
        unique_papers = len(set([m['file'] for m in st.session_state.chunk_metadata]))
        st.metric("Research Papers", unique_papers)
        st.metric("Total Chunks", len(st.session_state.knowledge_base))
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        show_sources = st.checkbox("Show sources in responses", value=True)
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About"):
        st.write("Ask questions about SAFI research on sustainable fibers, biomaterials, and life cycle assessment.")

# ============ MAIN CONTENT ============

# Header with centered icon (only when empty)
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="main-header">
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🎍</div>
            <h1 style='font-size: 2.5rem; margin: 0; font-weight: 400;'>
                SAFI Research Intelligence
            </h1>
            <p style='color: #4a6b4a; font-size: 1rem; margin-top: 0.5rem;'>
                Ask questions about SAFI research
            </p>
        </div>
    """, unsafe_allow_html=True)

# Display messages WITHOUT AVATARS
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-message">
                <div class="message-label">You</div>
                {msg["content"]}
            </div>
        """, unsafe_allow_html=True)
    else:
        content = msg["content"]
        sources = msg.get("sources", [])
        
        # Build sources display if enabled and sources exist
        sources_html = ""
        if st.session_state.get("show_sources", True) and sources:
            sources_list = "<br>".join([f"• {s}" for s in sources])
            sources_html = f'<div class="sources-box"><strong>Sources:</strong><br>{sources_list}</div>'
        
        st.markdown(f"""
            <div class="assistant-message">
                <div class="message-label">SAFI Research Intelligence</div>
                {content}
                {sources_html}
            </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about SAFI research..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response with chat history for context
    with st.spinner("Thinking..."):
        response, sources = generate_response(prompt, st.session_state.messages)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })
    st.rerun()
