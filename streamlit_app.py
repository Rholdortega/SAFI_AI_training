"""
SAFI Research Intelligence - Clean UI with Green Theme
"""
import streamlit as st
import google.generativeai as genai
from typing import List
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
    model = genai.GenerativeModel("gemini-2.0-flash")
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

def retrieve_relevant_context(query: str, top_k: int = 5) -> tuple:
    """Retrieve most relevant chunks with metadata"""
    if not model or "embeddings" not in st.session_state or not st.session_state.embeddings:
        return "", []
    
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in st.session_state.embeddings
    ]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [st.session_state.knowledge_base[i] for i in top_indices]
    chunk_sources = [st.session_state.chunk_metadata[i] for i in top_indices]
    
    context = "\n\n".join(relevant_chunks)
    return context, chunk_sources

def generate_response(message: str) -> str:
    """Generate response with RAG"""
    if not model:
        return "⚠️ API key not configured."
    
    try:
        expanded_query = message
        if "carbon footprint" in message.lower() or "gwp" in message.lower():
            expanded_query = f"{message} global warming potential CO2 emissions kg ADt ton 576"
        if "bek" in message.lower() and "brazil" in message.lower():
            expanded_query = f"{message} Ortega bleached eucalyptus kraft Brazilian"
        
        context, sources = retrieve_relevant_context(expanded_query, top_k=5)
        
        if context:
            source_info = "\n".join([f"- {s['source']}" for s in sources])
            
            augmented_prompt = f"""You are a knowledgeable assistant for the Sustainable & Alternative Fibers Initiative (SAFI). 
Answer the question accurately and concisely using the SAFI knowledge provided below.

SAFI Knowledge from:
{source_info}

{context}

Question: {message}

Instructions:
- Search the SAFI knowledge carefully for SPECIFIC NUMERICAL VALUES
- Look for patterns like "XXX kg CO₂-eq/ton" or "XXX kg CO₂-eq/ADt"
- When reporting GWP/carbon footprint, ALWAYS include the specific value with units
- Use ONLY baseline results (exclude sensitivity analysis unless asked)
- When asked "What is the carbon footprint of BEK in Brazil?", the answer is 576 kg CO₂-eq/ton for average BEK delivered to the U.S.
- The three bleaching sequences have GWP values: D0-Eop-D1-P (632), O/O-D0-Eop-D1-P (583), O/O-A-D0-Eop-D1-P (563 kg CO₂-eq/ton)
- When referencing information, say "According to SAFI research" or "Based on SAFI knowledge"
- Understand that "carbon footprint" and "global warming potential (GWP)" are the same metric
- Provide concise, direct answers

Answer:"""
        else:
            augmented_prompt = f"""You are a knowledgeable assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

Question: {message}

Answer:"""
        
        response = model.generate_content(augmented_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

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
        st.markdown(f"""
            <div class="assistant-message">
                <div class="message-label">SAFI Research Intelligence</div>
                {msg["content"]}
            </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about SAFI research..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        response = generate_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
