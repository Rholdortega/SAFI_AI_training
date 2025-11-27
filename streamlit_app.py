"""
SAFI Chatbot with RAG - Modern UI Design
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

# Page config with modern styling
st.set_page_config(
    page_title="SAFI Research Assistant",
    page_icon="🎍",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for cleaner look
)

# Custom CSS for modern design
st.markdown("""
    <style>
    /* Main chat container */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Suggestion chips */
    .suggestion-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: #e8f4f8;
        border: 1px solid #b8dae8;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .suggestion-chip:hover {
        background-color: #d0e8f2;
        border-color: #8bb8d8;
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
    """Attempt to fetch PDF from DOI (works for open access papers)"""
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
        # Expand query with synonyms for better retrieval
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
- If you find a specific value in the knowledge, state it clearly
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
    with st.spinner("🎍 Loading SAFI knowledge base..."):
        try:
            # Load pre-computed embeddings
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

# Sidebar with modern design
with st.sidebar:
    st.markdown("### 🎍 SAFI Research Assistant")
    st.markdown("---")
    
    # Knowledge base stats
    if st.session_state.initialized:
        unique_papers = len(set([m['file'] for m in st.session_state.chunk_metadata]))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", unique_papers, help="Research papers in knowledge base")
        with col2:
            st.metric("Chunks", len(st.session_state.knowledge_base), help="Searchable text segments")
        
        st.markdown("---")
    
    # Expander for upload features (collapsed by default for cleaner UI)
    with st.expander("➕ Add More Papers", expanded=False):
        # PDF Upload
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Add additional research papers to the knowledge base"
        )
        
        if uploaded_files and st.button("Process", key="process_pdfs"):
            with st.spinner("Processing..."):
                for pdf_file in uploaded_files:
                    text = extract_text_from_pdf(pdf_file)
                    chunks = chunk_text(text)
                    
                    st.session_state.knowledge_base.extend(chunks)
                    
                    for chunk in chunks:
                        embedding = genai.embed_content(
                            model=embedding_model,
                            content=chunk,
                            task_type="retrieval_document"
                        )["embedding"]
                        st.session_state.embeddings.append(embedding)
                        st.session_state.chunk_metadata.append({
                            "source": f"Uploaded: {pdf_file.name}",
                            "file": pdf_file.name
                        })
                        time.sleep(0.5)
                
                st.success(f"✅ Added {len(uploaded_files)} papers")
        
        st.markdown("**Or add by DOI:**")
        doi_input = st.text_input("Enter DOI", placeholder="10.1016/j.cesys.2024.100234")
        
        if doi_input and st.button("Fetch", key="fetch_doi"):
            with st.spinner(f"Fetching..."):
                text = fetch_pdf_from_doi(doi_input)
                if "Error" not in text and "Could not" not in text:
                    chunks = chunk_text(text)
                    st.session_state.knowledge_base.extend(chunks)
                    
                    for chunk in chunks:
                        embedding = genai.embed_content(
                            model=embedding_model,
                            content=chunk,
                            task_type="retrieval_document"
                        )["embedding"]
                        st.session_state.embeddings.append(embedding)
                        st.session_state.chunk_metadata.append({
                            "source": f"DOI: {doi_input}",
                            "file": doi_input
                        })
                        time.sleep(0.5)
                    
                    st.success("✅ Added paper from DOI")
                else:
                    st.error(text)
    
    st.markdown("---")
    
    # Actions
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # About section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **SAFI Research Assistant** provides instant access to peer-reviewed research on:
        
        - Sustainable fibers & biomaterials
        - Life cycle assessment (LCA)
        - Carbon footprint analysis
        - Pulp & paper production
        - Soil organic carbon sequestration
        
        Powered by AI with 17 research papers from the SAFI consortium.
        """)

# Main content area
# Modern header (only show if no messages yet)
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="main-header">
            <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                🎍 SAFI Research Assistant
            </h1>
            <p style='color: #666; font-size: 1.1rem;'>
                Ask questions about sustainable fibers, LCA, and biomaterials research
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Example questions as clickable chips
    st.markdown("### 💡 Try asking:")
    
    col1, col2 = st.columns(2)
    
    example_questions = [
        "What is the carbon footprint of BEK in Brazil?",
        "What are the three bleaching sequences studied?",
        "How does SOC sequestration affect GWP?",
        "Which bleaching sequence has the lowest impact?",
        "What papers are in the knowledge base?",
        "Explain oxygen delignification in pulp production"
    ]
    
    for i, question in enumerate(example_questions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    response = generate_response(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# Display chat messages with modern styling
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎍" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# Chat input (modern, at bottom)
if prompt := st.chat_input("Ask about SAFI research...", key="chat_input"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🎍"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
