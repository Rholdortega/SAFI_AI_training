"""
SAFI Chatbot with RAG - PDF and DOI support
"""
import streamlit as st
import google.generativeai as genai
from typing import List
import numpy as np
import PyPDF2
import requests
import time
from io import BytesIO

# Page config
st.set_page_config(
    page_title="SAFI Chatbot",
    page_icon="🎍",
    layout="centered"
)

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

# Header
st.markdown("""
    <h1 style='font-size: 3.5rem; text-align: center;'>
        <span style='font-size: 5rem;'>🎍</span> SAFI Chatbot
    </h1>
    <p style='text-align: center; color: #8B7355; font-size: 1.2rem;'>Sustainable & Alternative Fibers Initiative</p>
""", unsafe_allow_html=True)

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
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_context(query: str, top_k: int = 5) -> str:
    """Retrieve most relevant chunks"""
    if not model or "embeddings" not in st.session_state or not st.session_state.embeddings:
        return ""
    
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
    
    return "\n\n".join(relevant_chunks)

def generate_response(message: str) -> str:
    """Generate response with RAG"""
    if not model:
        return "⚠️ API key not configured."
    
    try:
        # Expand query with synonyms for better retrieval
        expanded_query = message
        if "carbon footprint" in message.lower() or "gwp" in message.lower():
            expanded_query = f"{message} global warming potential CO2 emissions kg ADt ton"
        
        context = retrieve_relevant_context(expanded_query, top_k=5)
        
        if context:
            augmented_prompt = f"""You are a knowledgeable assistant for the Sustainable & Alternative Fibers Initiative (SAFI). 
Answer the question accurately and concisely using the SAFI knowledge provided below.

SAFI Knowledge:
{context}

Question: {message}

Instructions:
- Search the SAFI knowledge carefully for SPECIFIC NUMERICAL VALUES
- Look for patterns like "XXX kg CO₂-eq/ton" or "XXX kg CO₂-eq/ADt"
- Look for phrases like "average GWP", "average BEK model", "576", "632", "583", "563"
- The answer should include the actual number from the research
- When reporting GWP/carbon footprint, ALWAYS include the specific value with units
- Use ONLY baseline results (exclude sensitivity analysis unless asked)
- If you find a specific GWP value in the knowledge above, state it clearly
- When asked "What is the carbon footprint of BEK in Brazil?", look specifically for the average BEK value of 576 kg CO₂-eq/ton delivered to the U.S.
- If no specific value is found, say so explicitly

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
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Load pre-existing paper on first run
if not st.session_state.initialized and model:
    with st.spinner("Loading SAFI knowledge base..."):
        try:
            # Path to your uploaded PDF
            pdf_path = "data/Rhonald-1.pdf"
            with open(pdf_path, 'rb') as f:
                text = extract_text_from_pdf(f)
                chunks = chunk_text(text)
                
                st.session_state.knowledge_base.extend(chunks)
                
                for chunk in chunks:
                    embedding = genai.embed_content(
                        model=embedding_model,
                        content=chunk,
                        task_type="retrieval_document"
                    )["embedding"]
                    st.session_state.embeddings.append(embedding)
                
                st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error loading initial paper: {str(e)}")

# Sidebar for document management
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    
    # PDF Upload
    uploaded_files = st.file_uploader(
        "Upload additional PDF papers",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
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
            
            st.success(f"✅ Processed {len(uploaded_files)} PDFs")
    
    # DOI Input
    st.markdown("### 🔗 Add by DOI")
    doi_input = st.text_input("Enter DOI (e.g., 10.1016/j.cesys.2024.100234)")
    
    if doi_input and st.button("Fetch from DOI"):
        with st.spinner(f"Fetching {doi_input}..."):
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
                
                st.success("✅ DOI paper added")
            else:
                st.error(text)
    
    # Stats
    st.info(f"📄 Chunks in knowledge base: {len(st.session_state.knowledge_base)}")
    
    if st.button("Clear Knowledge Base"):
        st.session_state.knowledge_base = []
        st.session_state.embeddings = []
        st.session_state.initialized = False
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"✨ {msg['content']}")
    else:
        st.markdown(f"**You:** {msg['content']}")

# Chat input
if prompt := st.chat_input("Ask about SAFI research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**You:** {prompt}")
    
    with st.spinner("Thinking..."):
        response = generate_response(prompt)
    st.markdown(f"✨ {response}")
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
