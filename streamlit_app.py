"""
SAFI Research Intelligence - Gemini 3.0 Final Stable
Updated: January 2026
"""
import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import pickle
import pandas as pd

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="SAFI Research Intelligence",
    page_icon="🎍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============ FILE PATHS ============
PRELOADED_EXCEL_FILE = "data/FQA_Compilation.xlsx"
PRELOADED_EXCEL_SHEET = "Fiber morphology"
EMBEDDINGS_FILE = "data/safi_embeddings.pkl"
EMBEDDING_MODEL = "models/gemini-embedding-001" 

# ============ STYLING ============
st.markdown("""
    <style>
    .main { background-color: #f0f5f0; }
    [data-testid="stSidebar"] { background-color: #e8f0e8; }
    .main-header { text-align: center; padding: 2rem 0; margin-bottom: 1rem; }
    .sources-box {
        background-color: #f8faf8;
        border-left: 3px solid #4a6b4a;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #5a7a5a;
    }
    /* Hide default Streamlit avatars if desired for a cleaner look */
    .stChatMessageAvatarCustom { display: none; }
    </style>
""", unsafe_allow_html=True)

# ============ API SETUP ============
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("⚠️ API Key missing. Please check Streamlit Secrets.")

# ============ DATA LOADING (Cached) ============
@st.cache_data
def load_data(file_path):
    """Loads the PKL file containing the 'Brain' of the chatbot."""
    if not os.path.exists(file_path):
        return [], [], [], {}
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    chunks = data["knowledge_base"]
    embeddings = data["embeddings"]
    metadata = data["chunk_metadata"]
    
    # Reconstruct full papers from chunks
    papers = {}
    for chunk, meta in zip(chunks, metadata):
        src = meta.get('source', 'Unknown')
        if src not in papers:
            papers[src] = []
        papers[src].append(chunk)
        
    papers_combined = {name: "\n".join(chunks) for name, chunks in papers.items()}
    return chunks, embeddings, metadata, papers_combined

@st.cache_data
def load_excel(file_path, sheet):
    """Loads the Excel data for specific lookups."""
    if not os.path.exists(file_path):
        return ""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
        return f"=== EXCEL DATA ({sheet}) ===\n{df.to_string(index=False)}\n"
    except:
        return ""

# ============ SIDEBAR SETTINGS ============
with st.sidebar:
    st.title("🎍 SAFI AI")
    st.markdown("---")
    
    # 1. THE MODE TOGGLE
    st.markdown("### ⚙️ Response Mode")
    mode = st.radio(
        "Choose Engine:",
        ["🚀 Fast Mode", "🧠 Thinking Mode"],
        captions=["Instant answers (Flash)", "Deep reasoning (Pro)"]
    )
    
    # 2. Dynamic Configuration
    if mode == "🚀 Fast Mode":
        current_model_name = "gemini-3-flash-preview"
        current_config = {
            "temperature": 0.1,         # Low temp = fast & precise
            "max_output_tokens": 2000   # Increased to allow for tables
        }
    else:
        current_model_name = "gemini-3-pro-preview"
        current_config = {
            "temperature": 0.4,         # Higher temp = creative/reasoning
            "max_output_tokens": 4000   # Allows detailed explanations
        }
        
    st.caption(f"Active Model: {current_model_name}")
    st.divider()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize the Model
if GEMINI_API_KEY:
    try:
        model = genai.GenerativeModel(
            model_name=current_model_name,
            generation_config=current_config
        )
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        model = None

# ============ APP INITIALIZATION ============
if "initialized" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        chunks, embs, meta, papers = load_data(EMBEDDINGS_FILE)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embs
        st.session_state.metadata = meta
        st.session_state.full_papers_context = "=== PAPERS ===\n" + "\n".join(papers.values())
        st.session_state.excel_context = load_excel(PRELOADED_EXCEL_FILE, PRELOADED_EXCEL_SHEET)
        st.session_state.initialized = True

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============ MAIN CHAT INTERFACE ============
st.markdown("<div class='main-header'><h1>🎍 SAFI Research Intelligence</h1></div>", unsafe_allow_html=True)

# 1. Display History (No Icons)
for msg in st.session_state.messages:
    # avatar=None removes the default icons
    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            source_text = " • ".join(msg["sources"])
            st.markdown(f"<div style='font-size:0.8em; color:#666;'>📚 Sources: {source_text}</div>", unsafe_allow_html=True)

# 2. Chat Input
if prompt := st.chat_input("Ask about fiber morphology, kappa numbers, or specific papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display User Message (No Icon)
    with st.chat_message("user", avatar=None):
        st.markdown(prompt)

    # 3. Generate Answer
    with st.chat_message("assistant", avatar=None):
        # A. Retrieval
        if st.session_state.embeddings:
            try:
                res = genai.embed_content(model=EMBEDDING_MODEL, content=prompt, task_type="retrieval_query")
                query_embedding = np.array(res["embedding"])
                embeddings_array = np.array(st.session_state.embeddings)
                
                # Cosine Similarity
                dot_products = np.dot(embeddings_array, query_embedding)
                norms = np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
                similarities = dot_products / norms
                
                # Top Matches
                top_indices = np.argsort(similarities)[-6:][::-1]
                relevant_indices = [i for i in top_indices if similarities[i] >= 0.35]
                
                retrieved_text = "\n---\n".join([st.session_state.chunks[i] for i in relevant_indices])
                sources = list(set([st.session_state.metadata[i].get('source', 'Unknown') for i in relevant_indices]))
            except:
                retrieved_text = ""
                sources = []
        else:
            retrieved_text = ""
            sources = []

        # B. Check Excel
        excel_keywords = ['fiber', 'length', 'width', 'kappa', 'coarseness', 'morphology', 'pulp']
        if any(kw in prompt.lower() for kw in excel_keywords):
            excel_data = st.session_state.excel_context
            if "Pre-loaded Excel Data" not in sources:
                sources.append("Pre-loaded Excel Data")
        else:
            excel_data = ""

        # --- SMART TABLE LOGIC ---
        table_keywords = ["table", "compare", "comparison", "list", "vs", "versus"]
        force_table_instruction = ""
        if any(kw in prompt.lower() for kw in table_keywords):
             force_table_instruction = "\nIMPORTANT: The user has requested a comparison. YOU MUST FORMAT THE OUTPUT AS A MARKDOWN TABLE."

        # C. Build Prompt
        final_prompt = f"""You are the SAFI Research Assistant.
        
        CONTEXT FROM PAPERS:
        {st.session_state.full_papers_context[:300000]} 
        
        SPECIFIC HIGHLIGHTS:
        {retrieved_text}
        
        EXCEL DATA:
        {excel_data}
        
        QUESTION: {prompt}
        {force_table_instruction}
        
        Please answer based on the context above. Cite the paper names when possible."""

        # D. Stream Response
        try:
            if model:
                stream = model.generate_content(final_prompt, stream=True)
                
                # Clean text extractor
                def stream_parser(stream):
                    for chunk in stream:
                        try:
                            if chunk.text:
                                yield chunk.text
                        except:
                            pass
                
                full_response = st.write_stream(stream_parser(stream))
                
                if sources:
                    st.markdown(f"<div class='sources-box'><strong>Sources used:</strong><br>{' • '.join(sources)}</div>", unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })
            else:
                st.error("Model not initialized. Check API Key.")
            
        except Exception as e:
            st.error(f"Generation Error: {str(e)}")
