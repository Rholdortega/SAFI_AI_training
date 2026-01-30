"""
SAFI Research Intelligence - Gemini 3.0 (No Icons - Custom HTML Fix)
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

# ============ STYLING (CUSTOM BUBBLES) ============
st.markdown("""
    <style>
    .main { background-color: #f0f5f0; }
    [data-testid="stSidebar"] { background-color: #e8f0e8; }
    .main-header { text-align: center; padding: 2rem 0; margin-bottom: 1rem; }
    
    /* User Message Bubble */
    .user-message {
        background-color: #e3f0e3;
        padding: 1rem;
        border-radius: 15px 15px 0px 15px; /* Rounded corners */
        margin: 1rem 0;
        margin-left: 20%; /* Push to right */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1a1a1a;
    }
    
    /* SAFI AI Message Bubble */
    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 15px 15px 15px 0px; /* Rounded corners */
        margin: 1rem 0;
        margin-right: 20%; /* Push to left */
        border: 1px solid #d0e0d0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1a1a1a;
    }
    
    /* Source Box Styling */
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
    if not os.path.exists(file_path):
        return [], [], [], {}
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    chunks = data["knowledge_base"]
    embeddings = data["embeddings"]
    metadata = data["chunk_metadata"]
    
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
    
    st.markdown("### ⚙️ Response Mode")
    mode = st.radio(
        "Choose Engine:",
        ["🚀 Fast Mode", "🧠 Thinking Mode"],
        captions=["Instant answers (Flash)", "Deep reasoning (Pro)"]
    )
    
    if mode == "🚀 Fast Mode":
        current_model_name = "gemini-3-flash-preview"
        current_config = {
            "temperature": 0.1,
            "max_output_tokens": 2000
        }
    else:
        current_model_name = "gemini-3-pro-preview"
        current_config = {
            "temperature": 0.4,
            "max_output_tokens": 4000
        }
        
    st.caption(f"Active Model: {current_model_name}")
    st.divider()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize Model
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

# 1. DISPLAY HISTORY (Using Custom HTML Divs - NO ICONS GUARANTEED)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <b>You:</b><br>{msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build source HTML if sources exist
        source_html = ""
        if "sources" in msg and msg["sources"]:
            s_list = " • ".join(msg["sources"])
            source_html = f"<div class='sources-box'><strong>Sources:</strong><br>{s_list}</div>"
            
        st.markdown(f"""
        <div class="assistant-message">
            <b>SAFI AI:</b><br>{msg['content']}
            {source_html}
        </div>
        """, unsafe_allow_html=True)

# 2. CHAT INPUT
if prompt := st.chat_input("Ask about fiber morphology, kappa numbers, or specific papers..."):
    # Save & Show User Message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
    <div class="user-message">
        <b>You:</b><br>{prompt}
    </div>
    """, unsafe_allow_html=True)

    # 3. GENERATE ANSWER
    # We use an empty placeholder to stream content into our custom HTML div
    response_placeholder = st.empty()
    full_response = ""
    sources = []

    # A. Retrieval Logic
    retrieved_text = ""
    if st.session_state.embeddings:
        try:
            res = genai.embed_content(model=EMBEDDING_MODEL, content=prompt, task_type="retrieval_query")
            query_embedding = np.array(res["embedding"])
            embeddings_array = np.array(st.session_state.embeddings)
            
            dot_products = np.dot(embeddings_array, query_embedding)
            norms = np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
            similarities = dot_products / norms
            
            top_indices = np.argsort(similarities)[-6:][::-1]
            relevant_indices = [i for i in top_indices if similarities[i] >= 0.35]
            
            retrieved_text = "\n---\n".join([st.session_state.chunks[i] for i in relevant_indices])
            sources = list(set([st.session_state.metadata[i].get('source', 'Unknown') for i in relevant_indices]))
        except:
            pass

    # B. Excel Logic
    excel_data = ""
    excel_keywords = ['fiber', 'length', 'width', 'kappa', 'coarseness', 'morphology', 'pulp']
    if any(kw in prompt.lower() for kw in excel_keywords):
        excel_data = st.session_state.excel_context
        if "Pre-loaded Excel Data" not in sources:
            sources.append("Pre-loaded Excel Data")

    # Smart Table Instruction
    force_table = ""
    if any(w in prompt.lower() for w in ["table", "compare", "vs", "list"]):
        force_table = "\nIMPORTANT: The user wants a comparison. FORMAT AS A MARKDOWN TABLE."

    # C. Build Prompt
    final_prompt = f"""You are the SAFI Research Assistant.
    CONTEXT: {st.session_state.full_papers_context[:300000]}
    HIGHLIGHTS: {retrieved_text}
    EXCEL: {excel_data}
    QUESTION: {prompt}
    {force_table}
    Answer based on context. Cite papers."""

    # D. Stream Response into Custom Div
    try:
        if model:
            stream = model.generate_content(final_prompt, stream=True)
            
            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    # Update placeholder with new HTML frame every time text arrives
                    response_placeholder.markdown(f"""
                    <div class="assistant-message">
                        <b>SAFI AI:</b><br>{full_response}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Final Update with Sources attached
            source_html = ""
            if sources:
                s_text = " • ".join(sources)
                source_html = f"<div class='sources-box'><strong>Sources:</strong><br>{s_text}</div>"
            
            response_placeholder.markdown(f"""
            <div class="assistant-message">
                <b>SAFI AI:</b><br>{full_response}
                {source_html}
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "sources": sources
            })
        else:
            st.error("Model Error")
            
    except Exception as e:
        st.error(f"Error: {e}")
