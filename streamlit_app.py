"""
SAFI Research Intelligence - Gemini 3.0 Optimized
Updated: January 2026
"""
import streamlit as st
import google.generativeai as genai
from typing import List, Tuple, Dict
import numpy as np
import os
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="SAFI Research Intelligence",
    page_icon="🎍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============ CONFIGURATION ============
# PATHS UPDATED: Pointing to your new 'data/' folder
PRELOADED_EXCEL_FILE = "data/FQA_Compilation.xlsx"
PRELOADED_EXCEL_SHEET = "Fiber morphology"

# Embeddings file path updated
EMBEDDINGS_FILE = "data/safi_embeddings.pkl"

# MODELS UPDATED: Gemini 3.0 Family
# Gemini 3 Flash is optimized for speed and high reasoning
MODEL_NAME = "gemini-3-flash-preview" 
# This embedding model requires specific "task_types" (implemented below)
EMBEDDING_MODEL = "models/gemini-embedding-001" 

# Context settings
MAX_PAPER_CHARS = 100000  # Increased limit for Gemini 3's 1M token window
# =======================================

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f5f0; }
    [data-testid="stSidebar"] { background-color: #e8f0e8; }
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
    .sources-box {
        background-color: #f8faf8;
        border-left: 3px solid #4a6b4a;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #5a7a5a;
    }
    .context-indicator {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.3rem 0.6rem;
        font-size: 0.75rem;
        color: #155724;
        display: inline-block;
        margin: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Get API key from Streamlit Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = None

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
else:
    model = None

# --- DATA LOADING FUNCTIONS ---

def load_embeddings_data(embeddings_file: str) -> Tuple[List[str], List[List[float]], List[dict], Dict[str, str]]:
    """Load embeddings and group chunks by paper"""
    if not os.path.exists(embeddings_file):
        return [], [], [], {}
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data["knowledge_base"]
    embeddings = data["embeddings"]
    metadata = data["chunk_metadata"]
    
    # Group chunks by paper for full context
    papers_text = {}
    for chunk, meta in zip(chunks, metadata):
        source = meta.get('source', meta.get('file', 'Unknown'))
        if source not in papers_text:
            papers_text[source] = []
        papers_text[source].append(chunk)
    
    # Combine chunks per paper (full text)
    papers_combined = {
        name: "\n".join(chunks_list) 
        for name, chunks_list in papers_text.items()
    }
    
    return chunks, embeddings, metadata, papers_combined


def load_preloaded_excel(file_path: str, sheet_name: str = None) -> Tuple[pd.DataFrame, str]:
    """Load the pre-configured Excel file"""
    if not os.path.exists(file_path):
        return None, ""
    
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        context = build_excel_context(df, sheet_name, is_preloaded=True)
        return df, context
    except Exception as e:
        st.warning(f"Could not load pre-configured Excel file: {e}")
        return None, ""


def build_excel_context(df: pd.DataFrame, sheet_name: str = None, is_preloaded: bool = False) -> str:
    """Build context string from Excel data"""
    if df is None or df.empty:
        return ""
    
    num_rows, num_cols = df.shape
    columns = df.columns.tolist()
    
    prefix = "PRE-LOADED" if is_preloaded else "UPLOADED"
    
    context_parts = [
        f"=== {prefix} EXCEL DATA {'(' + sheet_name + ')' if sheet_name else ''} ===",
        f"Dataset contains {num_rows} rows and {num_cols} columns.",
        f"Columns: {', '.join(columns)}",
        "",
        "Full data:",
        df.to_string(index=False),
        "",
        f"=== END OF {prefix} EXCEL DATA ==="
    ]
    
    return "\n".join(context_parts)


def build_full_papers_context(papers_combined: Dict[str, str], max_chars_per_paper: int = MAX_PAPER_CHARS) -> str:
    """Build full paper content context"""
    parts = ["=== SAFI RESEARCH PAPERS - FULL CONTENT ===\n"]
    parts.append("You have access to the complete text of the following research papers:\n")
    
    total_chars = 0
    for name, text in papers_combined.items():
        # Truncate very long papers if needed (though Gemini 3 handles large contexts)
        if len(text) > max_chars_per_paper:
            truncated_text = text[:max_chars_per_paper] + f"\n\n[... Paper truncated at {max_chars_per_paper} characters ...]"
        else:
            truncated_text = text
        
        paper_section = f"""
{'='*60}
PAPER: {name}
{'='*60}
{truncated_text}

"""
        parts.append(paper_section)
        total_chars += len(truncated_text)
    
    parts.append("=== END OF RESEARCH PAPERS ===")
    
    approx_tokens = total_chars // 4
    parts.insert(1, f"[Approximate context size: {total_chars:,} characters / ~{approx_tokens:,} tokens]\n")
    
    return "\n".join(parts)


def retrieve_relevant_chunks(
    query: str, 
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[dict],
    top_k: int = 6
) -> Tuple[str, List[str]]:
    """Retrieve relevant chunks using updated Gemini 3 embedding model logic"""
    if not embeddings:
        return "", []
    
    # UPGRADED: Explicitly using the new embedding model with task_type
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query" # Critical for Gemini 3
        )
        query_embedding = result["embedding"]
    except Exception as e:
        return f"Embedding Error: {e}", []
    
    # Calculate Cosine Similarity
    # Note: Ensure your embeddings list is a numpy array for this to work efficiently
    emb_array = np.array(embeddings)
    q_emb_array = np.array(query_embedding)
    
    dot_products = np.dot(emb_array, q_emb_array)
    norms = np.linalg.norm(emb_array, axis=1) * np.linalg.norm(q_emb_array)
    similarities = dot_products / norms
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Filter by similarity threshold
    filtered_indices = [i for i in top_indices if similarities[i] >= 0.35]
    
    if not filtered_indices:
        return "", []
    
    retrieved_chunks = [chunks[i] for i in filtered_indices]
    sources = list(set([metadata[i].get('source', 'Unknown') for i in filtered_indices]))
    
    context = "\n\n---\n\n".join(retrieved_chunks)
    return context, sources


def estimate_tokens(text: str) -> int:
    return len(text) // 4

def generate_response(
    message: str,
    full_papers_context: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[dict],
    preloaded_excel_context: str = "",
    uploaded_excel_context: str = "",
    chat_history: List[dict] = None
) -> Tuple[str, List[str]]:
    """Generate response using full context approach"""
    if not model:
        return "⚠️ API key not configured.", []
    
    try:
        # Retrieve most relevant chunks (RAG)
        detailed_context, sources = retrieve_relevant_chunks(
            message, chunks, embeddings, metadata, top_k=6
        )
        
        # Build conversation context
        conversation = ""
        if chat_history and len(chat_history) >= 2:
            recent = chat_history[-4:]
            conversation = "Recent conversation:\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                conversation += f"{role}: {content}\n"
            conversation += "\n"
        
        highlighted_section = ""
        if detailed_context:
            highlighted_section = f"""
=== MOST RELEVANT SECTIONS (retrieved via semantic search) ===
{detailed_context}
=== END OF HIGHLIGHTED SECTIONS ===
"""
        
        # Combine Excel contexts
        excel_section = ""
        if preloaded_excel_context or uploaded_excel_context:
            excel_parts = []
            if preloaded_excel_context: excel_parts.append(preloaded_excel_context)
            if uploaded_excel_context: excel_parts.append(uploaded_excel_context)
            excel_section = "\n\n".join(excel_parts)
        
        prompt = f"""You are a research assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

You have access to:
1. FULL TEXT of all 17 SAFI research papers
2. HIGHLIGHTED SECTIONS specific to this question
3. EXCEL DATA with fiber morphology measurements

{full_papers_context[:500000]} # Safe slice for large context

{highlighted_section}

{excel_section}

{conversation}Current question: {message}

Instructions:
- Use the FULL TEXT papers for comprehensive answers.
- Use Excel data for specific fiber properties (length, width, kappa, etc.).
- Cite the paper name when referencing findings.
- Answer as a domain expert in Forest Biomaterials.

Answer:"""
        
        response = model.generate_content(prompt)
        
        # Determine sources used (simple keyword check)
        excel_keywords = ['fiber', 'length', 'width', 'kappa', 'coarseness', 'morphology', 'pulp']
        if preloaded_excel_context and any(kw in message.lower() for kw in excel_keywords):
            if "Pre-loaded Excel Data" not in sources:
                sources.append("Pre-loaded Excel Data")
        
        return response.text, sources
    
    except Exception as e:
        return f"Error: {str(e)}", []


# ============ MAIN APP LOGIC ============

# Initialize session state
if "messages" not in st.session_state: st.session_state.messages = []
if "initialized" not in st.session_state: st.session_state.initialized = False
if "context_stats" not in st.session_state: st.session_state.context_stats = {}

# Load data on startup
if not st.session_state.initialized and model:
    with st.spinner("Loading SAFI knowledge base (Gemini 3.0 optimized)..."):
        try:
            # Load embeddings
            chunks, embeddings, metadata, papers_combined = load_embeddings_data(EMBEDDINGS_FILE)
            
            if chunks:
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.metadata = metadata
                st.session_state.paper_names = list(papers_combined.keys())
                st.session_state.full_papers_context = build_full_papers_context(papers_combined)
                
                st.session_state.context_stats["papers_tokens"] = estimate_tokens(st.session_state.full_papers_context)
                st.session_state.initialized = True
            else:
                st.error(f"Could not load embeddings from {EMBEDDINGS_FILE}")
                
        except Exception as e:
            st.error(f"Error loading papers: {str(e)}")

    # Load pre-configured Excel file
    try:
        preloaded_df, preloaded_context = load_preloaded_excel(PRELOADED_EXCEL_FILE, PRELOADED_EXCEL_SHEET)
        if preloaded_df is not None:
            st.session_state.preloaded_excel_df = preloaded_df
            st.session_state.preloaded_excel_context = preloaded_context
            st.session_state.context_stats["excel_tokens"] = estimate_tokens(preloaded_context)
    except Exception as e:
        st.warning(f"Could not load pre-configured Excel: {str(e)}")


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### SAFI Research Intelligence")
    st.caption(f"Model: {MODEL_NAME}")
    
    st.divider()
    
    if st.session_state.initialized:
        total = st.session_state.context_stats.get("papers_tokens", 0) + \
                st.session_state.context_stats.get("excel_tokens", 0)
        
        st.markdown(f"""
        <div class="context-indicator">📄 {len(st.session_state.paper_names)} papers</div>
        <div class="context-indicator">~{total:,} tokens</div>
        """, unsafe_allow_html=True)
        st.progress(min(total / 1000000, 1.0), text="1M Token Context Usage")

    st.divider()
    
    # Upload Additional Data
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
    if uploaded_file:
        xl = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xl, sheet_name=0)
        st.session_state.uploaded_excel_context = build_excel_context(df, "Uploaded", is_preloaded=False)
        st.success(f"Loaded {len(df)} rows")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============ MAIN CHAT ============
st.markdown("""
    <div class="main-header">
        <h1 style='font-size: 2.5rem;'>🎍 SAFI Research Intelligence</h1>
    </div>
""", unsafe_allow_html=True)

# Display Chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message"><div class="message-label">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            s_list = "<br>".join([f"• {s}" for s in msg["sources"]])
            sources_html = f'<div class="sources-box"><strong>Sources:</strong><br>{s_list}</div>'
        st.markdown(f'<div class="assistant-message"><div class="message-label">SAFI AI</div>{msg["content"]}{sources_html}</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask about fiber morphology, kappa numbers, or specific papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Generate Answer (after rerun to show user message immediately)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Analyzing with Gemini 3..."):
        response, sources = generate_response(
            st.session_state.messages[-1]["content"],
            st.session_state.get("full_papers_context", ""),
            st.session_state.get("chunks", []),
            st.session_state.get("embeddings", []),
            st.session_state.get("metadata", []),
            st.session_state.get("preloaded_excel_context", ""),
            st.session_state.get("uploaded_excel_context", ""),
            st.session_state.messages
        )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        st.rerun()
