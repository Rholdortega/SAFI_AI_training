"""
SAFI Research Intelligence - Full Context Version
Full paper content (1M token context) + RAG retrieval + Pre-loaded Excel data
"""
import streamlit as st
import google.generativeai as genai
from typing import List, Tuple, Dict
import numpy as np
import os
import pickle
import json
import time
import pandas as pd

# Page config
st.set_page_config(
    page_title="SAFI Research Intelligence",
    page_icon="🎍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============ CONFIGURATION ============
# Pre-loaded Excel file path (change this to your file location)
PRELOADED_EXCEL_FILE = "FQA_Compilation.xlsx"  # Place this file in same directory as app
PRELOADED_EXCEL_SHEET = "Fiber morphology"      # Sheet name to load

# Embeddings file
EMBEDDINGS_FILE = "safi_embeddings.pkl"

# Model settings - using Gemini 2.5 Flash with 1M context
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# Context settings
MAX_PAPER_CHARS = 50000  # Max characters per paper (adjust based on your needs)
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
    .stChatInputContainer { background-color: #f0f5f0; }
    .sources-box {
        background-color: #f8faf8;
        border-left: 3px solid #4a6b4a;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #5a7a5a;
    }
    .data-info {
        background-color: #e8f4e8;
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
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

# Get API key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = None

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    embedding_model = EMBEDDING_MODEL
else:
    model = None
    embedding_model = None


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
    """Build full paper content context (instead of just summaries)"""
    parts = ["=== SAFI RESEARCH PAPERS - FULL CONTENT ===\n"]
    parts.append("You have access to the complete text of the following research papers:\n")
    
    total_chars = 0
    for name, text in papers_combined.items():
        # Truncate very long papers if needed
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
    
    # Log approximate token count (rough estimate: 1 token ≈ 4 chars)
    approx_tokens = total_chars // 4
    parts.insert(1, f"[Approximate context size: {total_chars:,} characters / ~{approx_tokens:,} tokens]\n")
    
    return "\n".join(parts)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(
    query: str, 
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[dict],
    top_k: int = 6,
    min_similarity: float = 0.35
) -> Tuple[str, List[str]]:
    """Retrieve relevant chunks using embeddings (for highlighting specific sections)"""
    if not embeddings:
        return "", []
    
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in embeddings
    ]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    filtered_indices = [i for i in top_indices if similarities[i] >= min_similarity]
    
    if not filtered_indices:
        return "", []
    
    retrieved_chunks = [chunks[i] for i in filtered_indices]
    sources = list(set([metadata[i].get('source', 'Unknown') for i in filtered_indices]))
    
    context = "\n\n---\n\n".join(retrieved_chunks)
    return context, sources


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
        # Retrieve most relevant chunks (to highlight specific sections)
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
        
        # Build highlighted sections (from RAG retrieval)
        highlighted_section = ""
        if detailed_context:
            highlighted_section = f"""
=== MOST RELEVANT SECTIONS (retrieved via semantic search) ===
These sections are most relevant to the current question:

{detailed_context}

=== END OF HIGHLIGHTED SECTIONS ===
"""
        
        # Combine Excel contexts
        excel_section = ""
        if preloaded_excel_context or uploaded_excel_context:
            excel_parts = []
            if preloaded_excel_context:
                excel_parts.append(preloaded_excel_context)
            if uploaded_excel_context:
                excel_parts.append(uploaded_excel_context)
            excel_section = "\n\n".join(excel_parts)
        
        prompt = f"""You are a research assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

You have access to:
1. FULL TEXT of all 17 SAFI research papers (complete content below)
2. HIGHLIGHTED SECTIONS retrieved specifically for this question (most relevant excerpts)
3. EXCEL DATA with fiber morphology measurements and other data

{full_papers_context}

{highlighted_section}

{excel_section}

{conversation}Current question: {message}

Instructions:
- You have the COMPLETE text of all papers - use this for comprehensive answers
- The highlighted sections show the most relevant parts for quick reference
- Use Excel data when questions relate to fiber properties, morphology, or comparisons
- Include specific numerical values with units when available
- Cite the paper name when referencing findings
- When using Excel data, specify which biomass/fiber type you're referencing
- Provide thorough, well-researched answers since you have full paper access
- If information isn't available in any source, say so clearly

Answer:"""
        
        response = model.generate_content(prompt)
        
        # Determine sources used
        excel_keywords = ['fiber', 'length', 'width', 'kappa', 'coarseness', 'curl', 'fines', 
                         'morphology', 'pulp', 'biomass', 'tissue', 'benchmark', 'compare',
                         'fwt', 'fibrilation', 'kink', 'application']
        
        if preloaded_excel_context and any(kw in message.lower() for kw in excel_keywords):
            if "Pre-loaded Excel Data" not in sources:
                sources.append("Pre-loaded Excel Data")
        
        if uploaded_excel_context and any(kw in message.lower() for kw in excel_keywords):
            if "Uploaded Excel Data" not in sources:
                sources.append("Uploaded Excel Data")
        
        return response.text, sources
    
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Please try again later.", []
        elif "too long" in error_msg.lower() or "token" in error_msg.lower():
            return "⚠️ Context too long. Try reducing paper content or ask a more specific question.", []
        return f"Error: {error_msg}", []


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (1 token ≈ 4 characters)"""
    return len(text) // 4


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "full_papers_context" not in st.session_state:
    st.session_state.full_papers_context = ""
if "paper_names" not in st.session_state:
    st.session_state.paper_names = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "preloaded_excel_df" not in st.session_state:
    st.session_state.preloaded_excel_df = None
if "preloaded_excel_context" not in st.session_state:
    st.session_state.preloaded_excel_context = ""
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_sheet_name" not in st.session_state:
    st.session_state.uploaded_sheet_name = None
if "uploaded_excel_context" not in st.session_state:
    st.session_state.uploaded_excel_context = ""
if "context_stats" not in st.session_state:
    st.session_state.context_stats = {}

# Load data on startup
if not st.session_state.initialized and model:
    with st.spinner("Loading SAFI knowledge base (full papers)..."):
        try:
            # Load embeddings and full paper content
            chunks, embeddings, metadata, papers_combined = load_embeddings_data(EMBEDDINGS_FILE)
            
            if chunks:
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.metadata = metadata
                st.session_state.paper_names = list(papers_combined.keys())
                
                # Build FULL paper context (not just summaries)
                st.session_state.full_papers_context = build_full_papers_context(papers_combined)
                
                # Calculate context stats
                papers_chars = len(st.session_state.full_papers_context)
                st.session_state.context_stats["papers_chars"] = papers_chars
                st.session_state.context_stats["papers_tokens"] = estimate_tokens(st.session_state.full_papers_context)
                
                st.session_state.initialized = True
            else:
                st.error(f"Could not load: {EMBEDDINGS_FILE}")
                
        except Exception as e:
            st.error(f"Error loading papers: {str(e)}")
    
    # Load pre-configured Excel file
    with st.spinner("Loading pre-configured Excel data..."):
        try:
            preloaded_df, preloaded_context = load_preloaded_excel(PRELOADED_EXCEL_FILE, PRELOADED_EXCEL_SHEET)
            if preloaded_df is not None:
                st.session_state.preloaded_excel_df = preloaded_df
                st.session_state.preloaded_excel_context = preloaded_context
                st.session_state.context_stats["excel_chars"] = len(preloaded_context)
                st.session_state.context_stats["excel_tokens"] = estimate_tokens(preloaded_context)
        except Exception as e:
            st.warning(f"Could not load pre-configured Excel: {str(e)}")

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### SAFI Research Intelligence")
    st.markdown("##### Full Context Mode 🚀")
    
    st.divider()
    
    # Context Status
    st.markdown("#### 📊 Context Status")
    
    if st.session_state.initialized:
        total_tokens = st.session_state.context_stats.get("papers_tokens", 0) + \
                      st.session_state.context_stats.get("excel_tokens", 0)
        
        st.markdown(f"""
        <div class="context-indicator">📄 {len(st.session_state.paper_names)} papers loaded</div>
        <div class="context-indicator">~{st.session_state.context_stats.get("papers_tokens", 0):,} tokens</div>
        """, unsafe_allow_html=True)
        
        if st.session_state.preloaded_excel_df is not None:
            st.markdown(f"""
            <div class="context-indicator">📊 Excel: {st.session_state.preloaded_excel_df.shape[0]} rows</div>
            """, unsafe_allow_html=True)
        
        st.caption(f"Total context: ~{total_tokens:,} tokens")
        st.progress(min(total_tokens / 1000000, 1.0), text=f"{total_tokens/1000000*100:.1f}% of 1M limit")
    
    st.divider()
    
    # Pre-loaded Excel info
    if st.session_state.preloaded_excel_df is not None:
        st.markdown("#### 📊 Pre-loaded Data")
        st.caption(f"File: {PRELOADED_EXCEL_FILE}")
        st.caption(f"Sheet: {PRELOADED_EXCEL_SHEET}")
        st.caption(f"Size: {st.session_state.preloaded_excel_df.shape[0]} × {st.session_state.preloaded_excel_df.shape[1]}")
        
        with st.expander("📋 Preview pre-loaded data"):
            st.dataframe(st.session_state.preloaded_excel_df.head(10), use_container_width=True)
        
        with st.expander("📊 Columns"):
            for col in st.session_state.preloaded_excel_df.columns:
                st.caption(f"• {col}")
    
    st.divider()
    
    # Additional Excel file upload section
    st.markdown("#### 📤 Upload Additional Data")
    uploaded_file = st.file_uploader(
        "Upload Excel file", 
        type=["xlsx", "xls"],
        help="Upload additional data to include in analysis"
    )
    
    if uploaded_file is not None:
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("Select sheet", sheet_names)
            else:
                selected_sheet = sheet_names[0]
            
            df = pd.read_excel(xl, sheet_name=selected_sheet)
            
            st.session_state.uploaded_df = df
            st.session_state.uploaded_sheet_name = selected_sheet
            st.session_state.uploaded_excel_context = build_excel_context(df, selected_sheet, is_preloaded=False)
            
            st.success(f"✓ Loaded {len(df)} rows × {len(df.columns)} cols")
            
            with st.expander("📋 Preview uploaded data"):
                st.dataframe(df.head(10), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.uploaded_df is not None:
        if st.button("🗑️ Clear Uploaded Data", use_container_width=True):
            st.session_state.uploaded_df = None
            st.session_state.uploaded_sheet_name = None
            st.session_state.uploaded_excel_context = ""
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Papers list
    if st.session_state.initialized:
        with st.expander("📄 Papers in Context"):
            for name in st.session_state.paper_names:
                st.caption(f"• {name}")
    
    st.divider()
    
    with st.expander("ℹ️ About"):
        st.write("""
**SAFI Research Intelligence - Full Context Mode**

This version loads the COMPLETE text of all 17 research papers into context, leveraging Gemini's 1M token window.

**Features:**
- 📄 Full paper content (not just summaries)
- 🔍 Semantic search for highlighting relevant sections
- 📊 Pre-loaded Excel data (fiber morphology)
- 📤 Additional Excel upload support
- 💬 Conversation history

**Powered by:** Gemini 2.5 Flash
        """)

# ============ MAIN CONTENT ============
if len(st.session_state.messages) == 0:
    # Show data status in header
    data_status = []
    if st.session_state.initialized:
        data_status.append(f"📄 {len(st.session_state.paper_names)} papers (full text)")
    if st.session_state.preloaded_excel_df is not None:
        data_status.append(f"📊 Excel data loaded")
    
    status_text = " • ".join(data_status) if data_status else "Configure API key to start"
    
    token_info = ""
    if st.session_state.context_stats:
        total = st.session_state.context_stats.get("papers_tokens", 0) + \
                st.session_state.context_stats.get("excel_tokens", 0)
        token_info = f"~{total:,} tokens in context"
    
    st.markdown(f"""
        <div class="main-header">
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🎍</div>
            <h1 style='font-size: 2.5rem; margin: 0; font-weight: 400;'>
                SAFI Research Intelligence
            </h1>
            <p style='color: #4a6b4a; font-size: 1rem; margin-top: 0.5rem;'>
                Full Context Mode - Complete Paper Access
            </p>
            <p style='color: #6a8a6a; font-size: 0.85rem; margin-top: 0.25rem;'>
                {status_text}
            </p>
            <p style='color: #8a9a8a; font-size: 0.75rem; margin-top: 0.25rem;'>
                {token_info}
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
        content = msg["content"]
        sources = msg.get("sources", [])
        
        sources_html = ""
        if sources:
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
if prompt := st.chat_input("Ask about SAFI research (full paper access)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing with full context..."):
        response, sources = generate_response(
            prompt,
            st.session_state.full_papers_context,
            st.session_state.chunks,
            st.session_state.embeddings,
            st.session_state.metadata,
            st.session_state.preloaded_excel_context,
            st.session_state.uploaded_excel_context,
            st.session_state.messages
        )
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    st.rerun()
