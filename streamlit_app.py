"""
SAFI Research Intelligence - Hybrid Version
Paper summaries (full context) + RAG retrieval for details
"""
import streamlit as st
import google.generativeai as genai
from typing import List, Tuple, Dict
import numpy as np
import os
import pickle
import json
import time

# Page config
st.set_page_config(
    page_title="SAFI Research Intelligence",
    page_icon="🎍",
    layout="centered",
    initial_sidebar_state="expanded"
)

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


def load_embeddings_data(embeddings_file: str) -> Tuple[List[str], List[List[float]], List[dict], Dict[str, str]]:
    """Load embeddings and group chunks by paper"""
    if not os.path.exists(embeddings_file):
        return [], [], [], {}
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data["knowledge_base"]
    embeddings = data["embeddings"]
    metadata = data["chunk_metadata"]
    
    # Group chunks by paper for summary generation
    papers_text = {}
    for chunk, meta in zip(chunks, metadata):
        source = meta.get('source', meta.get('file', 'Unknown'))
        if source not in papers_text:
            papers_text[source] = []
        papers_text[source].append(chunk)
    
    # Combine chunks per paper
    papers_combined = {
        name: "\n".join(chunks_list) 
        for name, chunks_list in papers_text.items()
    }
    
    return chunks, embeddings, metadata, papers_combined


def generate_paper_summary(paper_name: str, paper_text: str, max_chars: int = 8000) -> str:
    """Generate a structured summary of a paper"""
    # Truncate if too long to summarize
    if len(paper_text) > 50000:
        paper_text = paper_text[:50000] + "\n[...truncated for summarization...]"
    
    prompt = f"""Summarize this research paper in a structured format. Be concise but include all key quantitative findings.

PAPER: {paper_name}

{paper_text}

Provide a summary with these sections (use 2-3 sentences each, more for key findings):
1. **Objective**: What the paper investigates
2. **Methods**: Key methodology (LCA approach, system boundaries, etc.)
3. **Key Findings**: Most important numerical results with units (e.g., GWP values, carbon footprints)
4. **Conclusions**: Main takeaways

Keep total summary under 500 words. Prioritize numerical data."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Summary generation failed: {str(e)}"


def load_or_generate_summaries(papers_combined: Dict[str, str], cache_file: str = "safi_summaries.json") -> Dict[str, str]:
    """Load cached summaries or generate new ones"""
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            # Check if all papers are present
            if set(cached.keys()) == set(papers_combined.keys()):
                return cached
        except:
            pass
    
    # Generate summaries
    summaries = {}
    progress_bar = st.progress(0, text="Generating paper summaries...")
    
    for i, (name, text) in enumerate(papers_combined.items()):
        progress_bar.progress(
            (i + 1) / len(papers_combined), 
            text=f"Summarizing: {name[:50]}..."
        )
        summaries[name] = generate_paper_summary(name, text)
        time.sleep(0.5)  # Rate limiting
    
    progress_bar.empty()
    
    # Cache summaries
    try:
        with open(cache_file, 'w') as f:
            json.dump(summaries, f, indent=2)
    except:
        pass
    
    return summaries


def build_summaries_context(summaries: Dict[str, str]) -> str:
    """Build the summaries section for the prompt"""
    parts = ["=== SAFI RESEARCH PAPER SUMMARIES ===\n"]
    
    for name, summary in summaries.items():
        parts.append(f"### {name}\n{summary}\n")
    
    parts.append("=== END OF SUMMARIES ===")
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
    """Retrieve relevant chunks using embeddings"""
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
    summaries_context: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[dict],
    chat_history: List[dict] = None
) -> Tuple[str, List[str]]:
    """Generate response using hybrid approach"""
    if not model:
        return "⚠️ API key not configured.", []
    
    try:
        # Retrieve detailed chunks
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
        
        # Build detailed context section
        detailed_section = ""
        if detailed_context:
            detailed_section = f"""
=== DETAILED EXCERPTS (most relevant to your question) ===
{detailed_context}
=== END OF EXCERPTS ===
"""
        
        prompt = f"""You are a research assistant for the Sustainable & Alternative Fibers Initiative (SAFI).

You have access to:
1. SUMMARIES of all SAFI research papers (for overview and cross-paper questions)
2. DETAILED EXCERPTS retrieved specifically for this question (for precise data)

{summaries_context}

{detailed_section}

{conversation}Current question: {message}

Instructions:
- Use summaries for overview and cross-paper synthesis questions
- Use detailed excerpts for specific numerical values and methodology details
- Include specific values with units when available
- Cite the paper when referencing findings
- If information isn't available in either source, say so clearly

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text, sources
    
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Please try again later.", []
        return f"Error: {error_msg}", []


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "summaries_context" not in st.session_state:
    st.session_state.summaries_context = ""
if "paper_names" not in st.session_state:
    st.session_state.paper_names = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Load data on startup
if not st.session_state.initialized and model:
    with st.spinner("Loading SAFI knowledge base..."):
        try:
            embeddings_file = "safi_embeddings.pkl"
            
            chunks, embeddings, metadata, papers_combined = load_embeddings_data(embeddings_file)
            
            if chunks:
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.metadata = metadata
                st.session_state.paper_names = list(papers_combined.keys())
                
                # Generate/load summaries
                summaries = load_or_generate_summaries(papers_combined)
                st.session_state.summaries_context = build_summaries_context(summaries)
                
                st.session_state.initialized = True
            else:
                st.error(f"Could not load: {embeddings_file}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### SAFI Research Intelligence")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    if st.session_state.initialized:
        st.metric("Papers", len(st.session_state.paper_names))
        
        with st.expander("📄 Papers"):
            for name in st.session_state.paper_names:
                st.caption(f"• {name}")
    
    st.divider()
    
    with st.expander("ℹ️ About"):
        st.write("""
 More than 30 global entities working together to study, develop and promote the utilization of alternative fibers for packaging, hygiene, nonwoven and textile products
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
                Ask about SAFI research
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
            sources_html = f'<div class="sources-box"><strong>Detailed sources:</strong><br>{sources_list}</div>'
        
        st.markdown(f"""
            <div class="assistant-message">
                <div class="message-label">SAFI Research Intelligence</div>
                {content}
                {sources_html}
            </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about SAFI research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing..."):
        response, sources = generate_response(
            prompt,
            st.session_state.summaries_context,
            st.session_state.chunks,
            st.session_state.embeddings,
            st.session_state.metadata,
            st.session_state.messages
        )
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    st.rerun()
