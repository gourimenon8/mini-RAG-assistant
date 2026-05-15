"""
app.py — Mini-RAG Assistant · Streamlit UI
"""

import streamlit as st
from rag_pipeline import RAGPipeline, make_embedder

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mini-RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Header */
.rag-title  { font-size:2rem; font-weight:700; color:#0f172a; margin-bottom:.1rem; }
.rag-sub    { color:#64748b; font-size:.9rem; margin-bottom:1.5rem; }

/* Source cards */
.src-card {
    border-left: 4px solid;
    padding: .65rem .9rem;
    border-radius: 4px;
    margin-bottom: .5rem;
    font-size: .84rem;
    background: #f8fafc;
}
.src-high { border-color: #16a34a; }
.src-med  { border-color: #d97706; }
.src-low  { border-color: #dc2626; }

/* Score badge */
.badge {
    display:inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size:.75rem;
    font-weight:600;
    margin-left: 6px;
}

/* Architecture diagram box */
.arch-box {
    background:#f1f5f9;
    border:1px solid #cbd5e1;
    border-radius:8px;
    padding:1rem 1.2rem;
    font-size:.85rem;
    line-height:1.8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached resources (survive reruns / users)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    """Load sentence-transformer once and cache across all sessions."""
    return make_embedder()


# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
defaults = {
    "messages": [],
    "pipeline": None,
    "ready":    False,
    "doc_count":  0,
    "file_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def score_style(score: float):
    """Return (css_class, hex_color, label) based on relevance score."""
    if score >= 0.65:
        return "src-high", "#16a34a", "High"
    if score >= 0.45:
        return "src-med",  "#d97706", "Medium"
    return "src-low",  "#dc2626", "Low"


def render_sources(sources):
    """Render retrieved source chunks inside an expander."""
    label = f"📚 Retrieved Sources ({len(sources)} chunks)"
    with st.expander(label):
        for i, src in enumerate(sources):
            css, color, level = score_style(src["score"])
            preview = src["text"][:320] + ("…" if len(src["text"]) > 320 else "")
            st.markdown(f"""
<div class="src-card {css}">
  <strong>Source {i+1}:</strong> <code>{src['source']}</code> · Page {src['page']}
  <span class="badge" style="background:{color}22;color:{color}">
    {src['score_pct']} · {level}
  </span>
  <br><br>
  <span style="color:#374151">{preview}</span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Key")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-… (optional if pre-configured)",
        help="Leave blank to use the server key. Your key is never stored.",
    )

    st.markdown("---")
    st.markdown("## 📄 Upload Documents")
    uploaded = st.file_uploader(
        "Research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers.",
    )

    with st.expander("⚙️ Chunking & Retrieval"):
        top_k      = st.slider("Chunks to retrieve (k)",  2, 8, 4)
        chunk_size = st.slider("Chunk size (words)",     200, 800, 400)
        overlap    = st.slider("Overlap (words)",         20, 120,  50)

    process = st.button("🔄 Process Documents", type="primary", use_container_width=True)

    if st.session_state.ready:
        st.success(
            f"✅ **{st.session_state.doc_count}** chunks indexed  \n"
            f"from **{st.session_state.file_count}** file(s)"
        )

    if st.session_state.ready and st.button("🗑️ Clear & Reset", use_container_width=True):
        for k in ("messages", "pipeline", "ready", "doc_count", "file_count"):
            st.session_state[k] = defaults[k]
        st.rerun()

    st.markdown("---")
    st.caption("Mini-RAG · Claude + FAISS + sentence-transformers")


# ─────────────────────────────────────────────
# Document processing
# ─────────────────────────────────────────────
if process:
    if not api_key:
        st.sidebar.error("Enter your Anthropic API key first.")
    elif not uploaded:
        st.sidebar.error("Upload at least one PDF.")
    else:
        with st.spinner("Chunking, embedding, and indexing…"):
            try:
                embedder = get_embedder()
                pipeline = RAGPipeline(api_key=api_key, embedder=embedder)
                files    = [(f.read(), f.name) for f in uploaded]
                n        = pipeline.add_documents(files, chunk_size=chunk_size, overlap=overlap)

                st.session_state.pipeline   = pipeline
                st.session_state.ready      = True
                st.session_state.doc_count  = n
                st.session_state.file_count = len(uploaded)
                st.session_state.messages   = []   # fresh chat per new index
                st.rerun()
            except Exception as e:
                st.error(f"Processing failed: {e}")


# ─────────────────────────────────────────────
# Main — title
# ─────────────────────────────────────────────
st.markdown('<div class="rag-title">🔍 Mini-RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rag-sub">'
    "Ask questions across uploaded research papers. "
    "Every answer is grounded in retrieved context with inline citations and confidence scores."
    "</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Welcome / how-it-works (shown before indexing)
# ─────────────────────────────────────────────
if not st.session_state.ready:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### How it works")
        st.markdown("""
1. **Upload** one or more research PDFs in the sidebar  
2. **Process** — text is chunked, embedded with `all-MiniLM-L6-v2`, and indexed in FAISS  
3. **Ask** any question in natural language  
4. **Retrieve** — the top-*k* most relevant chunks are fetched by cosine similarity  
5. **Generate** — Claude answers using *only* the retrieved context, citing `[Source N]` inline  
""")

    with col2:
        st.markdown("### Architecture")
        st.markdown("""
<div class="arch-box">
📄 PDF Upload<br>
↓ PyMuPDF text extraction<br>
↓ Word-level chunking (overlap)<br>
↓ <strong>all-MiniLM-L6-v2</strong> embeddings<br>
↓ <strong>FAISS</strong> IndexFlatIP (cosine sim)<br>
↓ Top-k retrieval + confidence scores<br>
↓ <strong>Claude</strong> grounded generation<br>
💬 Answer + cited sources
</div>""", unsafe_allow_html=True)

    st.info("👈 Upload PDFs in the sidebar and click **Process Documents** to begin.")
    st.stop()


# ─────────────────────────────────────────────
# Chat — replay history
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])


# ─────────────────────────────────────────────
# Chat — new input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build history for multi-turn context (exclude source dicts)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # RAG query — always refresh the client with the current sidebar key
    # so key changes take effect without rebuilding the FAISS index
    import anthropic as _anthropic
    st.session_state.pipeline.client = _anthropic.Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            result = st.session_state.pipeline.query(
                prompt,
                k=top_k,
                chat_history=history,
            )

        st.markdown(result["answer"])
        if result["sources"]:
            render_sources(result["sources"])

    # Persist to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
