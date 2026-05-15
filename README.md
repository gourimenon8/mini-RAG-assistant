# Mini-RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) prototype that answers questions across uploaded research PDFs. Built with sentence-transformers, FAISS, Claude (Anthropic), and Streamlit.

---

## Architecture

```
PDF Upload
  → PyMuPDF text extraction (per-page)
  → Word-level chunking with configurable size + overlap
  → all-MiniLM-L6-v2 embeddings (384-dim, L2-normalized)
  → FAISS IndexFlatIP (inner product on normalized = cosine similarity)
  → Top-k retrieval with confidence scores
  → Claude (grounded generation with [Source N] citations)
  → Streamlit chat UI
```

### Why these choices?
| Component | Choice | Reason |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Free, local, 384-dim, strong semantic retrieval for ~100M params |
| Vector store | FAISS `IndexFlatIP` | Exact cosine search, no server, deterministic, fast on CPU |
| LLM | Claude Sonnet | Strong instruction following, reliable citation behavior |
| PDF parsing | PyMuPDF (fitz) | Fast, handles multi-column layouts better than pdfplumber |
| UI | Streamlit | One-file deploy, Streamlit Cloud free tier |

### Confidence scoring
Cosine similarity is used directly as a relevance score (0–1). Scores are color-coded:
- 🟢 ≥ 65% — High relevance
- 🟡 45–64% — Medium relevance
- 🔴 < 45% — Low relevance

---

## Setup

### Local

```bash
git clone <repo>
cd mini-rag-assistant
pip install -r requirements.txt

# Add your API key
export ANTHROPIC_API_KEY=sk-ant-...

streamlit run app.py
```

### Streamlit Cloud
1. Push repo to GitHub (no secrets committed)
2. Go to share.streamlit.io → deploy
3. Add `ANTHROPIC_API_KEY` under **App Settings → Secrets**
4. Or enter the key directly in the app's sidebar at runtime

---

## Usage

1. Enter your Anthropic API key in the sidebar
2. Upload one or more research PDFs
3. Adjust chunk size and retrieval settings if needed
4. Click **Process Documents** — chunks are embedded and indexed
5. Ask questions in the chat input
6. Expand **Retrieved Sources** under any answer to inspect which chunks were used and their relevance scores

---

## Key design decisions

**Grounding over fluency**: Claude is instructed to cite `[Source N]` for every factual claim and explicitly say when information is absent from the context — prioritizing accuracy over a smooth-sounding answer.

**Chunk overlap**: Default overlap of 50 words prevents context loss at chunk boundaries, which is a common failure mode in naive RAG systems.

**Multi-turn memory**: The last 3 Q&A turns are included in each Claude call so follow-up questions resolve references correctly (e.g., "explain that in more detail").

**No external vector DB**: FAISS runs entirely in-memory with no server dependency, making the demo portable and free.

---

## Potential enhancements (given more time)
- **Hybrid search**: BM25 keyword search + dense retrieval re-ranked together
- **Precision@k evaluation**: Auto-evaluate retrieval quality on a labeled test set
- **MMR-based chunk selection**: Reduce redundancy in retrieved context
- **Source highlighting**: Highlight matched text spans in a PDF viewer
- **Persistent index**: Serialize FAISS index to disk so users don't re-process on every session
- **Multi-format support**: DOCX, TXT, web URLs via trafilatura

---

## Dependencies

```
anthropic>=0.40.0          # Claude API
streamlit>=1.40.0          # UI + deployment
faiss-cpu>=1.8.0           # Vector search
sentence-transformers>=3.0 # Embeddings
PyMuPDF>=1.24.0            # PDF parsing
numpy>=1.26.0
```

---

## Sample output

**Query:** *What methods do the papers use for uncertainty quantification?*

**Answer:**
> The papers describe two primary approaches to uncertainty quantification. [Source 1] presents a conformal prediction framework that constructs distribution-free prediction sets with guaranteed coverage. [Source 3] uses Monte Carlo dropout during inference to estimate epistemic uncertainty, reporting that this reduces overconfident predictions by 23% on out-of-distribution inputs. [Source 2] does not directly address uncertainty quantification.

**Retrieved Sources:**
| # | File | Page | Relevance |
|---|------|------|-----------|
| 1 | paper_A.pdf | 4 | 78.3% |
| 2 | paper_B.pdf | 7 | 61.1% |
| 3 | paper_C.pdf | 2 | 55.4% |
