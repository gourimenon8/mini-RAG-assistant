"""
rag_pipeline.py — Core RAG logic
  PDF parsing → chunking → embedding → FAISS index → retrieval → Claude generation
"""

import re
import numpy as np
import faiss
import fitz  # PyMuPDF
import anthropic
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output dimension
CLAUDE_MODEL   = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def load_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract per-page text from a PDF byte stream."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": filename,
            })
    doc.close()
    return pages


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split page text into overlapping word-level chunks.
    Each chunk carries its source filename and page number.
    """
    chunks = []
    for page in pages:
        words = page["text"].split()
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if len(chunk_words) < 20:          # skip tiny tail fragments
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source": page["source"],
                "page": page["page"],
            })
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def make_embedder() -> SentenceTransformer:
    """Load the sentence-transformer model (called once; caller should cache)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings (float32) so inner-product == cosine sim."""
    vecs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Wraps the full retrieval-augmented generation workflow.

    Usage:
        pipeline = RAGPipeline(api_key="sk-ant-...", embedder=cached_embedder)
        n_chunks = pipeline.add_documents([(bytes, "paper.pdf"), ...])
        result   = pipeline.query("What is the main finding?")
        # result = {"answer": str, "sources": [...]}
    """

    def __init__(self, api_key: str, embedder: SentenceTransformer):
        import os
        self.api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client   = anthropic.Anthropic(api_key=self.api_key)
        self.embedder = embedder

        # FAISS index (cosine via inner product on normalized vectors)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: List[Dict] = []      # parallel to index rows

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(
        self,
        files: List[Tuple[bytes, str]],
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> int:
        """
        Process a list of (bytes, filename) tuples, embed, and add to FAISS.
        Returns total chunk count.
        """
        all_chunks: List[Dict] = []
        for file_bytes, filename in files:
            pages  = load_pdf(file_bytes, filename)
            chunks = chunk_pages(pages, chunk_size, overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No readable text found. Check that the PDFs are not scanned images.")

        texts      = [c["text"] for c in all_chunks]
        embeddings = embed(self.embedder, texts)

        # (Re-)build index — always fresh on each call
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        self.chunk_metadata = all_chunks

        return len(all_chunks)

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """
        Return top-k chunks with cosine similarity scores.
        Score is in [0, 1]; higher = more relevant.
        """
        if not self.is_ready:
            return []

        k = min(k, self.index.ntotal)
        q_vec = embed(self.embedder, [query])
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.chunk_metadata[idx]
            results.append({
                **meta,
                "score":     float(score),
                "score_pct": f"{float(score) * 100:.1f}%",
            })
        return results

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        sources: List[Dict],
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a grounded answer from Claude using retrieved context.
        Claude is instructed to cite [Source N] inline.
        """
        # Build numbered context block
        context_block = ""
        for i, src in enumerate(sources):
            context_block += (
                f"[Source {i+1} | {src['source']} | Page {src['page']}]\n"
                f"{src['text']}\n\n"
            )

        system = (
            "You are a precise research assistant. Your answers must be grounded "
            "exclusively in the context chunks below. Follow these rules:\n"
            "1. Cite every factual claim with [Source N] immediately after it.\n"
            "2. If the answer is not found in the context, say: "
            "'This information is not present in the provided documents.'\n"
            "3. Do not fabricate facts or use outside knowledge.\n"
            "4. Be concise and structured. Use bullet points where helpful.\n"
            "5. If sources conflict, note the discrepancy explicitly."
        )

        user_msg = (
            f"--- CONTEXT ---\n{context_block}"
            f"--- QUESTION ---\n{query}"
        )

        # Include recent conversation turns for multi-turn support
        messages: List[Dict] = []
        if chat_history:
            for turn in chat_history[-6:]:     # cap at last 3 Q&A pairs
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_msg})

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: int = 4,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run the full RAG loop and return a structured result dict:
            {
                "answer":  str,
                "sources": [{"text", "source", "page", "score", "score_pct"}, ...]
            }
        """
        if not self.is_ready:
            return {
                "answer": "⚠️ No documents indexed yet. Please upload PDFs and click **Process Documents**.",
                "sources": [],
            }

        sources = self.retrieve(question, k=k)
        answer  = self.generate(question, sources, chat_history)
        return {"answer": answer, "sources": sources}
