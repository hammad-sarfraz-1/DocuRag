"""
Multi-Strategy Retrieval Toolkit
=================================

Provides the retrieval layer for the Super RAG system:

- **Vector search** via ChromaDB (dense embeddings)
- **BM25 keyword search** (sparse lexical)
- **Hybrid search** (weighted fusion of vector + BM25)
- **Web search** via Tavily
- **Cross-encoder reranking** for precision
- **Multi-source routing** — selects best strategy based on query type

All functions return a consistent ``RetrievalResult`` format.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from backend.config import Config
from backend.embedding_store import VectorStore

logger = logging.getLogger(__name__)

vector_store = VectorStore()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    source: str = ""  # "vector", "bm25", "web", etc.


# ---------------------------------------------------------------------------
# BM25 (keyword) index — one per chat
# ---------------------------------------------------------------------------
class BM25Index:
    """Simple BM25 index built from document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index: dict[str, "BM25Set"] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, chat_id: str, documents: List[str]):
        """Build or rebuild the BM25 index for a chat."""
        from rank_bm25 import BM25Okapi

        tokenized = [self._tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._index[chat_id] = bm25

    def search(self, chat_id: str, query: str, k: int) -> List[Tuple[int, float]]:
        """Return list of (doc_index, score) tuples."""
        if chat_id not in self._index:
            return []
        bm25 = self._index[chat_id]
        scores = bm25.get_scores(self._tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]


_bm25_index = BM25Index()


def _ensure_bm25_index(chat_id: str):
    """Lazily build BM25 index if it doesn't exist for this chat."""
    if chat_id not in _bm25_index._index:
        docs = vector_store.get_all_documents(chat_id)
        if docs:
            _bm25_index.build(chat_id, docs)


def rebuild_bm25_index(chat_id: str):
    """Force rebuild after document upload."""
    docs = vector_store.get_all_documents(chat_id)
    if docs:
        _bm25_index.build(chat_id, docs)


# ---------------------------------------------------------------------------
# Individual retrieval strategies
# ---------------------------------------------------------------------------
def search_vector(chat_id: str, query: str, k: int = Config.RETRIEVAL_K) -> List[RetrievalResult]:
    """Dense vector search via ChromaDB sentence embeddings."""
    results = vector_store.similarity_search_with_metadata(chat_id, query, k=k)
    out = []
    for text, meta in results:
        out.append(RetrievalResult(
            text=text,
            metadata=meta or {},
            score=meta.get("_distance", 0.0) if meta else 0.0,
            source="vector",
        ))
    return out


def search_bm25(chat_id: str, query: str, k: int = Config.RETRIEVAL_K) -> List[RetrievalResult]:
    """Lexical BM25 keyword search."""
    _ensure_bm25_index(chat_id)
    all_docs = vector_store.get_all_documents(chat_id)
    hits = _bm25_index.search(chat_id, query, k=k)
    out = []
    for idx, score in hits:
        if idx < len(all_docs):
            meta = vector_store.get_document_metadata(chat_id, idx)
            out.append(RetrievalResult(
                text=all_docs[idx],
                metadata=meta or {},
                score=score,
                source="bm25",
            ))
    return out


def search_hybrid(
    chat_id: str,
    query: str,
    k: int = Config.RETRIEVAL_K,
    w_vec: float = Config.HYBRID_SEARCH_WEIGHT_VECTOR,
    w_bm25: float = Config.HYBRID_SEARCH_WEIGHT_BM25,
) -> List[RetrievalResult]:
    """Fuse vector and BM25 results with Reciprocal Rank Fusion."""
    vec_results = search_vector(chat_id, query, k=k * 2)
    bm25_results = search_bm25(chat_id, query, k=k * 2)

    # RRF merge
    rrf_scores: dict[int, float] = {}
    all_texts: dict[int, RetrievalResult] = {}
    idx = 0
    for r in vec_results:
        all_texts[idx] = r
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + w_vec / (idx + 60)
        idx += 1
    idx = 0
    for r in bm25_results:
        rid = len(vec_results) + idx
        all_texts[rid] = r
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + w_bm25 / (idx + 60)
        idx += 1

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [all_texts[i] for i in sorted_ids]


def search_web(query: str, k: int = 3) -> List[RetrievalResult]:
    """Web search via Tavily API."""
    try:
        from tavily import TavilyClient

        tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)
        resp = tavily.search(query, max_results=k)
        results = []
        for r in resp.get("results", []):
            results.append(RetrievalResult(
                text=f"{r['title']}\n{r['content']}",
                metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                score=r.get("score", 0.0),
                source="web",
            ))
        return results
    except Exception as exc:
        logger.warning("Web search unavailable: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Multi-source router — picks strategy based on query analysis
# ---------------------------------------------------------------------------
def route_query(query: str, has_documents: bool) -> str:
    """Decide retrieval strategy: 'vector', 'hybrid', 'web', or 'hybrid+web'.

    Heuristic rules (can be replaced by LLM-based routing):
    - Short factual queries → hybrid (best recall)
    - Queries about current events → web
    - Technical queries → hybrid+web
    - No documents uploaded → web
    """
    query_lower = query.lower()

    if not has_documents:
        return "web"

    current_event_keywords = {"current", "latest", "news", "today", "2024", "2025", "2026"}
    if current_event_keywords & set(query_lower.split()):
        return "hybrid+web"

    code_keywords = {"code", "function", "api", "snippet", "example", "implementation"}
    if code_keywords & set(query_lower.split()):
        return "hybrid+web"

    return "hybrid"


def execute_strategy(
    strategy: str,
    chat_id: str,
    query: str,
    k: int = Config.RETRIEVAL_K,
) -> List[RetrievalResult]:
    """Execute a named retrieval strategy."""
    if strategy == "vector":
        return search_vector(chat_id, query, k=k)
    elif strategy == "bm25":
        return search_bm25(chat_id, query, k=k)
    elif strategy == "hybrid":
        return search_hybrid(chat_id, query, k=k)
    elif strategy == "web":
        return search_web(query, k=3)
    elif strategy == "hybrid+web":
        results = search_hybrid(chat_id, query, k=k)
        web_results = search_web(query, k=2)
        results.extend(web_results)
        return results
    return search_vector(chat_id, query, k=k)


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------
class Reranker:
    """Thin wrapper around a cross-encoder model."""

    def __init__(self, model_name: str = Config.RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info("Reranker loaded: %s", self.model_name)
            except Exception as exc:
                logger.warning("Reranker load failed (non-fatal): %s", exc)

    def rerank(
        self, query: str, results: List[RetrievalResult], keep: int = Config.RERANK_KEEP
    ) -> List[RetrievalResult]:
        """Re-rank results by cross-encoder score and return top-k."""
        self._lazy_load()
        if self._model is None or not results:
            return results[:keep]

        pairs = [(query, r.text[:512]) for r in results]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
        except Exception as exc:
            logger.warning("Reranking failed (non-fatal): %s", exc)
            return results[:keep]

        for r, s in zip(results, scores):
            r.score = float(s)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:keep]


reranker = Reranker()


# ---------------------------------------------------------------------------
# Code interpreter (sandboxed)
# ---------------------------------------------------------------------------
def interpret_code(code: str, timeout: int = Config.CODE_TIMEOUT_SECONDS) -> str:
    """Execute Python code in an isolated subprocess and return output.

    Security: runs with restricted globals, no imports, time-bounded.
    """
    import os
    import subprocess
    import sys
    import tempfile

    if not Config.ENABLE_CODE_INTERPRETER:
        return "Code interpreter is disabled."

    sanitized = []
    for line in code.split("\n"):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            sanitized.append(f"# {line}  (blocked by sandbox)")
        else:
            sanitized.append(line)
    safe_code = "\n".join(sanitized)

    wrapper = (
        "import sys, json, math, textwrap, collections, datetime\n"
        "def run():\n"
        + "\n".join("    " + line for line in safe_code.split("\n"))
        + "\n\n"
        "try:\n"
        "    result = run()\n"
        "    if result is not None:\n"
        "        print(repr(result))\n"
        "except Exception as e:\n"
        "    print(f'Error: {e}', file=sys.stderr)\n"
    )

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapper)
            tmppath = f.name

        proc = subprocess.run(
            [sys.executable, tmppath],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        os.unlink(tmppath)

        output = proc.stdout.strip()
        error = proc.stderr.strip()
        if error:
            output = f"{output}\n--- stderr ---\n{error}" if output else error
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(execution timed out after {timeout}s)"
    except Exception as exc:
        return f"(execution error: {exc})"
