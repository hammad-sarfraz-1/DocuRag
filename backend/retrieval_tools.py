import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from backend.config import Config
from backend.embedding_store import VectorStore

logger = logging.getLogger(__name__)

vector_store = VectorStore()


@dataclass
class RetrievalResult:
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    source: str = ""


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index: dict[str, "BM25Okapi"] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, chat_id: str, documents: List[str]):
        from rank_bm25 import BM25Okapi

        tokenized = [self._tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._index[chat_id] = bm25

    def search(self, chat_id: str, query: str, k: int) -> List[Tuple[int, float]]:
        if chat_id not in self._index:
            return []
        bm25 = self._index[chat_id]
        scores = bm25.get_scores(self._tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]


_bm25_index = BM25Index()


def _ensure_bm25_index(chat_id: str):
    if chat_id not in _bm25_index._index:
        docs = vector_store.get_all_documents(chat_id)
        if docs:
            _bm25_index.build(chat_id, docs)


def rebuild_bm25_index(chat_id: str):
    docs = vector_store.get_all_documents(chat_id)
    if docs:
        _bm25_index.build(chat_id, docs)


def search_vector(chat_id: str, query: str, k: int = Config.RETRIEVAL_K) -> List[RetrievalResult]:
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
    vec_results = search_vector(chat_id, query, k=k * 2)
    bm25_results = search_bm25(chat_id, query, k=k * 2)

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


def search_all_sources(chat_id: str, query: str, use_web: bool = True) -> List[RetrievalResult]:
    """Always search documents first (hybrid), optionally supplement with web."""
    doc_results = search_hybrid(chat_id, query)

    if use_web and Config.TAVILY_API_KEY:
        web_results = search_web(query, k=2)
        doc_results.extend(web_results)

    return doc_results


class Reranker:
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
