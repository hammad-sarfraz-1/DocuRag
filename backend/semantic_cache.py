"""
Semantic Cache
==============

Caches (query, response, citations) pairs and retrieves them when a new
query is semantically similar to a previously seen one.  Uses the same
sentence-transformer model as the vector store for embeddings.

This significantly reduces latency and LLM cost for repeated or very
similar questions within the same session.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from chromadb.utils import embedding_functions

from backend.config import Config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query: str
    answer: str
    citations: List[dict] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class SemanticCache:
    """LRU-ish semantic cache with configurable similarity threshold.

    Keys are chat_id + normalized query embedding.  When a query's
    embedding has cosine similarity ≥ *threshold* with a stored entry,
    the cached answer is returned.
    """

    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        threshold: float = Config.SEMANTIC_CACHE_SIMILARITY,
        max_size: int = Config.SEMANTIC_CACHE_SIZE,
    ):
        self.threshold = threshold
        self.max_size = max_size
        self._entries: Dict[str, List[CacheEntry]] = {}  # chat_id → entries
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, chat_id: str, query: str) -> Optional[Tuple[str, List[dict]]]:
        """Return cached (answer, citations) if a similar query exists."""
        if not Config.ENABLE_SEMANTIC_CACHE:
            return None

        entries = self._entries.get(chat_id)
        if not entries:
            return None

        q_emb = self._embed(query)
        best_entry = None
        best_sim = -1.0

        for entry in entries:
            if entry.embedding is None:
                continue
            sim = self._cosine_similarity(q_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.threshold:
            logger.info(
                "Semantic cache HIT (sim=%.3f) for: %.60s", best_sim, query
            )
            return best_entry.answer, best_entry.citations

        return None

    def put(self, chat_id: str, query: str, answer: str, citations: List[dict]):
        """Store a query + response pair in the cache."""
        if not Config.ENABLE_SEMANTIC_CACHE:
            return

        if chat_id not in self._entries:
            self._entries[chat_id] = []

        entries = self._entries[chat_id]
        entry = CacheEntry(
            query=query,
            answer=answer,
            citations=citations,
            embedding=self._embed(query),
        )
        entries.append(entry)

        # Evict oldest if over capacity
        if len(entries) > self.max_size:
            entries.pop(0)

        logger.debug("Semantic cache PUT (now %d entries)", len(entries))

    def invalidate(self, chat_id: str):
        """Drop all cache entries for a chat (e.g. after new upload)."""
        self._entries.pop(chat_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        emb = self._ef([text])
        return np.array(emb[0])

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


# Singleton
semantic_cache = SemanticCache()
