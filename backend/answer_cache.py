import json
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from backend.config import Config

_client = chromadb.PersistentClient(path=Config.PERSIST_DIR)
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=Config.EMBEDDING_MODEL
)


def _safe_collection_name(chat_id: str) -> str:
    return chat_id.replace("-", "_")


def _collection(chat_id: str):
    name = f"cache_{_safe_collection_name(chat_id)}"
    return _client.get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def get(chat_id: str, question: str) -> Optional[dict]:
    """Return {"answer", "citations", "similarity", "matched_question"} on a
    semantic hit, else None. Use `best_match` to see the nearest score even
    on a miss (for logging/debugging why a question didn't hit)."""
    collection = _collection(chat_id)
    if collection.count() == 0:
        return None
    results = collection.query(query_texts=[question], n_results=1)
    if not results["documents"] or not results["documents"][0]:
        return None
    distance = results["distances"][0][0]
    similarity = 1 - distance
    if similarity < Config.CACHE_SIMILARITY_THRESHOLD:
        return None
    metadata = results["metadatas"][0][0]
    return {
        "answer": metadata["answer"],
        "citations": json.loads(metadata["citations"]),
        "similarity": similarity,
        "matched_question": results["documents"][0][0],
    }


def best_match(chat_id: str, question: str) -> Optional[dict]:
    """Return {"similarity", "matched_question"} for the nearest cached
    question regardless of threshold, or None if the cache is empty."""
    collection = _collection(chat_id)
    if collection.count() == 0:
        return None
    results = collection.query(query_texts=[question], n_results=1)
    if not results["documents"] or not results["documents"][0]:
        return None
    return {
        "similarity": 1 - results["distances"][0][0],
        "matched_question": results["documents"][0][0],
    }


def put(chat_id: str, question: str, answer: str, citations: List[dict]):
    collection = _collection(chat_id)
    collection.add(
        documents=[question],
        ids=[f"q_{collection.count()}"],
        metadatas=[{"answer": answer, "citations": json.dumps(citations)}],
    )


def clear(chat_id: str):
    name = f"cache_{_safe_collection_name(chat_id)}"
    try:
        _client.delete_collection(name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass


def clear_all():
    """Clear every chat's cache. Documents are shared across chats, so a
    change to the shared corpus can make ANY chat's cached answers stale."""
    for collection in _client.list_collections():
        if collection.name.startswith("cache_"):
            try:
                _client.delete_collection(collection.name)
            except (ValueError, chromadb.errors.NotFoundError):
                pass
