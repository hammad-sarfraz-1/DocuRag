from backend.config import Config
from backend.retrieval_tools import RetrievalResult, search_hybrid
from backend.agents.state import RagState
from backend.agents.utils import vector_store


def retrieval_agent(state: RagState) -> dict:
    """Hybrid (BM25 + vector) retrieval. Widens k on retry rounds instead of
    rewriting the query — the simplest lever that makes a second round
    actually different from the first."""
    chat_id = state["chat_id"]
    query = state["query"]
    round_num = state.get("retrieval_round", 0) 

    all_chunks = vector_store.get_all_documents_with_metadata(chat_id)
    total = len(all_chunks)

    if total == 0:
        return {"doc_results": []}

    if total <= Config.SMALL_CORPUS_CHUNKS:
        # Small corpus: hand every chunk downstream so the right document is
        # always present; the Reranker orders them, never drops them.
        results = [
            RetrievalResult(text=text, metadata=meta or {}, source="document")
            for text, meta in all_chunks
        ]
    else:
        k = Config.RETRIEVAL_K * (round_num + 1)
        results = search_hybrid(chat_id, query, k=k)

    return {"doc_results": [r.__dict__ for r in results]}
