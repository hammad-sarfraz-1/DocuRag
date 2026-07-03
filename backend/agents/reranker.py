import math

from backend.config import Config
from backend.retrieval_tools import RetrievalResult, reranker
from backend.agents.state import RagState


def reranker_agent(state: RagState) -> dict:
    """Cross-encoder reranking, plus a calibrated 0..1 confidence score for
    the web-search gate (the raw cross-encoder logit isn't bounded, so it
    isn't directly comparable to WEB_FALLBACK_SCORE_THRESHOLD)."""
    query = state["query"]
    doc_results = [RetrievalResult(**r) for r in state.get("doc_results", [])]

    if not doc_results:
        return {"ranked_results": [], "retrieval_confidence": 0.0}

    if Config.ENABLE_RERANKING:
        total = len(doc_results)
        keep = total if total <= Config.SMALL_CORPUS_CHUNKS else Config.RERANK_KEEP
        ranked = reranker.rerank(query, doc_results, keep=keep)
    else:
        ranked = doc_results

    top_score = ranked[0].score if ranked else float("-inf")
    confidence = 1.0 / (1.0 + math.exp(-top_score))

    return {
        "ranked_results": [r.__dict__ for r in ranked],
        "retrieval_confidence": confidence,
    }


def route_after_reranker(state: RagState) -> str:
    # Only reached when has_documents is True (route_after_planner gates
    # entry to "retrieval"), so the gate is purely about retrieval quality.
    low_confidence = state.get("retrieval_confidence", 0.0) < Config.WEB_FALLBACK_SCORE_THRESHOLD
    if low_confidence and Config.TAVILY_API_KEY:
        return "web_search"
    return "synthesizer"
