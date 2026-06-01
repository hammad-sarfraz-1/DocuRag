import json
import logging
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from backend.config import Config
from backend.embedding_store import VectorStore
from backend.retrieval_tools import (
    RetrievalResult,
    reranker,
    search_all_sources,
    search_web,
)

logger = logging.getLogger(__name__)

llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MODEL_NAME)
vector_store = VectorStore()


class Citation(TypedDict):
    source: str
    chunk_index: int
    snippet: str


class AgenticState(TypedDict):
    input: str
    chat_id: str
    history: List[Dict[str, str]]

    retrieval_results: List[Dict]
    context: str
    citations: List[Citation]
    answer: str

    has_documents: bool
    use_web: bool
    iteration: int


def _fmt(history: List[Dict], n: int = 6) -> str:
    lines = []
    for m in history[-n:]:
        lines.append(f"{m['role'].capitalize()}: {m['content']}")
    return "\n".join(lines)


def _retrieval_to_context(results: List[RetrievalResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        label = r.source.upper()
        parts.append(f"[{i}][{label}] {r.text[:600]}")
    return "\n\n".join(parts)


def _has_uploaded_docs(chat_id: str) -> bool:
    try:
        return vector_store.get_document_count(chat_id) > 0
    except Exception:
        return False


def _needs_web(query: str) -> bool:
    keywords = {"current", "latest", "news", "today", "2024", "2025", "2026",
                "recent", "update", "new", "now"}
    words = set(query.lower().split())
    return bool(keywords & words)


def plan_query(state: AgenticState) -> dict:
    has_docs = _has_uploaded_docs(state["chat_id"])
    needs_web = _needs_web(state["input"])

    if has_docs:
        use_web = needs_web
    else:
        use_web = True

    return {
        "has_documents": has_docs,
        "use_web": use_web,
    }


def retrieve(state: AgenticState) -> dict:
    chat_id = state["chat_id"]
    query = state["input"]
    use_web = state.get("use_web", False)

    all_results = search_all_sources(chat_id, query, use_web=use_web)

    if Config.ENABLE_RERANKING and all_results:
        all_results = reranker.rerank(query, all_results)

    context = _retrieval_to_context(all_results)

    citations: List[Citation] = []
    seen_cites = set()
    for r in all_results:
        meta = r.metadata
        if r.source == "web":
            url = meta.get("url", "") if meta else ""
            title = meta.get("title", "") if meta else ""
            key = ("web", url)
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append(Citation(
                    source=f"Web: {title}" if title else "Web search",
                    chunk_index=0,
                    snippet=r.text[:200],
                ))
        elif meta and "source" in meta:
            key = (meta["source"], meta.get("chunk_index", 0))
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append(Citation(
                    source=meta["source"],
                    chunk_index=meta.get("chunk_index", 0),
                    snippet=r.text[:200],
                ))

    return {
        "retrieval_results": [r.__dict__ for r in all_results],
        "context": context,
        "citations": citations,
    }


def synthesize(state: AgenticState) -> dict:
    history_str = _fmt(state["history"])
    context = state.get("context", "")
    has_docs = state.get("has_documents", False)

    system_msg = (
        "You are a precise RAG assistant. You must ground every answer in the "
        "retrieved context. When you use information from a source, cite it "
        "inline like [1], [2], etc. At the end, list the sources numbered.\n\n"
        "If the context does not contain enough information to answer, say so "
        "honestly rather than making up an answer."
    )

    instruction = (
        "Provide a clear, accurate answer. Cite sources for every factual claim. "
        "If both document sources and web sources are available, use both."
    )

    if has_docs:
        doc_note = "Documents have been uploaded and are the primary source. Always consult them first."
    else:
        doc_note = "No documents have been uploaded yet. Answer based on available context."

    prompt = (
        f"{instruction}\n\n"
        f"{doc_note}\n\n"
        f"Retrieved context:\n{context[:4000]}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"User: {state['input']}\n\n"
        "Answer:"
    )

    try:
        resp = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
        answer = resp.content.strip()
    except Exception as exc:
        logger.exception("Synthesis failed")
        answer = f"I encountered an error generating the response: {exc}"

    return {"answer": answer}


workflow = StateGraph(AgenticState)

workflow.add_node("plan_query", plan_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("synthesize", synthesize)

workflow.set_entry_point("plan_query")
workflow.add_edge("plan_query", "retrieve")
workflow.add_edge("retrieve", "synthesize")
workflow.add_edge("synthesize", END)

graph = workflow.compile()


def run_agentic_rag(
    user_input: str, chat_id: str, history: List[Dict[str, str]]
) -> dict:
    initial: AgenticState = {
        "input": user_input,
        "chat_id": chat_id,
        "history": history,
        "retrieval_results": [],
        "context": "",
        "citations": [],
        "answer": "",
        "has_documents": False,
        "use_web": False,
        "iteration": 0,
    }

    try:
        final = graph.invoke(initial)
    except Exception as exc:
        logger.exception("Agentic orchestrator failed")
        return {
            "answer": f"An internal error occurred: {exc}",
            "citations": [],
        }

    return {
        "answer": final.get("answer", "I could not generate an answer."),
        "citations": final.get("citations", []),
    }
