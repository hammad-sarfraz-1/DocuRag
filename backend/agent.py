"""
Agentic RAG Pipeline (LangGraph)
=================================

A multi-node LangGraph that:

1. **Analyzes** the user query (rewrites follow-ups, extracts keywords).
2. **Retrieves** relevant document chunks from the vector store.
3. **Grades** retrieved chunks for relevance.
4. **Generates** an answer grounded in the retrieved context, complete
   with source citations.
5. **Augments with web search** when document context is insufficient.
6. **Validates** the final answer is grounded in the provided sources.

Architecture::

    analyze_query → retrieve → grade_documents
        │                           │
        │                    ┌──────┴──────┐
        │               [relevant]    [not relevant]
        │                    │              │
        │                    ↓              ↓
        │              generate      web_search
        │                    │              │
        │                    └──────┬───────┘
        │                           ↓
        │                     check_groundedness
        │                           │
        │                     [pass / fail]
        │                           │
        └───────────────────────────→ END
"""

import logging
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from backend.config import Config
from backend.embedding_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM & shared state
# ---------------------------------------------------------------------------
llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MODEL_NAME)
vector_store = VectorStore()


class Citation(TypedDict):
    source: str
    chunk_index: int
    snippet: str


class AgentState(TypedDict):
    input: str
    chat_id: str
    history: List[Dict[str, str]]
    rewritten_query: str
    retrieved_results: List[str]
    retrieved_metadata: List[dict]
    citations: List[Citation]
    web_context: str
    answer: str
    need_web: bool
    grounded: bool
    iteration: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_history(history: List[Dict[str, str]], max_exchanges: int = 6) -> str:
    lines = []
    for msg in history[-max_exchanges:]:
        lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    return "\n".join(lines)


def _format_citations(citations: List[Citation]) -> str:
    if not citations:
        return ""
    parts = []
    seen = set()
    for i, c in enumerate(citations, 1):
        key = (c["source"], c["chunk_index"])
        if key not in seen:
            seen.add(key)
            snippet = (c["snippet"][:100] + "...") if len(c["snippet"]) > 100 else c["snippet"]
            parts.append(f"[{i}] {c['source']} (chunk {c['chunk_index'] + 1}) — {snippet}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def analyze_query(state: AgentState) -> dict:
    """Rewrite the user query to be self-contained and extract search terms."""
    history_str = _format_history(state["history"])
    prompt = (
        "You are a query-analysis assistant. Given the conversation history and "
        "the current user query, rewrite the query to be fully self-contained "
        "for a document search. If it is not a follow-up, return the original.\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"Current query: {state['input']}\n\n"
        "Respond with ONLY the rewritten query, nothing else."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        rewritten = resp.content.strip()
    except Exception as exc:
        logger.warning("Query analysis failed, using original: %s", exc)
        rewritten = state["input"]

    return {"rewritten_query": rewritten, "iteration": state.get("iteration", 0) + 1}


def retrieve(state: AgentState) -> dict:
    """Fetch relevant document chunks + metadata from the vector store."""
    query = state.get("rewritten_query") or state["input"]
    results = vector_store.similarity_search_with_metadata(
        state["chat_id"], query, k=Config.RETRIEVAL_K
    )
    if not results:
        return {
            "retrieved_results": [],
            "retrieved_metadata": [],
            "citations": [],
        }

    docs, metas = zip(*results) if results else ([], [])
    citations = []
    for meta in metas:
        if meta and "source" in meta:
            citations.append(
                Citation(
                    source=meta["source"],
                    chunk_index=meta.get("chunk_index", 0),
                    snippet=(docs[metas.index(meta)][:200] if metas.index(meta) < len(docs) else ""),
                )
            )
    return {
        "retrieved_results": list(docs),
        "retrieved_metadata": list(metas),
        "citations": citations,
    }


def grade_documents(state: AgentState) -> dict:
    """Check whether any retrieved document is relevant to the query."""
    if not state["retrieved_results"]:
        return {"need_web": True}

    # If we have citations, consider it relevant (fast path)
    if state.get("citations"):
        return {"need_web": False}

    # Ask the LLM for a quick relevance check
    query = state.get("rewritten_query") or state["input"]
    context = "\n\n".join(state["retrieved_results"][:2])
    prompt = (
        f"Query: {query}\n\n"
        f"Document snippet:\n{context[:1000]}\n\n"
        "Does this document contain information that helps answer the query? "
        "Answer ONLY 'yes' or 'no'."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        answer = resp.content.strip().lower()
        need_web = "no" in answer
    except Exception:
        need_web = True

    return {"need_web": need_web}


def generate_with_citations(state: AgentState) -> dict:
    """Answer the query grounded in retrieved documents + citations."""
    query = state.get("rewritten_query") or state["input"]
    history_str = _format_history(state["history"])
    doc_context = "\n\n".join(state["retrieved_results"]) if state["retrieved_results"] else "No documents."
    citations_str = _format_citations(state.get("citations", []))

    if state.get("need_web") and state.get("web_context"):
        extra = f"\n\nWeb search results:\n{state['web_context']}"
    else:
        extra = ""

    prompt = (
        "You are a precise RAG assistant. Answer the user's question using "
        "the document context below. When you use information from a specific "
        "source, cite it like [1], [2] etc.\n\n"
        f"Document context:\n{doc_context}\n"
        f"{extra}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"User: {query}\n\n"
        "Answer:"
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        answer = resp.content.strip()
    except Exception as exc:
        logger.exception("Generation failed")
        answer = f"I encountered an error generating the response: {exc}"

    return {"answer": answer}


def web_search_and_answer(state: AgentState) -> dict:
    """Fallback: search the web via Tavily, then generate a combined answer."""
    web_context = ""
    try:
        from tavily import TavilyClient

        tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)
        resp = tavily.search(state.get("rewritten_query") or state["input"], max_results=3)
        web_results = []
        for r in resp.get("results", []):
            web_results.append(f"Source: {r['title']}\n{r['content']}")
        web_context = "\n\n".join(web_results) if web_results else "No web results."
    except Exception as exc:
        web_context = f"Web search unavailable: {exc}"

    return {"web_context": web_context, "need_web": False}


def check_groundedness(state: AgentState) -> dict:
    """Verify the answer is supported by the sources."""
    if not state.get("citations") and not state.get("web_context"):
        return {"grounded": True}  # nothing to check

    prompt = (
        f"Answer: {state['answer'][:1500]}\n\n"
        f"Sources:\n{_format_citations(state.get('citations', []))[:1500]}\n"
        f"\nIs every claim in the answer supported by at least one source? "
        f"Answer ONLY 'yes' or 'no'."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        grounded = "yes" in resp.content.strip().lower()
    except Exception:
        grounded = True
    return {"grounded": grounded}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def route_after_grade(state: AgentState) -> str:
    return "web_search" if state.get("need_web", False) else "generate"


def route_after_grounded(state: AgentState) -> str:
    return END if state.get("grounded", True) else "generate"  # re-generate if not grounded


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate_with_citations)
workflow.add_node("web_search", web_search_and_answer)
workflow.add_node("check_groundedness", check_groundedness)

workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_after_grade,
    {"generate": "generate", "web_search": "web_search"},
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "check_groundedness")
workflow.add_conditional_edges(
    "check_groundedness",
    route_after_grounded,
    {END: END, "generate": "generate"},
)

graph = workflow.compile()


# ---------------------------------------------------------------------------
# Entry point called from ChatEngine
# ---------------------------------------------------------------------------
def supervisor_agent(user_input: str, chat_id: str, history: List[Dict[str, str]]) -> dict:
    """Run the full agentic RAG pipeline and return ``{"answer": ..., "citations": [...]}``."""
    initial: AgentState = {
        "input": user_input,
        "chat_id": chat_id,
        "history": history,
        "rewritten_query": "",
        "retrieved_results": [],
        "retrieved_metadata": [],
        "citations": [],
        "web_context": "",
        "answer": "",
        "need_web": False,
        "grounded": True,
        "iteration": 0,
    }
    try:
        final = graph.invoke(initial)
    except Exception as exc:
        logger.exception("Agent graph invocation failed")
        return {
            "answer": f"An internal error occurred: {exc}",
            "citations": [],
        }

    return {
        "answer": final.get("answer", "I could not generate an answer."),
        "citations": final.get("citations", []),
    }
