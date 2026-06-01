"""
Multi-Agent Super RAG Orchestrator
===================================

A LangGraph state machine that implements the full agentic RAG loop:

┌──────────────────────────────────────────────────────────────────┐
│  check_cache ──[MISS]──► plan_query ──► retrieve_multi           │
│      │                       │              │                    │
│   [HIT]                      │              ▼                    │
│      │                       │         rerank_results            │
│      ▼                       │              │                    │
│   return ◄── store_cache ◄───┴──────┬───────┘                    │
│                                     │                            │
│                                critique_context                   │
│                                     │                            │
│                              ┌──────┴──────┐                     │
│                           [PASS]          [FAIL]                 │
│                              │              │                    │
│                              ▼              ▼                    │
│                         synthesize    reformulate (retry) ──────►│
│                              │                                    │
│                              ▼                                    │
│                         store_cache ──► return                    │
└──────────────────────────────────────────────────────────────────┘

Key agents
----------
- **Query Planner**: analyzes query, decomposes sub-questions, picks strategy
- **Retrieval Agent**: executes multi-strategy retrieval + reranking
- **Critic Agent**: validates context quality, triggers reformulation
- **Synthesis Agent**: builds final grounded answer with citations
- **Memory**: semantic cache + conversation history
"""

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
    execute_strategy,
    interpret_code,
    rebuild_bm25_index,
    reranker,
    route_query,
    search_hybrid,
    search_vector,
    search_web,
)
from backend.semantic_cache import SemanticCache, semantic_cache

logger = logging.getLogger(__name__)
# LLM
llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MODEL_NAME)
vector_store = VectorStore()


# State definition
class Citation(TypedDict):
    source: str
    chunk_index: int
    snippet: str


class QueryPlan(TypedDict):
    sub_questions: List[str]
    strategy: str
    needs_code: bool
    needs_web: bool
    complexity: str  # "simple" | "medium" | "complex"


class AgenticState(TypedDict):
    input: str
    chat_id: str
    history: List[Dict[str, str]]

    # Query planning
    query_plan: Optional[QueryPlan]

    # Retrieval
    retrieval_results: List[Dict]
    retrieval_strategy: str
    has_documents: bool

    # Critique loop
    critique_score: float
    critique_feedback: str
    retrieval_attempts: int

    # Synthesis
    context: str
    answer: str
    citations: List[Citation]

    # Tools
    code_result: str
    tool_calls: List[Dict]

    # Cache
    cache_hit: bool

    # Meta
    iteration: int


# Helpers
def _fmt(history: List[Dict], n: int = 6) -> str:
    lines = []
    for m in history[-n:]:
        lines.append(f"{m['role'].capitalize()}: {m['content']}")
    return "\n".join(lines)


def _citations_to_text(citations: List[Dict]) -> str:
    seen = set()
    parts = []
    for i, c in enumerate(citations, 1):
        key = (c["source"], c["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"[{i}] {c['source']} chunk {c['chunk_index']}")
    return "\n".join(parts)


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


# NODE 1 — Semantic Cache Lookup
def check_cache(state: AgenticState) -> dict:
    if not Config.ENABLE_SEMANTIC_CACHE:
        return {"cache_hit": False}

    cached = semantic_cache.get(state["chat_id"], state["input"])
    if cached is not None:
        answer, citations = cached
        return {
            "cache_hit": True,
            "answer": answer,
            "citations": citations,
        }
    return {"cache_hit": False}


# NODE 2 — Query Planner Agent
def plan_query(state: AgenticState) -> dict:
    """Analyze the query, decompose it, and choose a retrieval strategy."""
    history_str = _fmt(state["history"])
    has_docs = _has_uploaded_docs(state["chat_id"])

    prompt = (
        "You are a Query Planning Agent. Analyze the user's question and "
        "produce a JSON plan with these fields:\n"
        "- sub_questions: list of sub-questions to answer (one if simple)\n"
        "- strategy: 'hybrid', 'hybrid+web', 'web', or 'vector'\n"
        "- needs_code: true if the answer requires running/verifying code\n"
        "- needs_web: true if current/up-to-date info is required\n"
        "- complexity: 'simple', 'medium', or 'complex'\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"User question: {state['input']}\n\n"
        "Respond with ONLY valid JSON, no commentary."
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        # Strip markdown fences
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        plan = json.loads(content)
    except Exception as exc:
        logger.warning("Query planner fallback: %s", exc)
        plan = {
            "sub_questions": [state["input"]],
            "strategy": route_query(state["input"], has_docs),
            "needs_code": False,
            "needs_web": not has_docs,
            "complexity": "simple",
        }

    return {
        "query_plan": plan,
        "retrieval_strategy": plan.get("strategy", "hybrid"),
        "has_documents": has_docs,
    }


# NODE 3 — Multi-Strategy Retrieval Agent
def retrieve_multi(state: AgenticState) -> dict:
    """Execute retrieval using the planned strategy, then rerank."""
    plan: QueryPlan = state.get("query_plan", {})
    strategy = state.get("retrieval_strategy", "hybrid")
    chat_id = state["chat_id"]

    # Execute retrieval for each sub-question and merge
    all_results: List[RetrievalResult] = []
    sub_questions = plan.get("sub_questions", [state["input"]])

    for sq in sub_questions[:3]:  # max 3 sub-questions
        results = execute_strategy(strategy, chat_id, sq)
        all_results.extend(results)

    # Rerank if enabled
    if Config.ENABLE_RERANKING and all_results:
        all_results = reranker.rerank(state["input"], all_results)

    # Build context string
    context = _retrieval_to_context(all_results)

    # Extract citations from metadata
    citations: List[Citation] = []
    seen_cites = set()
    for r in all_results:
        meta = r.metadata
        if meta and "source" in meta:
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


# ===================================================================
# NODE 4 — Critic/Validation Agent
# ===================================================================
def critique_context(state: AgenticState) -> dict:
    """Evaluate whether the retrieved context is sufficient to answer."""
    context = state.get("context", "")
    if not context.strip():
        return {
            "critique_score": 0.0,
            "critique_feedback": "No context retrieved.",
        }

    prompt = (
        "You are a Critic Agent. Evaluate whether the retrieved document "
        "context is sufficient to answer the user's question.\n\n"
        f"Question: {state['input']}\n\n"
        f"Retrieved context:\n{context[:2000]}\n\n"
        "Rate from 0.0 (totally irrelevant) to 1.0 (fully sufficient).\n"
        "Respond with ONLY a JSON object: "
        '{"score": 0.85, "feedback": "brief reason", "missing": "what is missing (if anything)"}'
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        critique = json.loads(content)
        score = float(critique.get("score", 0.5))
        feedback = critique.get("feedback", "")
    except Exception as exc:
        logger.warning("Critique fallback: %s", exc)
        score = 0.5
        feedback = "Critique parse failed, proceeding with caution."

    return {
        "critique_score": score,
        "critique_feedback": feedback,
    }


# ===================================================================
# NODE 5 — Reformulation (retry loop)
# ===================================================================
def reformulate_query(state: AgenticState) -> dict:
    """Rewrite the query for a better retrieval attempt."""
    prompt = (
        "The previous retrieval attempt did not find sufficient information. "
        f"Original query: {state['input']}\n"
        f"Critique feedback: {state.get('critique_feedback', '')}\n\n"
        "Rewrite this query to be more specific and effective for search. "
        "Add synonyms, remove ambiguity, use technical terms.\n"
        "Respond with ONLY the rewritten query, nothing else."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        rewritten = resp.content.strip()
    except Exception:
        rewritten = state["input"]

    return {
        "input": rewritten,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def route_critique(state: AgenticState) -> str:
    """After critique: proceed to synthesis, or retry retrieval."""
    score = state.get("critique_score", 0.0)
    attempts = state.get("retrieval_attempts", 0)
    if score < Config.RELEVANCE_THRESHOLD and attempts < Config.MAX_RETRIEVAL_ATTEMPTS:
        return "reformulate"
    return "synthesize"


def route_cache(state: AgenticState) -> str:
    return "end" if state.get("cache_hit", False) else "plan"


# ===================================================================
# NODE 6 — Synthesis Agent
# ===================================================================
def synthesize_answer(state: AgenticState) -> dict:
    """Generate the final grounded answer with inline citations."""
    history_str = _fmt(state["history"])
    context = state.get("context", "")
    citations_text = _citations_to_text(state.get("citations", []))

    # Add code result if code was executed
    code_section = ""
    if state.get("code_result"):
        code_section = f"\nCode execution result:\n{state['code_result']}\n"

    plan: QueryPlan = state.get("query_plan", {})
    complexity = plan.get("complexity", "simple")

    # Instruction variations based on complexity
    if complexity == "complex":
        instruction = (
            "Provide a detailed, structured answer. Break it into sections. "
            "Use bullet points, tables, or numbered steps as appropriate. "
            "Cite sources for every factual claim."
        )
    elif complexity == "medium":
        instruction = (
            "Provide a clear answer with supporting details. "
            "Cite sources for every factual claim."
        )
    else:
        instruction = (
            "Answer concisely and directly. "
            "Cite sources where applicable."
        )

    system_msg = (
        "You are a precise RAG assistant. You must ground every answer in the "
        "retrieved context. When you use information from a source, cite it "
        "inline like [1], [2], etc. At the end, list the sources.\n\n"
        "If the context does not contain enough information to answer, say so "
        "honestly rather than making up an answer."
    )

    prompt = (
        f"{instruction}\n\n"
        f"Document context:\n{context[:3000]}\n"
        f"{code_section}"
        f"\nConversation history:\n{history_str}\n\n"
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


# ===================================================================
# NODE 7 — Store in Semantic Cache
# ===================================================================
def store_cache(state: AgenticState) -> dict:
    if Config.ENABLE_SEMANTIC_CACHE and not state.get("cache_hit"):
        semantic_cache.put(
            state["chat_id"],
            state["input"],
            state.get("answer", ""),
            state.get("citations", []),
        )
    return {}


# ===================================================================
# Graph construction
# ===================================================================
workflow = StateGraph(AgenticState)

workflow.add_node("check_cache", check_cache)
workflow.add_node("plan_query", plan_query)
workflow.add_node("retrieve_multi", retrieve_multi)
workflow.add_node("critique_context", critique_context)
workflow.add_node("reformulate_query", reformulate_query)
workflow.add_node("synthesize_answer", synthesize_answer)
workflow.add_node("store_cache", store_cache)

workflow.set_entry_point("check_cache")

# Cache hit → skip to end
workflow.add_conditional_edges(
    "check_cache",
    route_cache,
    {"plan": "plan_query", "end": "store_cache"},
)

workflow.add_edge("plan_query", "retrieve_multi")
workflow.add_edge("retrieve_multi", "critique_context")

# Critique loop
workflow.add_conditional_edges(
    "critique_context",
    route_critique,
    {
        "reformulate": "reformulate_query",
        "synthesize": "synthesize_answer",
    },
)
workflow.add_edge("reformulate_query", "retrieve_multi")

workflow.add_edge("synthesize_answer", "store_cache")
workflow.add_edge("store_cache", END)

graph = workflow.compile()


# Public entry point (called from agent.py / chat_engine.py)
def run_agentic_rag(
    user_input: str, chat_id: str, history: List[Dict[str, str]]
) -> dict:
    """Execute the full multi-agent RAG pipeline.

    Returns ``{"answer": str, "citations": list[dict]}``.
    """
    initial: AgenticState = {
        "input": user_input,
        "chat_id": chat_id,
        "history": history,
        "query_plan": None,
        "retrieval_results": [],
        "retrieval_strategy": "hybrid",
        "has_documents": False,
        "critique_score": 0.0,
        "critique_feedback": "",
        "retrieval_attempts": 0,
        "context": "",
        "answer": "",
        "citations": [],
        "code_result": "",
        "tool_calls": [],
        "cache_hit": False,
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


# Utility: invalidate cache on document upload
def invalidate_chat_cache(chat_id: str):
    semantic_cache.invalidate(chat_id)
    rebuild_bm25_index(chat_id)
