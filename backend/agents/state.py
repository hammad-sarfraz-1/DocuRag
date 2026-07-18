from enum import Enum
from typing import Dict, List, TypedDict


class RouteName(str, Enum):
    """Node names in the LangGraph workflow. Used both to register nodes
    (graph.py) and as the return values of route_after_* functions, so a
    typo can't silently create a dangling edge to a node that was never
    registered under that string."""

    PLANNER = "planner"
    CLARIFIER = "clarifier"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    WEB_SEARCH = "web_search"
    SYNTHESIZER = "synthesizer"
    EVALUATOR = "evaluator"


class Citation(TypedDict):
    source: str
    chunk_index: int
    snippet: str


class RagState(TypedDict):
    query: str
    chat_id: str
    history: List[Dict[str, str]]

    has_documents: bool
    max_rounds: int
    retrieval_round: int

    needs_clarification: bool
    clarifying_question: str

    doc_results: List[dict]
    ranked_results: List[dict]
    retrieval_confidence: float
    web_results: List[dict]

    context: str
    answer: str
    citations: List[Citation]
    eval_verdict: Dict
