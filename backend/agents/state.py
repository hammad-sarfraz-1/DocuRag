from typing import Dict, List, TypedDict


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

    doc_results: List[dict]
    ranked_results: List[dict]
    retrieval_confidence: float
    web_results: List[dict]

    context: str
    answer: str
    citations: List[Citation]
    eval_verdict: Dict
