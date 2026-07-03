from backend.config import Config
from backend.agents.state import RagState
from backend.agents.utils import _has_uploaded_docs


def planner_agent(state: RagState) -> dict:
    """Entry point. Decides the initial route and the retry budget for the
    whole run — everything downstream just executes that policy."""
    return {
        "has_documents": _has_uploaded_docs(state["chat_id"]),
        "max_rounds": Config.MAX_RETRIEVAL_ROUNDS,
        "retrieval_round": 0,
    }


def route_after_planner(state: RagState) -> str:
    if state["has_documents"]:
        return "retrieval"
    if Config.TAVILY_API_KEY:
        return "web_search"
    return "synthesizer"
