from backend.retrieval_tools import search_web
from backend.agents.state import RagState


def web_search_agent(state: RagState) -> dict:
    """Only reached when the router has already confirmed a Tavily key is
    configured, so no key check is needed here."""
    results = search_web(state["query"], k=2, chat_id=state["chat_id"])
    return {"web_results": [r.__dict__ for r in results]}
