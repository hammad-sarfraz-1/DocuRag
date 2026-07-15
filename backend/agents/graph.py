import logging
from typing import Dict, List

from langgraph.graph import StateGraph

from backend.agents.state import RagState
from backend.agents.planner import planner_agent, route_after_planner
from backend.agents.retrieval import retrieval_agent
from backend.agents.reranker import reranker_agent, route_after_reranker
from backend.agents.web_search import web_search_agent
from backend.agents.synthesizer import synthesizer_agent
from backend.agents.evaluator import evaluator_agent, route_after_evaluator

logger = logging.getLogger(__name__)

workflow = StateGraph(RagState)

workflow.add_node("planner", planner_agent)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("reranker", reranker_agent)
workflow.add_node("web_search", web_search_agent)
workflow.add_node("synthesizer", synthesizer_agent)
workflow.add_node("evaluator", evaluator_agent)

workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", route_after_planner)
workflow.add_edge("retrieval", "reranker")
workflow.add_conditional_edges("reranker", route_after_reranker)
workflow.add_edge("web_search", "synthesizer")
workflow.add_edge("synthesizer", "evaluator")
workflow.add_conditional_edges("evaluator", route_after_evaluator)

graph = workflow.compile()


def run_agentic_rag(
    user_input: str, chat_id: str, history: List[Dict[str, str]]
) -> dict:
    initial: RagState = {
        "query": user_input,
        "chat_id": chat_id,
        "history": history,
        "has_documents": False,
        "max_rounds": 0,
        "retrieval_round": 0,
        "doc_results": [],
        "ranked_results": [],
        "retrieval_confidence": 0.0,
        "web_results": [],
        "context": "",
        "answer": "",
        "citations": [],
        "eval_verdict": {},
    }

    try:
        final = graph.invoke(initial)
    except Exception:
        logger.exception("Agentic orchestrator failed")
        return {
            "answer": "I ran into an issue generating a response. Please try asking again.",
            "citations": [],
        }

    return {
        "answer": final.get("answer", "I could not generate an answer."),
        "citations": final.get("citations", []),
    }
