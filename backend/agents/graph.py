import logging
from typing import Dict, List

from langgraph.graph import StateGraph

from backend.agents.prompts import AGENTIC_ORCHESTRATOR_FAILURE_MESSAGE
from backend.agents.state import RagState, RouteName
from backend.agents.planner import planner_agent, route_after_planner
from backend.agents.clarifier import clarifier_agent, route_after_clarifier
from backend.agents.retrieval import retrieval_agent
from backend.agents.reranker import reranker_agent, route_after_reranker
from backend.agents.web_search import web_search_agent
from backend.agents.synthesizer import synthesizer_agent
from backend.agents.evaluator import evaluator_agent, route_after_evaluator
from backend.agents.utils import chat_logger

logger = logging.getLogger(__name__)

workflow = StateGraph(RagState)

workflow.add_node(RouteName.PLANNER, planner_agent)
workflow.add_node(RouteName.CLARIFIER, clarifier_agent)
workflow.add_node(RouteName.RETRIEVAL, retrieval_agent)
workflow.add_node(RouteName.RERANKER, reranker_agent)
workflow.add_node(RouteName.WEB_SEARCH, web_search_agent)
workflow.add_node(RouteName.SYNTHESIZER, synthesizer_agent)
workflow.add_node(RouteName.EVALUATOR, evaluator_agent)

workflow.set_entry_point(RouteName.PLANNER)
workflow.add_conditional_edges(RouteName.PLANNER, route_after_planner)
workflow.add_conditional_edges(RouteName.CLARIFIER, route_after_clarifier)
workflow.add_edge(RouteName.RETRIEVAL, RouteName.RERANKER)
workflow.add_conditional_edges(RouteName.RERANKER, route_after_reranker)
workflow.add_edge(RouteName.WEB_SEARCH, RouteName.SYNTHESIZER)
workflow.add_edge(RouteName.SYNTHESIZER, RouteName.EVALUATOR)
workflow.add_conditional_edges(RouteName.EVALUATOR, route_after_evaluator)

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
        "needs_clarification": False,
        "clarifying_question": "",
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
    except Exception as exc:
        logger.exception("Agentic orchestrator failed")
        chat_logger(chat_id).error("Agentic orchestrator failed: %s", exc)
        return {
            "answer": AGENTIC_ORCHESTRATOR_FAILURE_MESSAGE,
            "citations": [],
        }

    return {
        "answer": final.get("answer", "I could not generate an answer."),
        "citations": final.get("citations", []),
        "needs_clarification": final.get("needs_clarification", False),
    }
