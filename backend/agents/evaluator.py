import json
import logging
import re

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from backend.config import Config
from backend.agents.prompts import EVALUATOR_PROMPT_TEMPLATE
from backend.agents.state import RagState, RouteName
from backend.agents.utils import chat_logger, invoke_llm

logger = logging.getLogger(__name__)

_DEFAULT_VERDICT = {
    "faithful": True,
    "grounded": True,
    "complete": True,
    "retry": False,
    "reason": "",  
}


def _parse_verdict(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in evaluator response")
    verdict = json.loads(match.group(0))
    return {
        "faithful": bool(verdict.get("faithful", True)),
        "grounded": bool(verdict.get("grounded", True)),
        "complete": bool(verdict.get("complete", True)),
        "retry": bool(verdict.get("retry", False)),
        "reason": str(verdict.get("reason", "")),
    }


def evaluator_agent(state: RagState) -> dict:
    """Judges the synthesized answer against the retrieved context and decides
    whether another retrieval round could plausibly improve it. Skips the LLM
    call when there was no context to check (greetings/small talk) — nothing
    to ground, nothing to retry."""
    round_num = state.get("retrieval_round", 0)
    context = state.get("context", "")

    if not context.strip():
        return {
            "eval_verdict": {**_DEFAULT_VERDICT, "reason": "no retrieved context to evaluate"},
            "retrieval_round": round_num + 1,
        }

    eval_prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        query=state["query"],
        context=context[:4000],
        answer=state.get("answer", ""),
    )

    try:
        resp = invoke_llm([HumanMessage(content=eval_prompt)], state["chat_id"])
        verdict = _parse_verdict(resp.content)
    except Exception as exc:
        logger.warning("Evaluator judging failed, accepting answer as-is: %s", exc)
        chat_logger(state["chat_id"]).error("Evaluator judging failed: %s", exc)
        verdict = {**_DEFAULT_VERDICT, "reason": "evaluation failed, accepting answer"}

    return {"eval_verdict": verdict, "retrieval_round": round_num + 1}


def route_after_evaluator(state: RagState) -> str:
    verdict = state.get("eval_verdict", {})
    max_rounds = state.get("max_rounds", Config.MAX_RETRIEVAL_ROUNDS)
    if (
        verdict.get("retry")
        and state.get("has_documents", False)
        and state.get("retrieval_round", 0) < max_rounds
    ):
        return RouteName.RETRIEVAL
    return END
