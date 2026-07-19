import json
import logging
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from backend.agents.prompts import CLARIFIER_FALLBACK_QUESTION, CLARIFIER_PROMPT_TEMPLATE
from backend.agents.state import RagState, RouteName
from backend.agents.utils import _fmt, chat_logger, invoke_llm, vector_store

logger = logging.getLogger(__name__)

# Words too generic to count as identifying a specific document on their
# own (e.g. every filename could contain "report" or "data") — skipped when
# building each source's set of distinctive words.
_GENERIC_WORDS = {
    "the", "and", "for", "report", "data", "document", "file", "doc",
    "issues", "issue", "of", "on", "in", "to", "a", "an",
}
_MIN_WORD_LEN = 4


def _distinctive_words(source: str) -> set:
    """Words from a filename specific enough to identify it — drop the
    extension, split on non-alphanumeric, drop short/generic filler words."""
    name = re.sub(r"\.\w+$", "", source)
    words = re.findall(r"[a-zA-Z]+", name.lower())
    return {w for w in words if len(w) >= _MIN_WORD_LEN and w not in _GENERIC_WORDS}


def _find_unambiguous_source(query: str, sources: List[str]) -> Optional[str]:
    """If the question itself names exactly one document distinctly enough
    to identify it (e.g. "the agreement", "the CV", "teacher training"),
    there's nothing to clarify — skip the LLM judgment call entirely rather
    than rely on it to reliably notice this on its own."""
    query_words = re.findall(r"[a-zA-Z]+", query.lower())

    def _mentions(source: str) -> bool:
        distinctive = _distinctive_words(source)
        # A distinctive word must be a PREFIX of a query word (not just any
        # substring) — catches a possessive/plural like "hammads" or
        # "agreements" while not matching e.g. "me" against "agreement".
        return any(
            qw.startswith(word) or word.startswith(qw)
            for word in distinctive
            for qw in query_words
            if len(qw) >= _MIN_WORD_LEN
        )

    matches = [source for source in sources if _mentions(source)]
    if len(matches) == 1:
        return matches[0]
    return None


def _parse_verdict(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in clarifier response")
    verdict = json.loads(match.group(0))
    return {
        "ambiguous": bool(verdict.get("ambiguous", False)),
        "clarifying_question": str(verdict.get("clarifying_question", "")),
    }


def _resolve_pending_clarification(history: List[dict]) -> Optional[str]:
    """If the last assistant turn WAS a clarifying question, the user's
    current message is answering it ("yes the agreement"), not asking a new
    question. The real question is the one BEFORE that clarifying question —
    combine it with the user's reply so retrieval runs against what they
    actually wanted to know, instead of the reply alone (which usually has
    no retrievable content by itself)."""
    if len(history) < 2:
        return None
    last = history[-1]
    if last.get("role") != "assistant" or not last.get("needs_clarification"):
        return None
    original_question = history[-2].get("content", "")
    if not original_question:
        return None
    return original_question


def clarifier_agent(state: RagState) -> dict:
    """Checks whether the question is ambiguous about which uploaded document
    it refers to (e.g. "what were the issues?" after multiple documents were
    mentioned). Only runs the LLM when there's more than one document to be
    ambiguous between, AND there's prior conversation for a follow-up to be
    vague relative to — the first message in a chat has no prior turn that
    could have "listed multiple documents", so there's nothing to
    disambiguate yet regardless of what the LLM might guess."""
    chat_id = state["chat_id"]
    history = state["history"]
    sources = vector_store.get_sources(chat_id)

    # If we just asked "which document did you mean?", this turn is the
    # user's answer, not a new question — resolve deterministically instead
    # of running the ambiguity check again on a reply like "the agreement".
    resolved = _resolve_pending_clarification(history)
    if resolved is not None:
        return {
            "needs_clarification": False,
            "clarifying_question": "",
            "query": f"{resolved} (the document being asked about: {state['query']})",
        }

    if len(sources) < 2 or not history:
        return {"needs_clarification": False, "clarifying_question": ""}

    # If the question itself names one document distinctly (e.g. "does the
    # AGREEMENT say...", "what's on hammad's CV"), it isn't ambiguous no
    # matter how many other documents exist — check this deterministically
    # instead of trusting the LLM to always notice it (it doesn't reliably).
    if _find_unambiguous_source(state["query"], sources) is not None:
        return {"needs_clarification": False, "clarifying_question": ""}

    prompt = CLARIFIER_PROMPT_TEMPLATE.format(
        num_sources=len(sources),
        sources=", ".join(sources),
        history_str=_fmt(state["history"]),
        query=state["query"],
    )

    try:
        resp = invoke_llm([HumanMessage(content=prompt)], chat_id)
        verdict = _parse_verdict(resp.content)
    except Exception as exc:
        logger.warning("Clarifier judging failed, skipping clarification: %s", exc)
        chat_logger(chat_id).error("Clarifier judging failed: %s", exc)
        return {"needs_clarification": False, "clarifying_question": ""}

    if not verdict["ambiguous"]:
        return {"needs_clarification": False, "clarifying_question": ""}

    question = verdict["clarifying_question"] or CLARIFIER_FALLBACK_QUESTION
    return {
        "needs_clarification": True,
        "clarifying_question": question,
        # Set directly so the graph can END here with this as the response —
        # no separate LLM call to "generate" a message that already exists.
        "answer": question,
        "citations": [],
    }


def route_after_clarifier(state: RagState) -> str:
    if state.get("needs_clarification"):
        return END
    return RouteName.RETRIEVAL
