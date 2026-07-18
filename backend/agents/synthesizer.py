import logging
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import Config
from backend.agents.prompts import (
    SYNTHESIS_FAILURE_MESSAGE,
    SYNTHESIZER_GUIDANCE_DOCS_AND_WEB,
    SYNTHESIZER_GUIDANCE_DOCS_ONLY,
    SYNTHESIZER_GUIDANCE_NO_CONTEXT,
    SYNTHESIZER_GUIDANCE_WEB_ONLY,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_PROMPT_TEMPLATE,
)
from backend.agents.state import Citation, RagState
from backend.agents.utils import _fmt, llm

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 5  # rough estimate; no exact tokenizer for the Groq-hosted model


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _retrieval_to_context(results: List[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata") or {}
        if r.get("source") == "web":
            label = meta.get("title") or meta.get("url") or "web result"
        else:
            label = meta.get("source", "document")
            ci = meta.get("chunk_index")
            if ci is not None:
                label = f"{label}, part {ci + 1}"
        parts.append(f"[{i}] ({label}) {r.get('text', '')[:800]}")
    return "\n\n".join(parts)


def _build_citations(results: List[dict]) -> List[Citation]:
    """Build display citations from retrieval-result dicts (keys: text, metadata, source)."""
    citations: List[Citation] = []
    seen_cites = set()
    for r in results:
        meta = r.get("metadata") or {}
        text = r.get("text", "")
        if r.get("source") == "web":
            url = meta.get("url", "")
            title = meta.get("title", "")
            key = ("web", url)
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append(Citation(
                    source=f"Web: {title}" if title else "Web search",
                    chunk_index=0,
                    snippet=text[:200],
                ))
        elif "source" in meta:
            key = (meta["source"], meta.get("chunk_index", 0))
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append(Citation(
                    source=meta["source"],
                    chunk_index=meta.get("chunk_index", 0),
                    snippet=text[:200],
                ))
    return citations


def _citations_for_answer(answer: str, retrieval_results: List[dict]) -> List[Citation]:
    """Cite ONLY the excerpts the LLM actually referenced inline (e.g. [2], [4, 5]),
    mapped back by position to the numbered context. This keeps the Sources panel
    truthful — a single-document answer shows a single document."""
    nums = set()
    for grp in re.findall(r"\[([\d,\s]+)\]", answer or ""):
        for part in grp.split(","):
            part = part.strip()
            if part.isdigit():
                nums.add(int(part))
    chosen = [
        retrieval_results[n - 1]
        for n in sorted(nums)
        if 1 <= n <= len(retrieval_results)
    ]
    return _build_citations(chosen)


def synthesizer_agent(state: RagState) -> dict:
    history_str = _fmt(state["history"])
    context_results = state.get("ranked_results", []) + state.get("web_results", [])
    has_docs = state.get("has_documents", False)
    has_web = any(r.get("source") == "web" for r in context_results)

    system_msg = SYNTHESIZER_SYSTEM_PROMPT

    if has_docs and has_web:
        guidance = SYNTHESIZER_GUIDANCE_DOCS_AND_WEB
    elif has_docs:
        guidance = SYNTHESIZER_GUIDANCE_DOCS_ONLY
    elif has_web:
        guidance = SYNTHESIZER_GUIDANCE_WEB_ONLY
    else:
        guidance = SYNTHESIZER_GUIDANCE_NO_CONTEXT

    context = _retrieval_to_context(context_results)

    reserved_tokens = (
        _estimate_tokens(system_msg)
        + _estimate_tokens(history_str)
        + Config.RESPONSE_TOKEN_RESERVE
    )
    context_char_budget = max(Config.MAX_CONTEXT_TOKENS - reserved_tokens, 0) * _CHARS_PER_TOKEN
    context_block = (
        context[:context_char_budget]
        if context.strip() and context_char_budget > 0
        else "(no excerpts retrieved)"
    )

    prompt = SYNTHESIZER_USER_PROMPT_TEMPLATE.format(
        guidance=guidance,
        context_block=context_block,
        history_str=history_str,
        query=state["query"],
    )

    try:
        resp = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
        answer = resp.content.strip()
    except Exception:
        logger.exception("Synthesis failed")
        answer = SYNTHESIS_FAILURE_MESSAGE

    citations = _citations_for_answer(answer, context_results)

    return {"context": context, "answer": answer, "citations": citations}
