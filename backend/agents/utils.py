import logging
import os
from typing import Dict, List

from langchain_groq import ChatGroq

from backend.config import Config
from backend.resilience import CircuitBreaker, ServiceUnavailable, call_with_retry
from backend.retrieval_tools import vector_store

# max_retries=0: our own call_with_retry() already retries with backoff:
# letting ChatGroq retry internally too would multiply attempts silently.
llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MODEL_NAME, timeout=Config.GROQ_TIMEOUT, max_retries=0)

groq_circuit_breaker = CircuitBreaker(
    threshold=Config.GROQ_CIRCUIT_BREAKER_THRESHOLD,
    cooldown=Config.GROQ_CIRCUIT_BREAKER_COOLDOWN,
)


def invoke_llm(messages, chat_id: str):
    """Call the Groq LLM with timeout + retry + backoff, guarded by a circuit
    breaker that trips after repeated consecutive failures (across all
    chats — Groq being down isn't a per-chat condition). Raises
    ServiceUnavailable on failure; callers already catch and degrade
    gracefully (skip to next agent, log per-chat)."""
    if groq_circuit_breaker.is_open():
        raise ServiceUnavailable("Groq circuit breaker open, skipping call")

    try:
        result = call_with_retry(
            lambda: llm.invoke(messages),
            service="groq",
            chat_id=chat_id,
            timeout=Config.GROQ_TIMEOUT,
            attempts=Config.GROQ_MAX_RETRIES,
        )
    except ServiceUnavailable:
        groq_circuit_breaker.record_failure()
        raise
    groq_circuit_breaker.record_success()
    return result

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


def _file_logger(name: str, filename: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, filename))
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
    return log


def chat_logger(chat_id: str) -> logging.Logger:
    """One log file per chat: logs/{chat_id}.log — shared by chat_engine and
    the agents so real LLM-call errors land in the same file as cache hits."""
    return _file_logger(f"chat.{chat_id}", f"{chat_id}.log")


def _fmt(history: List[Dict], n: int = 6) -> str:
    lines = []
    for m in history[-n:]:
        lines.append(f"{m['role'].capitalize()}: {m['content']}")
    return "\n".join(lines)


def _has_uploaded_docs(chat_id: str) -> bool:
    try:
        return vector_store.get_document_count(chat_id) > 0
    except Exception:
        return False
