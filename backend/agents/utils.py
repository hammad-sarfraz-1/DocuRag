import logging
import os
from typing import Dict, List

from langchain_groq import ChatGroq

from backend.config import Config
from backend.retrieval_tools import vector_store

llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MODEL_NAME, timeout=60, max_retries=2)

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
