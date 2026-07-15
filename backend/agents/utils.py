from typing import Dict, List

from langchain_groq import ChatGroq

from backend.config import Config
from backend.retrieval_tools import vector_store

llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MODEL_NAME, timeout=60, max_retries=2)


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
