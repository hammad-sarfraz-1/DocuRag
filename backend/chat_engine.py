"""Chat session manager — maintains per-chat history and delegates to the agent."""

import logging
from typing import Dict, List

from backend.embedding_store import VectorStore
from backend.agent import supervisor_agent

logger = logging.getLogger(__name__)


class ChatEngine:
    """Owns in-memory conversation history for all active chats."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(chat_id, [])

    def answer(self, chat_id: str, question: str) -> dict:
        """Run the agent on a question and return ``{"answer": …, "citations": […]}``."""
        history = self.get_history(chat_id)

        result = supervisor_agent(
            user_input=question,
            chat_id=chat_id,
            history=history,
        )

        # Persist to session history
        if chat_id not in self.sessions:
            self.sessions[chat_id] = []
        self.sessions[chat_id].append({"role": "user", "content": question})
        self.sessions[chat_id].append(
            {"role": "assistant", "content": result["answer"]}
        )

        return result

    def clear_history(self, chat_id: str):
        if chat_id in self.sessions:
            self.sessions[chat_id] = []

    def delete_chat(self, chat_id: str):
        self.vector_store.delete_chat(chat_id)
        self.sessions.pop(chat_id, None)
