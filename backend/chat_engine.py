"""Chat session manager — maintains per-chat history and delegates to the agent."""

import json
import logging
import os
from typing import Dict, List

from backend.config import Config
from backend.embedding_store import VectorStore
from backend.agent import supervisor_agent

logger = logging.getLogger(__name__)


class ChatEngine:
    """Owns conversation history for all chats, persisted to disk so it
    survives restarts/redeploys."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.sessions: Dict[str, List[Dict[str, str]]] = self._load_sessions()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_sessions(self) -> Dict[str, List[Dict[str, str]]]:
        try:
            with open(Config.HISTORY_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_sessions(self):
        try:
            os.makedirs(os.path.dirname(Config.HISTORY_FILE) or ".", exist_ok=True)
            with open(Config.HISTORY_FILE, "w") as f:
                json.dump(self.sessions, f)
        except Exception:
            logger.exception("Failed to persist chat history")

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
            {
                "role": "assistant",
                "content": result["answer"],
                # Persist citations alongside the message so they survive a chat
                # reopen (the /history endpoint returns these dicts verbatim).
                "citations": result.get("citations", []),
            }
        )
        self._save_sessions()

        return result

    def clear_history(self, chat_id: str):
        if chat_id in self.sessions:
            self.sessions[chat_id] = []
            self._save_sessions()

    def delete_chat(self, chat_id: str):
        self.vector_store.delete_chat(chat_id)
        self.sessions.pop(chat_id, None)
        self._save_sessions()
