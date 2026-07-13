"""Chat session manager — maintains per-chat history and delegates to the agent."""

import json
import logging
import os
from typing import Dict, List

from backend.config import Config
from backend.embedding_store import VectorStore
from backend.agent import supervisor_agent
from backend import answer_cache

logger = logging.getLogger(__name__)

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


def _chat_logger(chat_id: str) -> logging.Logger:
    """One log file per chat: logs/{chat_id}.log."""
    return _file_logger(f"chat.{chat_id}", f"{chat_id}.log")


_cache_logger = _file_logger("answer_cache", "answer_cache.log")


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

    def create_chat(self, chat_id: str):
        """Start this chat's own log file. Called when a new chat starts."""
        _chat_logger(chat_id).info("chat created")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(chat_id, [])

    def answer(self, chat_id: str, question: str) -> dict:
        """Run the agent on a question and return ``{"answer": …, "citations": […]}``."""
        history = self.get_history(chat_id)

        cached = answer_cache.get(chat_id, question) if Config.ENABLE_ANSWER_CACHE else None
        if cached is not None:
            result = {
                "answer": cached["answer"],
                "citations": cached["citations"],
                "cached": True,
            }
            _cache_logger.info(
                "HIT  chat=%s question=%r similarity=%.4f matched=%r",
                chat_id, question, cached["similarity"], cached["matched_question"],
            )
            _chat_logger(chat_id).info(
                "HIT  question=%r similarity=%.4f matched=%r",
                question, cached["similarity"], cached["matched_question"],
            )
        else:
            match = answer_cache.best_match(chat_id, question) if Config.ENABLE_ANSWER_CACHE else None
            result = supervisor_agent(
                user_input=question,
                chat_id=chat_id,
                history=history,
            )
            result["cached"] = False
            if match is not None:
                _cache_logger.info(
                    "MISS chat=%s question=%r similarity=%.4f (threshold=%.2f) nearest=%r",
                    chat_id, question, match["similarity"], Config.CACHE_SIMILARITY_THRESHOLD, match["matched_question"],
                )
                _chat_logger(chat_id).info(
                    "MISS question=%r similarity=%.4f (threshold=%.2f) nearest=%r",
                    question, match["similarity"], Config.CACHE_SIMILARITY_THRESHOLD, match["matched_question"],
                )
            else:
                _cache_logger.info("MISS chat=%s question=%r (cache empty)", chat_id, question)
                _chat_logger(chat_id).info("MISS question=%r (cache empty)", question)
            if Config.ENABLE_ANSWER_CACHE:
                answer_cache.put(
                    chat_id, question, result["answer"], result.get("citations", [])
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
