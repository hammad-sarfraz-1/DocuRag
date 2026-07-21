"""Chat session manager — maintains per-chat history and delegates to the agent."""

import logging
from typing import Dict, List

from backend.config import Config
from backend.embedding_store import VectorStore
from backend.agent import supervisor_agent
from backend.agents.utils import _file_logger, chat_logger as _chat_logger
from backend import answer_cache
from backend.db import Message, SessionLocal

logger = logging.getLogger(__name__)

_cache_logger = _file_logger("answer_cache", "answer_cache.log")


class ChatEngine:
    """Owns conversation history for all chats, persisted to Postgres via the
    ORM (messages table, one row per message) so it survives restarts/redeploys."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def create_chat(self, chat_id: str):
        """Start this chat's own log file. Called when a new chat starts."""
        _chat_logger(chat_id).info("chat created")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        with SessionLocal() as session:
            messages = (
                session.query(Message)
                .filter(Message.chat_id == chat_id)
                .order_by(Message.id)
                .all()
            )
            return [
                {
                    "role": m.role,
                    "content": m.content,
                    "citations": m.citations,
                    "needs_clarification": m.needs_clarification,
                }
                for m in messages
            ]

    def answer(self, chat_id: str, question: str) -> dict:
        """Run the agent on a question and return ``{"answer": …, "citations": […]}``."""
        history = self.get_history(chat_id)

        cached = (
            answer_cache.get(chat_id, question, history_len=len(history))
            if Config.ENABLE_ANSWER_CACHE
            else None
        )
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
            # Never cache a clarifying question — it's an answer to "which
            # document did you mean", not to the user's actual question, so
            # reusing it for a differently-worded question later would be wrong.
            if Config.ENABLE_ANSWER_CACHE and not result.get("needs_clarification"):
                answer_cache.put(
                    chat_id, question, result["answer"], result.get("citations", []),
                    history_len=len(history),
                )

        with SessionLocal() as session:
            session.add(Message(chat_id=chat_id, role="user", content=question))
            session.add(
                Message(
                    chat_id=chat_id,
                    role="assistant",
                    content=result["answer"],
                    # Persist citations alongside the message so they survive a chat
                    # reopen (the /history endpoint returns these dicts verbatim).
                    citations=result.get("citations", []),
                    # Tagged when this message IS a clarifying question (Clarifier
                    # agent), so the NEXT turn can tell "the user is now resolving
                    # an ambiguity" apart from an ordinary follow-up, and resume
                    # the original question instead of treating the reply
                    # ("yes, the agreement") as a brand new question to retrieve.
                    needs_clarification=result.get("needs_clarification", False),
                )
            )
            session.commit()

        return result

    def clear_history(self, chat_id: str):
        with SessionLocal() as session:
            session.query(Message).filter(Message.chat_id == chat_id).delete()
            session.commit()

    def delete_chat(self, chat_id: str):
        self.vector_store.delete_chat(chat_id)

        # Close and detach the per-chat FileHandler so its open file
        # descriptor is released — logging.getLogger() never forgets a name,
        # so without this every deleted chat still leaks one FD forever.
        chat_logger = logging.getLogger(f"chat.{chat_id}")
        for handler in chat_logger.handlers[:]:
            handler.close()
            chat_logger.removeHandler(handler)
