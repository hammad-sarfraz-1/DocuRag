"""
Agent Entry Point
=================

Thin wrapper that delegates to the multi-agent orchestrator
(``backend.agentic_orchestrator``) for backward compatibility.
"""

import logging
from typing import Dict, List

from backend.agentic_orchestrator import run_agentic_rag

logger = logging.getLogger(__name__)


def supervisor_agent(user_input: str, chat_id: str, history: List[Dict[str, str]]) -> dict:
    """Run the multi-agent Super RAG pipeline.

    Returns ``{"answer": str, "citations": list[dict]}``.
    """
    return run_agentic_rag(user_input, chat_id, history)
