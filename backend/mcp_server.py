"""
MCP (Model Context Protocol) Server
====================================

Exposes document-retrieval tools so that any MCP-aware client (Claude
Desktop, Cursor, etc.) can search the RAG vector store.  The LangGraph
agent inside this application also uses these tools via ``MCPClient``.

Run standalone::

    python -c "from backend.mcp_server import run_mcp_server; run_mcp_server()"

Or start it alongside the FastAPI server from ``run.py``.
"""

import logging

from mcp.server.fastmcp import FastMCP

from backend.config import Config
from backend.embedding_store import VectorStore

logger = logging.getLogger(__name__)

# Singleton vector-store instance shared with the agent
vector_store = VectorStore()

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Document RAG Server",
    instructions=(
        "Search uploaded documents and list available sources. "
        "Use 'search_documents' when the user asks a question about "
        "their uploaded files. Use 'list_chat_sources' to see which "
        "documents are available in a chat session."
    ),
    host=Config.MCP_HOST,
    port=Config.MCP_PORT,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def search_documents(query: str, chat_id: str, k: int = Config.RETRIEVAL_K) -> str:
    """Search the uploaded document chunks for the given *chat_id* and
    return the *k* most relevant excerpts with source attribution.

    Args:
        query: The natural-language search query.
        chat_id: UUID of the chat session to search within.
        k: How many results to return (default 4).

    Returns:
        Formatted text with numbered excerpts and source filenames.
    """
    try:
        results = vector_store.similarity_search_with_metadata(chat_id, query, k=k)
        if not results:
            return "No relevant documents found in this chat."

        parts = []
        for i, (text, meta) in enumerate(results, 1):
            source = meta.get("source", "unknown") if meta else "unknown"
            # Truncate very long excerpts for readability
            excerpt = (text[:600] + "...") if len(text) > 600 else text
            parts.append(f"[{i}] From: {source}\n{excerpt}")
        return "\n\n".join(parts)
    except Exception as exc:
        logger.exception("MCP search_documents error")
        return f"Search error: {exc}"


@mcp.tool()
def list_chat_sources(chat_id: str) -> str:
    """List every document source file that has been uploaded to a chat.

    Args:
        chat_id: UUID of the chat session.

    Returns:
        Bulleted list of source filenames or a "no sources" message.
    """
    try:
        sources = vector_store.get_sources(chat_id)
        if not sources:
            return "No documents have been uploaded to this chat yet."
        return "Document sources:\n" + "\n".join(f"- {s}" for s in sources)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def get_chat_context(chat_id: str) -> str:
    """Return a brief summary of what documents exist in a chat session.

    Args:
        chat_id: UUID of the chat session.

    Returns:
        Summary line with chunk count and source filenames.
    """
    try:
        count = vector_store.get_document_count(chat_id)
        sources = vector_store.get_sources(chat_id)
        parts = [f"Chat {chat_id[:8]}… has {count} document chunk(s)"]
        if sources:
            parts.append(f"from {len(sources)} source(s): {', '.join(sources)}")
        return " ".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_mcp_server():
    """Start the MCP server with SSE transport (blocking call)."""
    logger.info("Starting MCP server on %s:%s", Config.MCP_HOST, Config.MCP_PORT)
    mcp.run(transport="sse")
