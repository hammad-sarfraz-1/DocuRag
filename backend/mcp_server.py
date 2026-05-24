"""
MCP (Model Context Protocol) Server
====================================

Exposes document-retrieval tools so that any MCP-aware client (Claude
Desktop, Cursor, etc.) can search the RAG vector store.  The LangGraph
agent inside this application also uses these tools via ``MCPClient``.

Run standalone::

    python -c "from backend.mcp_server import run_mcp_server; run_mcp_server()"

Tools
-----
- ``search_documents``       — Vector similarity search
- ``keyword_search``         — BM25 lexical search
- ``hybrid_search``          — Combined vector + BM25 with RRF fusion
- ``list_chat_sources``      — List uploaded filenames
- ``get_chat_context``       — Summary of documents in a chat
- ``execute_python``         — Sandboxed code execution (for verifying doc snippets)
- ``rerank_query``           — Re-rank results using cross-encoder
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from backend.config import Config
from backend.embedding_store import VectorStore
from backend.retrieval_tools import (
    interpret_code,
    reranker,
    search_bm25,
    search_hybrid,
    search_vector,
)

logger = logging.getLogger(__name__)

vector_store = VectorStore()

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Document RAG Server",
    instructions=(
        "Super RAG tool server. Use 'search_documents' for semantic search, "
        "'keyword_search' for exact term matching, 'hybrid_search' for best "
        "results, 'execute_python' to run code snippets from documentation, "
        "and 'rerank_query' to reorder results by relevance."
    ),
    host=Config.MCP_HOST,
    port=Config.MCP_PORT,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def search_documents(query: str, chat_id: str, k: int = Config.RETRIEVAL_K) -> str:
    """Semantic vector search using sentence embeddings.

    Args:
        query: The search query.
        chat_id: Chat session UUID.
        k: Number of results (default 6).
    """
    try:
        results = vector_store.similarity_search_with_metadata(chat_id, query, k=k)
        if not results:
            return "No relevant documents found."
        parts = []
        for i, (text, meta) in enumerate(results, 1):
            source = meta.get("source", "unknown") if meta else "unknown"
            excerpt = (text[:500] + "...") if len(text) > 500 else text
            parts.append(f"[{i}] ({source})\n{excerpt}")
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def keyword_search(query: str, chat_id: str, k: int = Config.RETRIEVAL_K) -> str:
    """Exact keyword (BM25) search — good for technical terms and code.

    Args:
        query: Search terms.
        chat_id: Chat session UUID.
        k: Number of results (default 6).
    """
    try:
        results = search_bm25(chat_id, query, k=k)
        if not results:
            return "No keyword matches found."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown") if r.metadata else "unknown"
            excerpt = (r.text[:500] + "...") if len(r.text) > 500 else r.text
            parts.append(f"[{i}] ({source}) score={r.score:.3f}\n{excerpt}")
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def hybrid_search(query: str, chat_id: str, k: int = Config.RETRIEVAL_K) -> str:
    """Hybrid search combining vector similarity and BM25 keyword matching.

    Args:
        query: Search query.
        chat_id: Chat session UUID.
        k: Number of results (default 6).
    """
    try:
        results = search_hybrid(chat_id, query, k=k)
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown") if r.metadata else "unknown"
            excerpt = (r.text[:500] + "...") if len(r.text) > 500 else r.text
            parts.append(f"[{i}] ({source}) [{r.source}] score={r.score:.3f}\n{excerpt}")
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def list_chat_sources(chat_id: str) -> str:
    """List every document source uploaded to a chat."""
    try:
        sources = vector_store.get_sources(chat_id)
        if not sources:
            return "No documents uploaded yet."
        return "Document sources:\n" + "\n".join(f"- {s}" for s in sources)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def get_chat_context(chat_id: str) -> str:
    """Summary of document count and sources in a chat."""
    try:
        count = vector_store.get_document_count(chat_id)
        sources = vector_store.get_sources(chat_id)
        parts = [f"Chat has {count} document chunk(s)"]
        if sources:
            parts.append(f"from {len(sources)} source(s): {', '.join(sources)}")
        return " ".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed environment (for code verification).

    Args:
        code: Python code to execute. Imports are blocked for security.
    """
    return interpret_code(code)


@mcp.tool()
def rerank_query(query: str, documents: str, k: Optional[int] = None) -> str:
    """Re-rank a set of documents by relevance to the query.

    Args:
        query: The original query.
        documents: Newline-separated document texts.
        k: Number of top results to return (default: all).
    """
    try:
        from backend.retrieval_tools import RetrievalResult

        texts = [d.strip() for d in documents.split("\n") if d.strip()]
        results = [RetrievalResult(text=t) for t in texts]
        ranked = reranker.rerank(query, results, keep=k or len(results))
        parts = []
        for i, r in enumerate(ranked, 1):
            excerpt = (r.text[:400] + "...") if len(r.text) > 400 else r.text
            parts.append(f"[{i}] score={r.score:.4f}\n{excerpt}")
        return "\n\n".join(parts) if parts else "No documents provided."
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_mcp_server():
    """Start the MCP server with SSE transport (blocking call)."""
    logger.info("Starting MCP server on %s:%s", Config.MCP_HOST, Config.MCP_PORT)
    mcp.run(transport="sse")
