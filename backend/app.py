"""
FastAPI Application — RAG Chatbot Backend
=========================================

Routes
------
- ``GET  /``                     — Serve the frontend UI
- ``GET  /rag``                  — RAG chatbot page
- ``GET  /architecture``         — Architecture docs page
- ``POST /chats/new``            — Create a new chat session
- ``GET  /chats``                — List all chats
- ``DELETE /chats/{chat_id}``    — Delete a chat + its vector data
- ``POST /chats/{chat_id}/upload`` — Upload files to a chat
- ``POST /chats/{chat_id}/chat`` — Ask a question (returns answer + citations)
- ``GET  /chats/{chat_id}/history`` — Retrieve chat message history
- ``GET  /documents/{source}``   — Serve an original uploaded file
- ``GET  /health``               — Health check
"""

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Configure the root logger here, before any other module's `getLogger()` calls
# emit anything — app.py is the entrypoint every other module gets imported
# through, so this runs exactly once, first.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)

from backend.config import Config
from backend.document_processor import (
    SUPPORTED_EXTENSIONS,
    build_chunk_metadata,
    chunk_text,
    extract_text,
)
from backend.embedding_store import VectorStore
from backend.chat_engine import ChatEngine
from backend.retrieval_tools import rebuild_bm25_index

logger = logging.getLogger(__name__)

vector_store = VectorStore()
chat_engine = ChatEngine(vector_store)

os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Reduce a user-supplied filename to a safe single path component.

    os.path.basename alone strips directories but still lets through control
    characters, an empty result, or bare "." / "..". Reject those here.
    """
    name = os.path.basename(name)
    name = "".join(c for c in name if ord(c) >= 32 and c != "\x7f")
    name = name[:255]
    if name in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def _document_path(source: str) -> str:
    """Path for a raw uploaded file, keyed by its filename (the shared
    document store is centralized, so one file = one path for everyone)."""
    return os.path.join(Config.DOCUMENTS_DIR, _sanitize_filename(source))


def _load_chat_metadata() -> dict:
    try:
        with open(Config.CHAT_META_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        logger.exception("Failed to parse %s; starting with empty chat registry", Config.CHAT_META_FILE)
        return {}


def _save_chat_metadata(meta: dict):
    tmp_path = Config.CHAT_META_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, Config.CHAT_META_FILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Chatbot",
    version="2.0.0",
    lifespan=lifespan,
)
chat_metadata = _load_chat_metadata()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("frontend/index.html", "r") as f:
        return f.read()


@app.get("/rag", response_class=HTMLResponse)
async def serve_rag():
    with open("frontend/rag.html", "r") as f:
        return f.read()


@app.get("/architecture", response_class=HTMLResponse)
async def serve_architecture():
    with open("frontend/architecture.html", "r") as f:
        return f.read()


@app.get("/health")
async def health():
    failures = []
    if not Config.GROQ_API_KEY:
        failures.append("groq_api_key_missing")
    try:
        vector_store.client.heartbeat()
    except Exception:
        logger.exception("Health check: vector store heartbeat failed")
        failures.append("vector_store_unreachable")
    if failures:
        raise HTTPException(status_code=503, detail={"status": "error", "failed": failures})
    return {"status": "ok"}


@app.post("/chats/new")
async def create_chat(name: str = Form("New Chat")):
    chat_id = str(uuid.uuid4())
    chat_metadata[chat_id] = {"name": name}
    _save_chat_metadata(chat_metadata)
    chat_engine.create_chat(chat_id)
    return {"chat_id": chat_id, "name": name}


@app.get("/chats")
async def list_chats():
    return [
        {"chat_id": cid, "name": data["name"]}
        for cid, data in chat_metadata.items()
    ]


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    chat_engine.delete_chat(chat_id)
    chat_metadata.pop(chat_id, None)
    _save_chat_metadata(chat_metadata)
    return {"status": "deleted"}


def _process_upload(chat_id: str, filename: str, file_bytes: bytes) -> int:
    """Synchronous, potentially slow (OCR, chunking, ChromaDB write) body of a
    single file's upload processing. Run off the event loop via run_in_threadpool."""
    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"No text could be extracted from {filename}.",
        )
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"Could not split {filename} into chunks.",
        )
    metadatas = build_chunk_metadata(chunks, filename)
    # Add each file on its own so per-source replacement and id
    # namespacing in add_documents work correctly (one source per call).
    vector_store.add_documents(chat_id, chunks, metadatas)
    with open(_document_path(filename), "wb") as f:
        f.write(file_bytes)
    return len(chunks)


@app.post("/chats/{chat_id}/upload")
async def upload_files(chat_id: str, files: List[UploadFile] = File(...)):
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")

    total_chunks = 0
    per_file = []

    for file in files:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            )
        try:
            file_bytes = await file.read(Config.MAX_UPLOAD_SIZE_BYTES + 1)
            if len(file_bytes) > Config.MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{file.filename} exceeds the upload size limit "
                    f"({Config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB).",
                )
            num_chunks = await run_in_threadpool(_process_upload, chat_id, file.filename, file_bytes)
            total_chunks += num_chunks
            per_file.append({"filename": file.filename, "chunks": num_chunks})
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error processing upload %s", file.filename)
            raise HTTPException(
                status_code=400,
                detail=f"Error processing {file.filename}. Please try again or "
                "contact support if this persists.",
            )

    # Rebuild the keyword index once so BM25/hybrid search reflects new uploads.
    await run_in_threadpool(rebuild_bm25_index, chat_id)
    return {"status": "success", "chunks": total_chunks, "files": per_file}


@app.delete("/chats/{chat_id}/documents")
async def delete_document(chat_id: str, source: str):
    """Remove a single uploaded document (by filename) from the shared store."""
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    await run_in_threadpool(vector_store.delete_document, chat_id, source)
    await run_in_threadpool(rebuild_bm25_index, chat_id)
    try:
        os.remove(_document_path(source))
    except FileNotFoundError:
        pass
    return {"status": "deleted", "source": source}


@app.get("/documents/{source}")
async def get_document(source: str):
    """Serve the original uploaded file so a citation can open it directly."""
    path = _document_path(source)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(path, filename=source)


@app.post("/chats/{chat_id}/chat")
async def chat(chat_id: str, question: str = Form(...)):
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = await run_in_threadpool(chat_engine.answer, chat_id, question.strip())
    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
    }


@app.get("/chats/{chat_id}/history")
async def get_history(chat_id: str):
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"history": chat_engine.get_history(chat_id)}
