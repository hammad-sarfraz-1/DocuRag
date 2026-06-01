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
- ``GET  /health``               — Health check
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Config
from backend.document_processor import (
    SUPPORTED_EXTENSIONS,
    build_chunk_metadata,
    chunk_text,
    extract_text,
)
from backend.embedding_store import VectorStore
from backend.chat_engine import ChatEngine

logger = logging.getLogger(__name__)

vector_store = VectorStore()
chat_engine = ChatEngine(vector_store)


def _load_chat_metadata() -> dict:
    try:
        with open(Config.CHAT_META_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_chat_metadata(meta: dict):
    with open(Config.CHAT_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


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
    return {"status": "ok"}


@app.post("/chats/new")
async def create_chat(name: str = Form("New Chat")):
    chat_id = str(uuid.uuid4())
    chat_metadata[chat_id] = {"name": name}
    _save_chat_metadata(chat_metadata)
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


@app.post("/chats/{chat_id}/upload")
async def upload_files(chat_id: str, files: List[UploadFile] = File(...)):
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")

    all_chunks = []
    all_metadatas = []

    for file in files:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            )
        try:
            file_bytes = await file.read()
            text = extract_text(file_bytes, file.filename)
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"No text could be extracted from {file.filename}.",
                )
            chunks = chunk_text(text)
            metadatas = build_chunk_metadata(chunks, file.filename)
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing {file.filename}: {e}",
            )

    vector_store.add_documents(chat_id, all_chunks, all_metadatas)
    return {"status": "success", "chunks": len(all_chunks)}


@app.post("/chats/{chat_id}/chat")
async def chat(chat_id: str, question: str = Form(...)):
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = chat_engine.answer(chat_id, question.strip())
    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
    }


@app.get("/chats/{chat_id}/history")
async def get_history(chat_id: str):
    if chat_id not in chat_metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"history": chat_engine.get_history(chat_id)}
