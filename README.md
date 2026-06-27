# DocuRag — Document Chat Assistant

An agentic Retrieval-Augmented Generation (RAG) chatbot. Upload documents
(PDF, DOCX, TXT) and ask questions about them in natural language. Answers are
grounded in the uploaded files and come with inline source citations.

Built with FastAPI, LangGraph, ChromaDB, and the Groq LLM. The whole app —
API and UI — runs as a single Uvicorn process.

---

## Features

- **Multi-chat** — each chat is isolated, with its own document set and vector collection.
- **Document upload** — PDF, DOCX, and TXT. Scanned/image PDFs fall back to OCR.
- **Hybrid retrieval** — vector (semantic) search + BM25 (keyword) search fused with
  Reciprocal Rank Fusion, so exact terms like names and IDs aren't lost.
- **Cross-encoder reranking** — retrieved chunks are reordered by a reranker before
  they reach the LLM.
- **Small-corpus mode** — when a chat holds few chunks, every chunk is sent to the LLM
  (reranked for order only), so the right document is always in context.
- **Grounded citations** — only the excerpts the model actually references inline
  (`[1]`, `[2]`) are shown as sources.
- **Optional web search** — set a Tavily key to supplement answers with live web results.
- **Conversation memory** — per-chat history is kept for follow-up questions.
- **Natural conversation** — greetings, small talk, and general questions are answered
  normally, without forcing a document lookup.

---

## Architecture

The frontend and backend are one Uvicorn process: FastAPI serves both the JSON API and
the static HTML pages.

```
Browser (upload + chat UI)
        │
        ▼
FastAPI  ──────────────────────────────────────────────┐
  • routes (upload, chat, chat management, history)     │
  • ChatEngine  → per-chat conversation history         │
        │                                               │
        ▼                                               │
LangGraph pipeline                                      │
  plan_query  →  retrieve  →  synthesize                │
     │             │              │                     │
     │             │              └─ Groq LLM → answer + citations
     │             │
     │             ├─ hybrid search (vector + BM25, RRF-fused)
     │             ├─ cross-encoder reranker
     │             └─ optional Tavily web search
     │
     └─ decides: are there docs? is web search needed?
        │
        ▼
ChromaDB (persistent) + SentenceTransformer embeddings
```

### Pipeline steps

1. **plan_query** — checks whether the chat has uploaded documents and whether the
   question looks like it needs fresh web results.
2. **retrieve** — runs hybrid search (or full-context mode for small corpora), reranks
   the results, and optionally adds web results.
3. **synthesize** — Groq generates the answer from the retrieved excerpts; citations are
   built from the inline `[n]` markers the model used.

---

## Tech Stack

| Layer | Choice |
|---|---|
| API / server | FastAPI + Uvicorn |
| Orchestration | LangGraph (state graph) |
| LLM | Groq (`llama-3.3-70b-versatile` by default) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent) |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Document parsing | `pypdf`, `python-docx`, EasyOCR + `pdf2image` (OCR fallback) |
| Web search (optional) | Tavily |
| Frontend | Static HTML served by FastAPI |

---

## Quick Start (local)

Requires Python 3.11 and a [Groq API key](https://console.groq.com).

```bash
# 1. Set your key
echo "GROQ_API_KEY=gsk_..." > .env

# 2. Install + run (Makefile)
make install
make up            # serves http://localhost:8000
```

First boot downloads the embedding and reranker models (~20s).

Useful targets: `make status`, `make logs`, `make down`, `make restart`, `make clean`.

Prefer to run it directly:

```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
# or, with auto-reload for development:
python run.py
```

Then open **http://localhost:8000**.

---

## Docker

```bash
docker build -t docurag .

docker run -p 8000:8000 \
  -e GROQ_API_KEY=gsk_... \
  -v docurag_data:/app/chroma_db \
  docurag
```

The embedding and reranker models are baked into the image, so the container starts
without reaching Hugging Face at runtime. Mount a volume at `/app/chroma_db` to persist
uploaded documents across restarts.

---

## Deploy to Railway

1. Connect the repo; Railway builds from the `Dockerfile`.
2. Add variables: `GROQ_API_KEY` (required), `PORT` set to match the port on your
   public domain (Railway routes the domain to that port; the app binds `$PORT`), and
   `TAVILY_API_KEY` (optional) to enable web search.
3. Add a **Volume** mounted at `/app/chroma_db` to keep documents and chat history
   across deploys.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `GET` | `/rag` | RAG chatbot page |
| `GET` | `/architecture` | Architecture page |
| `GET` | `/health` | Health check |
| `POST` | `/chats/new` | Create a chat (form field `name`) |
| `GET` | `/chats` | List chats |
| `DELETE` | `/chats/{chat_id}` | Delete a chat and its vector data |
| `POST` | `/chats/{chat_id}/upload` | Upload files (multipart `files`) |
| `DELETE` | `/chats/{chat_id}/documents?source=<filename>` | Remove one document |
| `POST` | `/chats/{chat_id}/chat` | Ask a question (form field `question`) → answer + citations |
| `GET` | `/chats/{chat_id}/history` | Get conversation history |

---

## Configuration

All settings are environment variables (see `backend/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq LLM access. |
| `TAVILY_API_KEY` | — | Optional. Enables web search. |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text chunking. |
| `RETRIEVAL_K` | `8` | Chunks retrieved per query. |
| `SMALL_CORPUS_CHUNKS` | `25` | At/below this chunk count, use full-context mode. |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranking. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `RERANK_KEEP` | `6` | Chunks kept after reranking. |
| `ENABLE_BM25` | `true` | BM25 keyword search. |
| `HYBRID_SEARCH_WEIGHT_VECTOR` / `_BM25` | `0.6` / `0.4` | Hybrid fusion weights. |
| `USE_OCR` | `true` | OCR fallback for scanned PDFs. |
| `PERSIST_DIR` | `./chroma_db` | Vector store location. |
| `CHAT_META_FILE` | `./chat_metadata.json` | Chat metadata file. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address. |

---

## Project Structure

```
backend/
  app.py                   FastAPI app and routes
  config.py                Environment-based configuration
  chat_engine.py           Per-chat history; delegates to the agent
  agent.py                 Entry point into the pipeline
  agentic_orchestrator.py  LangGraph pipeline (plan → retrieve → synthesize)
  retrieval_tools.py       Vector / BM25 / hybrid search, reranker, web search
  embedding_store.py       ChromaDB vector store
  document_processor.py    Text extraction, OCR fallback, chunking
frontend/                  Static HTML pages
run.py                     Dev entry point (uvicorn with reload)
Makefile                   install / up / down / status / logs
Dockerfile                 Production image
requirements.txt           Pinned dependencies (Python 3.11)
```
