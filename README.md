# DocuRag — Document Chat Assistant

An agentic Retrieval-Augmented Generation (RAG) chatbot. Upload documents
(PDF, DOCX, XLSX, TXT) and ask questions about them in natural language.
Answers are grounded in the uploaded files and come with inline source
citations you can click to open the original document.

Built with FastAPI, LangGraph, ChromaDB, PostgreSQL, and the Groq LLM. The
frontend and backend are one Uvicorn process; Postgres runs alongside it as
a second container via Docker Compose.

---

## Features

- **Centralized document store** — every chat reads from and writes to one
  shared ChromaDB collection. Upload once, retrievable from any chat.
  Deleting a chat never touches the shared documents.
- **Document upload** — PDF, DOCX, XLSX, and TXT. Scanned/image PDFs fall
  back to OCR. A standalone `/documents` page uploads without needing a
  chat, and lists the most recently uploaded documents.
- **7-agent LangGraph pipeline** — Planner → Clarifier → Retrieval →
  Reranker → Web Search → Synthesizer → Evaluator (see below).
- **Ask, don't guess** — the Clarifier catches ambiguous follow-ups
  ("what were the issues?" across several documents) and asks which one
  before answering, instead of guessing or answering across all of them.
- **Hybrid retrieval** — vector (semantic) search + BM25 (keyword) search
  fused with Reciprocal Rank Fusion, so exact terms like names and IDs
  aren't lost.
- **Cross-encoder reranking** — retrieved chunks are reordered and given a
  calibrated 0–1 confidence score before reaching the LLM.
- **Self-critique loop** — an Evaluator agent judges the answer's
  faithfulness, grounding, and completeness, and can send the run back to
  Retrieval with a wider search (capped at 2 rounds total).
- **Semantic chunking** — chunk boundaries are cut where meaning actually
  shifts, not at a fixed character count, with a small overlap between
  consecutive chunks so facts spanning a boundary aren't lost.
- **Semantic answer cache** — a reworded repeat question in the same chat
  can skip the whole graph and return the cached answer.
- **Grounded citations** — only the excerpts the model actually references
  inline (`[1]`, `[2]`) are shown as sources; markers are stripped from the
  visible answer text.
- **Optional web search** — set a Tavily key to supplement answers with
  live web results, used only as a fallback when local confidence is low.
- **Auto-titled chats** — the first message's answer and a chat title are
  generated concurrently, so titling adds no perceived latency.
- **Resilience** — timeout + retry with backoff on every external call
  (Groq, Tavily, Chroma), a circuit breaker on Groq, and graceful
  degradation (BM25-only if Chroma is down, no web results if Tavily is
  down) instead of a raw error.
- **Per-chat logs** — `logs/{chat_id}.log` records cache hits/misses and
  the real error behind any failure (e.g. a Groq rate limit), separate
  from the generic message shown to the user.

---

## Architecture

```
Browser (chat UI, documents UI)
        │
        ▼
FastAPI  ─────────────────────────────────────────────────┐
  • routes (chats, upload, chat, history, documents)       │
  • ChatEngine → per-chat history, answer cache             │
        │                                                   │
        ▼                                                   │
LangGraph (7 agents)                                        │
  Planner → Clarifier → Retrieval → Reranker                │
     │          │            │         │                    │
     │          │            │         └─ cross-encoder + calibrated confidence
     │          │            └─ hybrid search (vector + BM25, RRF-fused)
     │          └─ asks a clarifying question instead of guessing
     │
     ├─ low confidence? → Web Search (Tavily, fallback only)
     └─ → Synthesizer (Groq LLM, inline [n] citations)
              → Evaluator (LLM-as-judge; retry Retrieval if unsatisfied, capped at 2 rounds)
        │
        ▼
ChromaDB (shared vector store + BM25) · PostgreSQL (chats + messages)
```

### Agents (`backend/agents/`)

1. **Planner** — rule-based router, no LLM call. Checks whether the shared
   store has documents and sets the retry budget for the run.
2. **Clarifier** — skipped instantly when the question already names a
   document (deterministic filename matching) or there's no prior turn to
   be ambiguous against. Otherwise asks an LLM whether a short follow-up is
   genuinely ambiguous, and if so ends the run with a clarifying question.
3. **Retrieval** — hybrid BM25 + vector search, widened automatically on a
   retry round. Falls back to BM25-only if Chroma is unavailable.
4. **Reranker** — cross-encodes candidates and converts the top score into
   a calibrated 0–1 confidence value.
5. **Web Search** — fires only when confidence drops below
   `WEB_FALLBACK_SCORE_THRESHOLD`, or there are no documents at all.
6. **Synthesizer** — builds a token-budgeted context window and generates
   the answer with inline `[n]` citations.
7. **Evaluator** — an independent LLM call scores faithfulness, grounding,
   and completeness; can send the run back to Retrieval.

---

## Tech Stack

| Layer | Choice |
|---|---|
| API / server | FastAPI + Uvicorn |
| Orchestration | LangGraph (state graph, `backend/agents/`) |
| LLM | Groq (`llama-3.3-70b-versatile` by default) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, one shared collection) |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chat/message persistence | PostgreSQL via SQLAlchemy ORM |
| Document parsing | `pypdf`, `python-docx`, `openpyxl`, EasyOCR + `pdf2image` (OCR fallback) |
| Web search (optional) | Tavily |
| Deployment | Docker Compose (`app` + `postgres`) |
| Frontend | Static HTML served by FastAPI |

---

## Quick Start (local, Docker)

Requires Docker and a [Groq API key](https://console.groq.com).

```bash
# 1. Set your key
echo "GROQ_API_KEY=gsk_..." > .env

# 2. Build + run (app + postgres via Docker Compose)
make build
make up            # serves http://localhost:8000
```

First boot downloads the embedding and reranker models (~20s).

Useful targets: `make status`, `make logs`, `make down`, `make restart`, `make clean`.

## Quick Start (local, without Docker)

Requires Python 3.11 and a running PostgreSQL instance.

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/docurag
uvicorn backend.app:app --host 0.0.0.0 --port 8000
# or, with auto-reload for development:
python run.py
```

Then open **http://localhost:8000**.

---

## Docker Compose

`docker-compose.yml` defines two services:

- `postgres` — Postgres 16, data persisted at `./pgdata`.
- `app` — the FastAPI app, built from `Dockerfile`. Waits for Postgres to
  be healthy before starting; the app creates its own schema (`chats`,
  `messages`) on boot.

```bash
docker build -t docurag .
docker compose up -d
```

Chroma data persists at `./chroma_db`, logs at `./logs`, Postgres data at
`./pgdata` — all bind-mounted so they survive a full rebuild.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `GET` | `/rag` | RAG chatbot page |
| `GET` | `/documents` | Document upload/management page |
| `GET` | `/architecture` | Architecture page |
| `GET` | `/health` | Health check (Groq key, Chroma, Postgres) |
| `POST` | `/chats/new` | Create a chat (form field `name`) |
| `GET` | `/chats` | List chats |
| `DELETE` | `/chats/{chat_id}` | Delete a chat + its history (cascade) |
| `POST` | `/chats/{chat_id}/upload` | Upload files to a chat (multipart `files`) |
| `DELETE` | `/chats/{chat_id}/documents?source=<filename>` | Remove one document |
| `POST` | `/chats/{chat_id}/chat` | Ask a question (form field `question`) → answer + citations |
| `GET` | `/chats/{chat_id}/history` | Get conversation history |
| `GET` | `/documents/{source}` | Download/open an original uploaded file |
| `GET` | `/api/documents/recent?limit=8` | List recently uploaded documents |
| `POST` | `/api/documents/upload` | Upload files with no chat context (multipart `files`) |

---

## Configuration

All settings are environment variables (see `backend/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq LLM access. |
| `DATABASE_URL` | `postgresql+psycopg://docurag:docurag@localhost:5432/docurag` | Postgres connection string. |
| `TAVILY_API_KEY` | — | Optional. Enables web search. |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model. |
| `MAX_CHUNK_CHARS` | `2000` | Fallback split ceiling for the semantic chunker. |
| `RETRIEVAL_K` | `8` | Chunks retrieved per query. |
| `SMALL_CORPUS_CHUNKS` | `25` | At/below this chunk count, feed every chunk to the LLM. |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranking. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `RERANK_KEEP` | `6` | Chunks kept after reranking. |
| `WEB_FALLBACK_SCORE_THRESHOLD` | `0.7` | Confidence below which Web Search fires. |
| `MAX_RETRIEVAL_ROUNDS` | `2` | Evaluator retry budget. |
| `ENABLE_BM25` | `true` | BM25 keyword search. |
| `HYBRID_SEARCH_WEIGHT_VECTOR` / `_BM25` | `0.6` / `0.4` | Hybrid fusion weights. |
| `ENABLE_ANSWER_CACHE` | `true` | Semantic answer cache. |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity cutoff for a cache hit. |
| `USE_OCR` | `true` | OCR fallback for scanned PDFs. |
| `MAX_UPLOAD_SIZE_BYTES` | `25MB` | Per-file upload size cap. |
| `GROQ_TIMEOUT` / `GROQ_MAX_RETRIES` | `60` / `3` | Groq call timeout + retry. |
| `GROQ_CIRCUIT_BREAKER_THRESHOLD` / `_COOLDOWN` | `5` / `30` | Groq circuit breaker. |
| `TAVILY_TIMEOUT` / `TAVILY_MAX_RETRIES` | `10` / `2` | Tavily call timeout + retry. |
| `CHROMA_TIMEOUT` / `CHROMA_MAX_RETRIES` | `5` / `1` | Chroma call timeout + retry. |
| `PERSIST_DIR` | `./chroma_db` | Vector store location. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address. |

---

## Project Structure

```
backend/
  app.py                   FastAPI app and routes
  config.py                Environment-based configuration
  db.py                    SQLAlchemy models (Chat, Message) + schema init
  chat_engine.py           Per-chat history (Postgres) + answer cache
  agent.py                 Entry point into the LangGraph pipeline
  agents/
    graph.py               LangGraph wiring (7 nodes)
    planner.py             Route decision, no LLM call
    clarifier.py           Ambiguous follow-up detection
    retrieval.py           Hybrid search dispatch
    reranker.py            Cross-encoder + calibrated confidence
    web_search.py          Tavily fallback
    synthesizer.py         Answer generation + citations
    evaluator.py           LLM-as-judge, retry decision
    prompts.py             All prompt text, kept out of agent logic
    utils.py               Shared LLM client, per-chat logger
  retrieval_tools.py       Vector / BM25 / hybrid search, reranker, web search
  embedding_store.py       ChromaDB vector store (shared collection)
  document_processor.py    Text extraction, OCR fallback, semantic chunking
  answer_cache.py          Semantic answer cache
  resilience.py            Timeout/retry/circuit-breaker helpers
frontend/                  Static HTML pages (home, rag, documents, architecture)
run.py                     Dev entry point (uvicorn with reload)
docker-compose.yml         app + postgres services
Makefile                   build / up / down / status / logs / clean
Dockerfile                 Production image
requirements.txt           Pinned dependencies (Python 3.11)
```
