# RAG Chatbot — Multi-Chat Document Assistant

An **agentic Retrieval-Augmented Generation (RAG)** chatbot that lets users upload documents (PDF, DOCX, TXT) and ask questions in natural language. Built with FastAPI, LangGraph, ChromaDB, and the Groq LLM — with a dedicated **MCP (Model Context Protocol) server** for external tool access.

---

## Table of Contents

- [Architecture](#architecture)
- [Why These Technologies?](#why-these-technologies)
- [RAG Pipeline (Step by Step)](#rag-pipeline-step-by-step)
- [Agentic Enhancements](#agentic-enhancements)
- [MCP Server](#mcp-server)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client (Browser)                             │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐  │
│   │ File Upload   │   │ Chat Input   │   │ Response + Citations  │  │
│   └──────┬───────┘   └──────┬───────┘   └──────────┬────────────┘  │
└──────────┼──────────────────┼───────────────────────┼───────────────┘
           │                  │                       │
      ┌────▼──────────────────▼───────────────────────▼────────────┐
      │                    FastAPI Server (:8000)                    │
      │  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐ │
      │  │ app.py   │─▶│ ChatEngine   │─▶│  Agent Orchestrator  │ │
      │  │ (routes) │  │ (sessions)   │  │  (LangGraph)         │ │
      │  └──────────┘  └──────┬───────┘  └──────────────────────┘ │
      │                       │                                    │
      │  ┌─────────────────────────────────────────────────────┐  │
      │  │              Multi-Agent System                      │  │
      │  │  ┌────────────┐ ┌────────────┐ ┌───────────────┐   │  │
      │  │  │  Query     │ │  Retrieval │ │  Critic       │   │  │
      │  │  │  Planner   │ │  Agent     │ │  Agent        │   │  │
      │  │  └────────────┘ └─────┬──────┘ └───────┬───────┘   │  │
      │  │                       │                 │           │  │
      │  │  ┌────────────────────▼─────────────────▼────────┐  │  │
      │  │  │         Multi-Strategy Retrieval              │  │  │
      │  │  │  ┌──────┐ ┌──────┐ ┌───────┐ ┌──────┐       │  │  │
      │  │  │  │Vector│ │ BM25 │ │Hybrid │ │ Web  │       │  │  │
      │  │  │  │Search│ │Search│ │Search │ │Search│       │  │  │
      │  │  │  └──────┘ └──────┘ └───────┘ └──────┘       │  │  │
      │  │  └──────────────────────────┬───────────────────┘  │  │
      │  │                             │                       │  │
      │  │  ┌──────────────────────────▼───────────────────┐  │  │
      │  │  │        Cross-Encoder Reranker                │  │  │
      │  │  └──────────────────────────┬───────────────────┘  │  │
      │  │                             │                       │  │
      │  │  ┌──────────────────────────▼───────────────────┐  │  │
      │  │  │        Synthesis Agent (final answer)        │  │  │
      │  │  └──────────────────────────────────────────────┘  │  │
      │  │                             │                       │  │
      │  │  ┌──────────────────────────▼───────────────────┐  │  │
      │  │  │        Semantic Cache                        │  │  │
      │  │  └──────────────────────────────────────────────┘  │  │
      │  └─────────────────────────────────────────────────────┘  │
      │                                                           │
      │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
      │  │ doc_processor│  │ VectorStore  │  │ Code           │  │
      │  │ (PDF/DOCX/   │  │ (ChromaDB)   │  │ Interpreter    │  │
      │  │  TXT + OCR)  │  │ + BM25 Index │  │ (sandbox)      │  │
      │  └──────────────┘  └──────────────┘  └────────────────┘  │
      └──────────────────────────┬────────────────────────────────┘
                                 │
      ┌──────────────────────────┼────────────────────────────┐
      │        MCP Server (:8100)│                            │
      │  ┌──────────────────────────────┐                     │
      │  │ Tools:  • search_documents   │◄────────────────────┘
      │  │         • keyword_search     │    Any MCP client
      │  │         • hybrid_search      │    (Claude Desktop,
      │  │         • execute_python     │     Cursor, etc.)
      │  │         • rerank_query       │
      │  │         • list_chat_sources  │
      │  └──────────────────────────────┘
      └─────────────────────────────────────────────────────┘
```

### Data Flow

1. **Upload**: Files → `document_processor` (extract text + OCR) → `chunk_text` → `VectorStore.add_documents` (embed + store in ChromaDB) → rebuild BM25 index → invalidate semantic cache
2. **Question**: User query → `ChatEngine` → multi-agent orchestrator (LangGraph) → [semantic cache lookup] → [query planner → retrieval agent → reranker → critic loop → synthesis agent] → response with citations
3. **MCP**: External MCP clients can access the same vector store via the MCP server on port 8100

---

## Why These Technologies?

| Layer | Choice | Rationale |
|---|---|---|
| **Framework** | FastAPI | Async-native, automatic OpenAPI docs, high throughput, easy AWS deployment via Uvicorn |
| **Agent Orchestration** | LangGraph | State-graph architecture with explicit node-by-node control — ideal for multi-agent planning → retrieval → critique → synthesis loops |
| **Vector Store** | ChromaDB | Persistent, embeddable, zero external dependencies. SentenceTransformer integration out of the box. Per-chat collections for clean isolation |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Lightweight (80 MB), fast CPU inference, good semantic quality. Same model powers semantic cache |
| **Lexical Search** | BM25 (rank-bm25) | Complements dense embeddings with exact keyword matching. Critical for technical terms, code, and proper nouns |
| **Reranker** | Cross-encoder (ms-marco-MiniLM-L-6-v2) | Re-ranks retrieved chunks by true semantic relevance. Significantly improves precision over pure embedding similarity |
| **LLM** | Groq (Llama 3.3 70B) | 2000+ tok/s inference speed, generous free tier. Drop-in replaceable via config with any OpenAI-compatible endpoint |
| **Document Parsing** | pypdf + python-docx + EasyOCR | Scanned PDFs handled via OCR fallback. Table extraction from text chunks |
| **Web Augmentation** | Tavily | Purpose-built search API for LLMs; returns clean, relevant content directly |
| **Semantic Cache** | SentenceTransformer + cosine similarity | Avoids redundant LLM calls for repeated or very similar queries. Configurable threshold (default 0.88) |
| **Code Interpreter** | Subprocess sandbox | Verifies code snippets found in documentation by executing them in an isolated Python process |
| **MCP** | `mcp` SDK (Anthropic) | Industry-standard protocol for LLM tool access. Exposes 7 tools to any MCP-aware client |

---

## Multi-Agent Super RAG Pipeline

The core intelligence lives in `backend/agentic_orchestrator.py` as a LangGraph state machine with 7 nodes, orchestrated by 4 specialized agents:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│  AGENT 1: SEMANTIC CACHE LOOKUP                                    │
│                                                                    │
│  check_cache() — query → embed → cosine-sim against stored entries │
│                                                                    │
│  [HIT: sim ≥ 0.88] → return cached answer immediately              │
│  [MISS]            → proceed to planning                           │
└───────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│  AGENT 2: QUERY PLANNER                                           │
│                                                                    │
│  plan_query() — LLM analyzes the question and outputs:             │
│  • sub_questions: ["What is X?", "How does Y work?"]              │
│  • strategy: "hybrid" | "hybrid+web" | "web" | "vector"          │
│  • needs_code: true/false                                         │
│  • complexity: "simple" | "medium" | "complex"                    │
│                                                                    │
│  For follow-ups, it rewrites the query to be self-contained       │
│  using conversation history                                        │
└───────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│  AGENT 3: RETRIEVAL AGENT (Multi-Strategy)                        │
│                                                                    │
│  retrieve_multi() — executes the planned strategy:                 │
│                                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  VECTOR  │  │   BM25   │  │  HYBRID  │  │   WEB    │         │
│  │  SEARCH  │  │  SEARCH  │  │ (RRF     │  │  (Tavily)│         │
│  │(ChromaDB)│  │(keyword) │  │  fusion) │  │          │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                                                    │
│  Merges results from all sub-questions                             │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  CROSS-ENCODER RERANKER                                       │ │
│  │  Re-scores all retrieved chunks by query-document relevance   │ │
│  │  Keeps top-K (default 4)                                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  Extracts source citations from chunk metadata                    │
└───────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│  AGENT 4: CRITIC / VALIDATION AGENT                               │
│                                                                    │
│  critique_context() — LLM evaluates retrieved context:            │
│  • Score 0.0 – 1.0 for sufficiency                                │
│  • Feedback on what's missing                                     │
│                                                                    │
│         ┌──────────────────────────────┐                          │
│         │  score ≥ threshold (0.4)?    │                          │
│         └──────────┬───────────┬───────┘                          │
│                    │           │                                  │
│                 [YES]        [NO]                                 │
│                    │           │                                  │
│                    ▼           ▼                                  │
│              SYNTHESIZE   REFORMULATE & RETRY                     │
│                           (max 3 attempts)                        │
│                                                                    │
│  If max retries exceeded → proceed with best-effort context       │
└───────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│  AGENT 5: SYNTHESIS AGENT                                         │
│                                                                    │
│  synthesize_answer():                                              │
│  • Adapts response format to query complexity:                    │
│    - Simple: concise answer with inline citations [1], [2]       │
│    - Medium: structured paragraphs with supporting details        │
│    - Complex: multi-section with bullet points / tables           │
│  • If code execution was needed, includes code results            │
│  • If context is insufficient, says so honestly                   │
│                                                                    │
│  → Also runs groundedness verification as a final quality check   │
└───────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│  SEMANTIC CACHE STORE                                             │
│  Stores (query_embedding → answer + citations) for future hits    │
└───────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
                           FINAL RESPONSE
                    {"answer": "...", "citations": [...]}
```

### Source Citations

Every retrieved chunk carries metadata:

```python
# backend/document_processor.py:build_chunk_metadata()
{
    "source": "report.pdf",
    "chunk_index": 3,
    "total_chunks": 15,
}
```

The `synthesize_answer` node instructs the LLM to cite inline (`[1]`, `[2]`). The API returns a structured `citations` array, rendered by the frontend as a styled source panel.

---

## Multi-Strategy Retrieval (`backend/retrieval_tools.py`)

Four retrieval strategies, selectable per query:

| Strategy | Method | Best For |
|---|---|---|
| **Vector** | ChromaDB sentence embeddings (dense) | Semantic similarity, paraphrased queries |
| **BM25** | Keyword-based lexical matching (sparse) | Technical terms, code, proper names |
| **Hybrid** | Reciprocal Rank Fusion of vector + BM25 | General purpose (default) |
| **Web** | Tavily API | Current events, external context |

**Reranking**: After retrieval, a cross-encoder scores each (query, chunk) pair for true relevance. Only the top-K chunks reach the LLM.

**Routing**: The query planner selects the optimal strategy based on query content. Queries mentioning code or current events automatically get `hybrid+web`.

---

## Agentic Enhancements

Beyond basic RAG, this implementation adds:

- **Multi-Agent Orchestration** — 5 specialized agents (cache, planner, retrieval, critic, synthesis) coordinated by a LangGraph state machine
- **Autonomous Critique Loop** — If retrieved context scores below threshold, the query is reformulated and re-retrieved (up to 3 attempts)
- **Query Decomposition** — Complex questions are broken into sub-questions, each searched independently, then merged
- **Semantic Caching** — Embeds every query and compares against stored entries (cosine sim ≥ 0.88). Avoinds redundant LLM calls entirely
- **Cross-Encoder Reranking** — Precision boost from `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Code Interpreter** — Sandboxed Python execution for verifying code snippets in documentation
- **Groundedness Verification** — The final answer is checked against sources; unsubstantiated claims trigger re-generation
- **Graceful Degradation** — Every node has try/except fallbacks. If the LLM or any tool fails, the pipeline continues with sensible defaults
- **BM25 + Hybrid Search** — Keyword search catches technical terms that dense embeddings might miss
- **Configurable Everything** — 30+ settings in `config.py`, all overridable via environment variables (see table below)

---

## MCP Server

The [Model Context Protocol](https://modelcontextprotocol.io) is an open standard from Anthropic that lets LLM applications discover and call tools.

**`backend/mcp_server.py`** creates a FastMCP server exposing three tools:

| Tool | Description |
|---|---|
| `search_documents` | Search uploaded document chunks by query + chat_id |
| `list_chat_sources` | List all filenames uploaded to a chat |
| `get_chat_context` | Summary of document count + sources for a chat |

### Using with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-chatbot": {
      "url": "http://localhost:8100"
    }
  }
}
```

The MCP server starts automatically as a background thread inside the FastAPI lifespan. If it fails (e.g. port conflict), the main application continues unaffected.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) (free tier)
- [Tavily API key](https://tavily.com) (free tier — optional, web search will be skipped if missing)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
echo "GROQ_API_KEY=gsk_your_key_here" > .env
echo "TAVILY_API_KEY=tvly-your-key-here" >> .env

# Run
python run.py
```

Open **http://localhost:8000** in your browser.

---

## Docker Deployment

```bash
# Build the image
docker build -t rag-chatbot .

# Run with environment variables
docker run -p 8000:8000 -p 8100:8100 \
  -e GROQ_API_KEY=gsk_your_key \
  -e TAVILY_API_KEY=tvly_your_key \
  -v chroma_data:/app/chroma_db \
  rag-chatbot
```

### Environment Variables

All configuration is driven by environment variables (see `backend/config.py`):

| Variable | Default | Description |
|---|---|---|
| **Core** | | |
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `TAVILY_API_KEY` | — | Optional. Web search key |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model ID |
| `HOST` | `0.0.0.0` | FastAPI bind address |
| `PORT` | `8000` | FastAPI port |
| **Document Processing** | | |
| `CHUNK_SIZE` | `500` | Document chunk characters |
| `CHUNK_OVERLAP` | `50` | Chunk overlap characters |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `RETRIEVAL_K` | `6` | Initial chunks to retrieve (before reranking) |
| `USE_OCR` | `true` | Enable OCR for scanned PDFs |
| **Multi-Agent / Super RAG** | | |
| `ENABLE_SEMANTIC_CACHE` | `true` | Enable semantic caching of Q&A pairs |
| `SEMANTIC_CACHE_SIMILARITY` | `0.88` | Cosine sim threshold for cache hit |
| `SEMANTIC_CACHE_SIZE` | `200` | Max entries per chat |
| `ENABLE_RERANKING` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name |
| `RERANK_KEEP` | `4` | Chunks kept after reranking |
| `ENABLE_BM25` | `true` | Enable BM25 keyword search |
| `HYBRID_SEARCH_WEIGHT_VECTOR` | `0.6` | Vector weight in hybrid RRF fusion |
| `HYBRID_SEARCH_WEIGHT_BM25` | `0.4` | BM25 weight in hybrid RRF fusion |
| `MAX_RETRIEVAL_ATTEMPTS` | `3` | Max critique → reformulate loops |
| `RELEVANCE_THRESHOLD` | `0.4` | Min critique score to skip reformulation |
| `ENABLE_CODE_INTERPRETER` | `true` | Enable sandboxed code execution |
| `CODE_TIMEOUT_SECONDS` | `15` | Code execution timeout |
| **MCP Server** | | |
| `MCP_HOST` | `127.0.0.1` | MCP server bind address |
| `MCP_PORT` | `8100` | MCP server port |

---

## AWS Deployment

### Option 1: ECS (Fargate)

1. Push the Docker image to Amazon ECR:
   ```bash
   aws ecr create-repository --repository-name rag-chatbot
   docker tag rag-chatbot:latest <account>.dkr.ecr.<region>.amazonaws.com/rag-chatbot:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/rag-chatbot:latest
   ```

2. Create an ECS cluster + task definition using the image.
3. Set environment variables (`GROQ_API_KEY`, etc.) in the task definition.
4. Add an EFS volume mounted at `/app/chroma_db` for persistent vector storage.
5. Configure ALB with target group on port 8000.
6. Open the ALB DNS — the frontend is served automatically.

### Option 2: EC2

```bash
# SSH into your EC2 instance
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Pull and run
docker pull <your-ecr-image>
docker run -d -p 80:8000 -e GROQ_API_KEY=... <image>
```

**Important**: Attach an EFS volume to persist the ChromaDB data across restarts.

---

## API Reference

### `POST /chats/new`
Create a new chat session.

| Field | Type | Description |
|---|---|---|
| `name` | form | Chat display name |

**Response:** `{"chat_id": "uuid", "name": "..."}`

### `GET /chats`
List all chats.

**Response:** `[{"chat_id": "uuid", "name": "..."}]`

### `DELETE /chats/{chat_id}`
Delete a chat and its documents.

### `POST /chats/{chat_id}/upload`
Upload files to a chat.

| Field | Type | Description |
|---|---|---|
| `files` | File[] | PDF, DOCX, or TXT files (multiple) |

**Response:** `{"status": "success", "chunks": 42}`

### `POST /chats/{chat_id}/chat`
Ask a question.

| Field | Type | Description |
|---|---|---|
| `question` | form | Natural language question |

**Response:**
```json
{
  "answer": "The revenue grew by 23% in Q3 [1]. The uptick was driven by...",
  "citations": [
    {
      "source": "annual_report.pdf",
      "chunk_index": 7,
      "snippet": "Q3 revenue increased 23% year-over-year to $4.2B..."
    }
  ]
}
```

### `GET /chats/{chat_id}/history`
Get conversation history.

**Response:** `{"history": [{"role": "user"/"assistant", "content": "..."}]}`

### `GET /health`
Health check. **Response:** `{"status": "ok"}`

---

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application & routes
│   ├── config.py                 # Environment-based configuration (30+ settings)
│   ├── document_processor.py     # File parsing, chunking, OCR
│   ├── embedding_store.py        # ChromaDB vector store + metadata
│   ├── chat_engine.py            # Session management & history
│   ├── agent.py                  # Thin entry point → delegates to orchestrator
│   ├── agentic_orchestrator.py   # ★ Multi-agent LangGraph (core orchestrator)
│   ├── retrieval_tools.py        # ★ Multi-strategy retrieval (vector, BM25, hybrid, web, reranker, code)
│   ├── semantic_cache.py         # ★ Semantic cache (embedding-based query dedup)
│   └── mcp_server.py             # MCP protocol server (7 exposed tools)
├── frontend/
│   └── index.html                # SPA (vanilla JS, no build step)
├── .env                          # API keys (not committed)
├── .gitignore
├── Dockerfile
├── requirements.txt
├── run.py                        # Dev entry point
└── README.md
```

---

## Design Decisions

### Why not LlamaIndex?
LlamaIndex is excellent but opinionated. LangGraph gives **explicit control** over every step of the RAG pipeline — crucial when you need to debug, extend, or add custom validation (like groundedness checks).

### Why ChromaDB over FAISS?
ChromaDB is **persistent by default** and handles metadata filtering natively. FAISS requires separate serialization and offers no metadata support. ChromaDB's collection-per-chat model maps naturally to the multi-chat UI.

### Why not async everything?
The Groq API and ChromaDB synchronous client are both synchronous. Making them async would add complexity without throughput benefit for a single-user chatbot. The endpoints use `async def` for FastAPI compatibility but the core pipeline is synchronous — easy to convert to full async with `asyncio.to_thread()` if needed.

### Why EasyOCR + pdf2image?
Many real-world PDFs are scanned images. EasyOCR (which uses a deep learning model) extracts text from images with high accuracy. The `USE_OCR` flag can be disabled for pure text PDFs to save memory. The OCR model loads lazily to avoid slowing down server startup.

---

## License

MIT
