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
┌──────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│   │ File Upload   │  │ Chat Input   │  │ Response + Sources   │  │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└──────────┼─────────────────┼──────────────────────┼──────────────┘
           │                 │                      │
      ┌────▼─────────────────▼──────────────────────▼──────────┐
      │                   FastAPI Server (:8000)                 │
      │                                                         │
      │  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐  │
      │  │ app.py   │─▶│ ChatEngine   │─▶│   LangGraph      │  │
      │  │ (routes) │  │ (sessions)   │  │   Agent          │  │
      │  └────┬─────┘  └──────┬───────┘  └────────┬────────┘  │
      │       │               │                    │           │
      │  ┌────▼─────┐  ┌──────▼───────┐  ┌────────▼────────┐  │
      │  │ doc_     │  │ VectorStore  │  │  Supervisor     │  │
      │  │ processor│  │ (ChromaDB)   │  │  (orchestrator) │  │
      │  └──────────┘  └──────────────┘  └────────┬────────┘  │
      │                                           │           │
      └───────────────────────────────────────────┼───────────┘
                                                  │
          ┌───────────────────────────────────────┼───────────┐
          │           MCP Server (:8100)          │           │
          │  ┌──────────────────────────────┐    │           │
          │  │ Tools:                       │◄───┘           │
          │  │  • search_documents          │                │
          │  │  • list_chat_sources         │                │
          │  │  • get_chat_context          │                │
          │  └──────────────────────────────┘                │
          │                                   Any MCP client │
          └──────────────────────────────────────────────────┘
```

### Data Flow

1. **Upload**: Files → `document_processor` (extract text + OCR) → `chunk_text` → `VectorStore.add_documents` (embed + store in ChromaDB)
2. **Question**: User query → `ChatEngine` → `supervisor_agent` (LangGraph) → retrieve → grade → generate → respond
3. **MCP**: External MCP clients (Claude Desktop, Cursor, etc.) can query the same vector store via the MCP server on port 8100

---

## Why These Technologies?

| Layer | Choice | Rationale |
|---|---|---|
| **Framework** | FastAPI | Async-native, automatic OpenAPI docs, high throughput, easy AWS deployment via Uvicorn |
| **Agent Orchestration** | LangGraph | State-graph architecture allows explicit node-by-node control over the RAG pipeline — superior to linear chains for debugging and extension |
| **Vector Store** | ChromaDB | Persistent, embeddable, zero external dependencies. Perfect for single-server deployments. SentenceTransformer integration out of the box |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Lightweight (80 MB), fast CPU inference, good semantic quality. No GPU needed |
| **LLM** | Groq (Llama 3.3 70B) | 2000+ tok/s inference speed, generous free tier, OpenAI-compatible API. Drop-in replaceable with any OpenAI-compatible endpoint via config |
| **Document Parsing** | pypdf + python-docx + EasyOCR | Scanned PDFs are handled via OCR fallback — a common real-world requirement |
| **Web Augmentation** | Tavily | Purpose-built search API for LLMs; returns clean, relevant content directly |
| **MCP** | `mcp` SDK (Anthropic) | Industry-standard protocol for LLM tool access. Makes the vector store available to any MCP-aware client |

---

## RAG Pipeline (Step by Step)

The core intelligence lives in `backend/agent.py` as a LangGraph state machine with 6 nodes:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  1. analyze_query                                     │
│     • Rewrites follow-up questions to be              │
│       self-contained using conversation history       │
│     • Falls back to original query if analysis fails  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  2. retrieve                                          │
│     • Calls VectorStore.similarity_search_with_       │
│       metadata() to get top-K chunks                 │
│     • Returns both text AND metadata                 │
│       (source filename, chunk index)                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  3. grade_documents                                   │
│     • Checks if retrieved chunks are relevant         │
│     • Fast-path: if citations exist → relevant        │
│     • LLM-based relevance check as fallback           │
│     • Routes to "generate" or "web_search"           │
└──────────┬──────────────────────────────────┬───────┘
           │                                  │
    [relevant]                          [not relevant]
           │                                  │
           ▼                                  ▼
┌──────────────────┐          ┌──────────────────────────────┐
│  4. generate      │          │  5. web_search               │
│     • Answer with │          │     • Tavily API search      │
│       document    │          │     • Merges web results     │
│       context     │          │       with document context  │
│     • Inline [1], │          │     • Feeds to generate node  │
│       [2] refs    │          └──────────────┬───────────────┘
└────────┬─────────┘                         │
         │                                   │
         └──────────┬────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  6. check_groundedness                                │
│     • LLM verifies every claim has a source           │
│     • If ungrounded → re-generate                     │
│     • Completes when grounded or max iterations       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
                  Final Answer
              (answer + citations[])
```

### Source Citations

Every retrieved chunk carries metadata about its origin:

```python
# backend/document_processor.py:build_chunk_metadata()
{
    "source": "report.pdf",       # Original filename
    "chunk_index": 3,             # Position in document
    "total_chunks": 15,           # Total chunks for this file
}
```

The `generate` node's prompt instructs the LLM to cite sources inline like `[1]`, `[2]`. The API returns a structured `citations` array alongside the answer text, which the frontend renders as a styled source panel below each assistant response.

---

## Agentic Enhancements

Beyond basic RAG, this implementation adds:

- **Query Rewriting** — Follow-up questions like "what about the second paragraph?" are rewritten using conversation history before retrieval
- **Document Grading** — Retrieved chunks are checked for relevance before generation; irrelevant results trigger web search instead
- **Groundedness Verification** — The answer is checked against sources; if unsubstantiated claims are found, the pipeline re-generates
- **Graceful Degradation** — If the LLM call fails at any node, the pipeline falls through with a sensible default rather than crashing
- **Configurable Knobs** — `CHUNK_SIZE`, `RETRIEVAL_K`, `RELEVANCE_THRESHOLD`, `MAX_ITERATIONS`, `USE_OCR` all live in `config.py` and can be overridden via environment variables

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
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `TAVILY_API_KEY` | — | Optional. Web search key |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHUNK_SIZE` | `500` | Document chunk characters |
| `CHUNK_OVERLAP` | `50` | Chunk overlap characters |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve |
| `USE_OCR` | `true` | Enable OCR for scanned PDFs |
| `MCP_PORT` | `8100` | MCP server port |
| `HOST` | `0.0.0.0` | FastAPI bind address |
| `PORT` | `8000` | FastAPI port |

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
│   ├── app.py                # FastAPI application & routes
│   ├── config.py             # Environment-based configuration
│   ├── document_processor.py # File parsing, chunking, OCR
│   ├── embedding_store.py    # ChromaDB vector store interface
│   ├── chat_engine.py        # Session management & history
│   ├── agent.py              # LangGraph RAG agent
│   └── mcp_server.py         # MCP protocol server
├── frontend/
│   └── index.html            # SPA (vanilla JS, no build step)
├── .env                      # API keys (not committed)
├── .gitignore
├── Dockerfile
├── requirements.txt
├── run.py                    # Dev entry point
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
