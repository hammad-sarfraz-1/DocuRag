import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

    # Persistence
    PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
    CHAT_META_FILE = os.getenv("CHAT_META_FILE", "./chat_metadata.json")

    # OCR
    USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"
    OCR_LANG = ["en"]

    # ---- Multi-Agent / Super RAG settings ----
    MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3"))
    ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
    SEMANTIC_CACHE_SIMILARITY = float(os.getenv("SEMANTIC_CACHE_SIMILARITY", "0.88"))
    SEMANTIC_CACHE_SIZE = int(os.getenv("SEMANTIC_CACHE_SIZE", "200"))

    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_KEEP = int(os.getenv("RERANK_KEEP", "4"))

    ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"
    HYBRID_SEARCH_WEIGHT_VECTOR = float(os.getenv("HYBRID_SEARCH_WEIGHT_VECTOR", "0.6"))
    HYBRID_SEARCH_WEIGHT_BM25 = float(os.getenv("HYBRID_SEARCH_WEIGHT_BM25", "0.4"))

    ENABLE_CODE_INTERPRETER = os.getenv("ENABLE_CODE_INTERPRETER", "true").lower() == "true"
    CODE_TIMEOUT_SECONDS = int(os.getenv("CODE_TIMEOUT_SECONDS", "15"))

    # Agent
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))

    # MCP Server
    MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT = int(os.getenv("MCP_PORT", "8100"))

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
