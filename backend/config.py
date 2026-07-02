import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    # Total prompt+response token budget for the synthesis call. Retrieved-chunk
    # context is sized dynamically to whatever's left after the system prompt, chat
    # history, and RESPONSE_TOKEN_RESERVE are subtracted (see agentic_orchestrator.py).
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
    RESPONSE_TOKEN_RESERVE = int(os.getenv("RESPONSE_TOKEN_RESERVE", "1024"))

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
    # When a chat holds this few chunks or fewer, skip top-k truncation and feed
    # every chunk to the LLM (rerank only to order them) so the right document is
    # always in context. Above this, fall back to ranked top-k retrieval.
    SMALL_CORPUS_CHUNKS = int(os.getenv("SMALL_CORPUS_CHUNKS", "25"))

    PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
    CHAT_META_FILE = os.getenv("CHAT_META_FILE", "./chat_metadata.json")
    # Kept inside PERSIST_DIR by default so chat history lands on the same
    # persisted volume as the vector store (survives restarts/redeploys).
    HISTORY_FILE = os.getenv("HISTORY_FILE", os.path.join(PERSIST_DIR, "chat_history.json"))

    USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"
    OCR_LANG = ["en"]

    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_KEEP = int(os.getenv("RERANK_KEEP", "6"))
    # Web search kicks in when the top retrieved chunk's score falls below this
    # (cross-encoder relevance score; below 0 roughly means "not actually relevant").
    WEB_FALLBACK_SCORE_THRESHOLD = float(os.getenv("WEB_FALLBACK_SCORE_THRESHOLD", "0.0"))

    ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"
    HYBRID_SEARCH_WEIGHT_VECTOR = float(os.getenv("HYBRID_SEARCH_WEIGHT_VECTOR", "0.6"))
    HYBRID_SEARCH_WEIGHT_BM25 = float(os.getenv("HYBRID_SEARCH_WEIGHT_BM25", "0.4"))

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
