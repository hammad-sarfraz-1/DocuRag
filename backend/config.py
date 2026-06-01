import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

    PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
    CHAT_META_FILE = os.getenv("CHAT_META_FILE", "./chat_metadata.json")

    USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"
    OCR_LANG = ["en"]

    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_KEEP = int(os.getenv("RERANK_KEEP", "4"))

    ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"
    HYBRID_SEARCH_WEIGHT_VECTOR = float(os.getenv("HYBRID_SEARCH_WEIGHT_VECTOR", "0.6"))
    HYBRID_SEARCH_WEIGHT_BM25 = float(os.getenv("HYBRID_SEARCH_WEIGHT_BM25", "0.4"))

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
