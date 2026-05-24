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
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

    # Persistence
    PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
    CHAT_META_FILE = os.getenv("CHAT_META_FILE", "./chat_metadata.json")

    # OCR
    USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"
    OCR_LANG = ["en"]

    # Agent
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))

    # MCP Server
    MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT = int(os.getenv("MCP_PORT", "8100"))

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
