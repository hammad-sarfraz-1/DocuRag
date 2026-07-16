"""Single load point for the embedding model, shared across the app.

embedding_store.py, answer_cache.py, and document_processor.py each used to
create their own copy of the same underlying model (Config.EMBEDDING_MODEL),
tripling load time/RAM at startup. Two singletons here, one per interface:
Chroma's EmbeddingFunction protocol and LangChain's Embeddings interface are
different classes and can't stand in for each other, so each gets one shared
instance instead of trying to force a single wrapper for both.
"""
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import Config

CHROMA_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=Config.EMBEDDING_MODEL
)
LANGCHAIN_EMBEDDING_FN = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
