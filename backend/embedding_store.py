from typing import List, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from backend.config import Config


class VectorStore:
    """ChromaDB vector store for document chunk persistence and retrieval."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.PERSIST_DIR)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )

    def _safe_collection_name(self, chat_id: str) -> str:
        return chat_id.replace("-", "_")

    def get_collection(self, chat_id: str):
        safe_id = self._safe_collection_name(chat_id)
        return self.client.get_or_create_collection(
            name=safe_id, embedding_function=self.embedding_fn
        )

    def add_documents(
        self,
        chat_id: str,
        chunks: List[str],
        metadatas: Optional[List[dict]] = None,
    ):
        collection = self.get_collection(chat_id)

        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas if metadatas else None,
        )

    def similarity_search(
        self, chat_id: str, query: str, k: int = Config.RETRIEVAL_K
    ) -> List[str]:
        """Return only document text for the top-k results."""
        collection = self.get_collection(chat_id)
        results = collection.query(query_texts=[query], n_results=k)
        return results["documents"][0] if results["documents"] else []

    def similarity_search_with_metadata(
        self, chat_id: str, query: str, k: int = Config.RETRIEVAL_K
    ) -> List[Tuple[str, dict]]:
        """Return (text, metadata) pairs so the agent can cite sources."""
        collection = self.get_collection(chat_id)
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        return list(zip(docs, metas))

    def get_all_documents(self, chat_id: str) -> List[str]:
        """Return every chunk stored for a chat (for full-context prompts)."""
        collection = self.get_collection(chat_id)
        data = collection.get(include=["documents"])
        return data.get("documents", [])

    def get_sources(self, chat_id: str) -> List[str]:
        """Return sorted unique source filenames for a chat."""
        collection = self.get_collection(chat_id)
        data = collection.get(include=["metadatas"])
        if not data["metadatas"]:
            return []
        sources = set()
        for m in data["metadatas"]:
            if m and "source" in m:
                sources.add(m["source"])
        return sorted(sources)

    def get_document_count(self, chat_id: str) -> int:
        collection = self.get_collection(chat_id)
        return len(collection.get()["ids"])

    def delete_chat(self, chat_id: str):
        safe_id = self._safe_collection_name(chat_id)
        try:
            self.client.delete_collection(safe_id)
        except ValueError:
            pass
