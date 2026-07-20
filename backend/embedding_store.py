import hashlib
from typing import List, Optional, Tuple
import chromadb
from backend.config import Config
from backend import answer_cache
from backend.embeddings import CHROMA_EMBEDDING_FN
from backend.resilience import ServiceUnavailable, call_with_retry


DOCUMENTS_COLLECTION = "documents"


class VectorStore:
    """ChromaDB vector store for document chunk persistence and retrieval.

    All chats share ONE collection: an upload is visible to every chat, and
    deleting a chat never touches the shared documents. `chat_id` params are
    kept on these methods for API compatibility but no longer affect scoping.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.PERSIST_DIR)
        self.embedding_fn = CHROMA_EMBEDDING_FN

    def get_collection(self, chat_id: str = None):
        return self.client.get_or_create_collection(
            name=DOCUMENTS_COLLECTION, embedding_function=self.embedding_fn
        )

    def add_documents(
        self,
        chat_id: str,
        chunks: List[str],
        metadatas: Optional[List[dict]] = None,
    ):
        collection = self.get_collection(chat_id)

        # Figure out which document these chunks belong to. We replace only
        # THIS document's existing chunks (so re-uploading the same file
        # refreshes it) while leaving other documents untouched.
        source = None
        document_id = None
        if metadatas and metadatas[0]:
            source = metadatas[0].get("source")
            document_id = metadatas[0].get("document_id")
        content_hash = hashlib.sha256("".join(chunks).encode("utf-8")).hexdigest()
        for m in metadatas or []:
            m["content_hash"] = content_hash

        if source:
            existing = collection.get(where={"source": source}, include=["metadatas"])
            existing_metas = existing.get("metadatas") or []
            if existing_metas:
                existing_hash = existing_metas[0].get("content_hash")
                # Collections are shared across all chats now, so a same-named
                # upload with different content is NOT necessarily a new
                # version of the same document — it may be an unrelated file
                # from another chat that happens to share a filename. We can't
                # tell those apart from filename alone, so rather than
                # silently deleting someone else's document, refuse and make
                # the caller handle it (e.g. rename the file before upload).
                if existing_hash is not None and existing_hash != content_hash:
                    raise ValueError(
                        f"A different document named '{source}' already exists "
                        f"in the shared store. Rename the file and re-upload, "
                        f"or delete the existing '{source}' first."
                    )
            try:
                collection.delete(where={"source": source})
            except Exception:
                pass
            # Namespace ids by document_id (unique per upload) so re-running
            # this same upload never collides with itself or anything else.
            ids = [f"{document_id}::chunk_{i}" for i in range(len(chunks))]
        else:
            # Fallback (no source metadata): preserve old single-doc behaviour.
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
            ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas if metadatas else None,
        )
        answer_cache.clear_all()

    def similarity_search(
        self, chat_id: str, query: str, k: int = Config.RETRIEVAL_K
    ) -> List[str]:
        """Return only document text for the top-k results."""
        collection = self.get_collection(chat_id)
        results = call_with_retry(
            lambda: collection.query(query_texts=[query], n_results=k),
            service="chroma", chat_id=chat_id,
            timeout=Config.CHROMA_TIMEOUT, attempts=Config.CHROMA_MAX_RETRIES,
        )
        return results["documents"][0] if results["documents"] else []

    def similarity_search_with_metadata(
        self, chat_id: str, query: str, k: int = Config.RETRIEVAL_K
    ) -> List[Tuple[str, dict, float]]:
        """Return (text, metadata, distance) triples so the agent can cite
        sources and score results using Chroma's real distance. Raises
        ServiceUnavailable if Chroma doesn't respond within the retry budget —
        callers (search_vector) fall back to BM25-only rather than erroring."""
        collection = self.get_collection(chat_id)
        results = call_with_retry(
            lambda: collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            ),
            service="chroma", chat_id=chat_id,
            timeout=Config.CHROMA_TIMEOUT, attempts=Config.CHROMA_MAX_RETRIES,
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)
        return list(zip(docs, metas, dists))

    def get_all_documents(self, chat_id: str) -> List[str]:
        """Return every chunk stored for a chat (for full-context prompts)."""
        collection = self.get_collection(chat_id)
        data = collection.get(include=["documents"])
        return data.get("documents", [])

    def get_all_documents_with_metadata(
        self, chat_id: str
    ) -> List[Tuple[str, dict]]:
        """Return every chunk as (text, metadata) pairs from a SINGLE get() call,
        so text and metadata are guaranteed to stay index-aligned."""
        collection = self.get_collection(chat_id)
        data = collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in docs]
        return list(zip(docs, metas))

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

    def get_document_list(self) -> List[dict]:
        """One entry per uploaded document (deduped by source), newest first,
        for a document-management UI. upload_date is stamped once per upload
        in build_chunk_metadata and identical across a document's own chunks."""
        collection = self.get_collection()
        data = collection.get(include=["metadatas"])
        by_source = {}
        for m in data["metadatas"] or []:
            if m and "source" in m:
                by_source[m["source"]] = m.get("upload_date", "")
        docs = [{"source": s, "upload_date": d} for s, d in by_source.items()]
        docs.sort(key=lambda d: d["upload_date"], reverse=True)
        return docs

    def get_document_count(self, chat_id: str) -> int:
        collection = self.get_collection(chat_id)
        return len(collection.get()["ids"])

    def get_document_metadata(self, chat_id: str, index: int) -> Optional[dict]:
        """Return metadata for a specific chunk by its position index."""
        collection = self.get_collection(chat_id)
        data = collection.get(include=["metadatas"])
        if data["metadatas"] and index < len(data["metadatas"]):
            return data["metadatas"][index]
        return None

    def get_chunk_count(self, chat_id: str) -> int:
        collection = self.get_collection(chat_id)
        return len(collection.get()["ids"])

    def delete_document(self, chat_id: str, source: str):
        """Remove an uploaded document (all its chunks) from the shared store."""
        collection = self.get_collection(chat_id)
        try:
            collection.delete(where={"source": source})
        except Exception:
            pass
        answer_cache.clear_all()

    def delete_chat(self, chat_id: str):
        """Deleting a chat only clears its own cache — the shared document
        store is untouched, so every other chat keeps working."""
        answer_cache.clear(chat_id)
