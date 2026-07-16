# Plan — In-App Semantic Answer Cache (DocuRag / DDRPRIV)

Cache final answers per chat, keyed by the *meaning* of the question, so repeated
or reworded questions skip the whole LangGraph run (retrieval → rerank → LLM).
Reuses what's already here — no new service, no extra network hop, no third party.

## Why in-app (not the external proxy)

- Data never leaves the process; no trust boundary, no extra hop.
- **Reuses DocuRag's own embeddings** (`all-MiniLM-L6-v2`) and ChromaDB — the cache's
  notion of "similar" matches the retriever's. One embedding space, no double cost.
- ~40 lines total. We throw away the proxy/router/streaming/multi-provider machinery
  from the standalone experiment — none of it is needed for one app.

## What we cache

`question → {answer, citations}` per chat. The answer, not the retrieved chunks.

Correctness hinges on **invalidation**: an answer is only valid for the chat's current
document set, so any document change drops that chat's cache (below).

## Design

Reuse the existing `VectorStore` (`backend/embedding_store.py`): same Chroma
`PersistentClient` + `SentenceTransformerEmbeddingFunction`. Add one cache collection
per chat: `cache_{chat_id}`, created with `metadata={"hnsw:space": "cosine"}` so
distance is cosine (Chroma defaults to L2 — must set this explicitly).

- **Lookup**: embed question (via the existing embedding fn) → query `cache_{chat_id}`
  top-1 → `similarity = 1 - distance`. Hit if `similarity >= CACHE_SIMILARITY_THRESHOLD`.
- **Store**: on a miss, after the graph produces the answer, add
  `documents=[question], metadatas=[{answer, citations(json)}]` to `cache_{chat_id}`.
- **Isolation**: per-chat collection = the isolation we already have; no cross-chat bleed.

New file `backend/answer_cache.py` (~40 lines): `get(chat_id, question)` /
`put(chat_id, question, answer, citations)` / `clear(chat_id)`.

## Steps

1. **Config** — `backend/config.py`: add
   `CACHE_SIMILARITY_THRESHOLD = float(os.getenv(..., "0.7"))` and
   `ENABLE_ANSWER_CACHE = os.getenv(..., "true")`. Conservative default — wrong cached
   answers are worse than a miss.
   - verify: values load.
2. **Cache module** — `backend/answer_cache.py`: `get/put/clear` over a
   `cache_{chat_id}` cosine collection, reusing `VectorStore`'s client + embedding fn.
   - verify: `put` then `get` same question → hit; unrelated question → `None`.
3. **Hook the read/write** — `backend/chat_engine.py:47 answer()`: before running the
   graph, `hit = answer_cache.get(chat_id, question)` → return it if present (skip graph).
   After the graph, `answer_cache.put(...)`. Gate on `ENABLE_ANSWER_CACHE`.
   - verify: ask once (miss, graph runs), re-ask a paraphrase (hit, graph does NOT run).
4. **Invalidate on document change** — call `answer_cache.clear(chat_id)` in:
   `embedding_store.add_documents` (upload, `app.py:149`), `delete_document`
   (`app.py:166`), and `delete_chat` (`chat_engine.py:79` / `embedding_store.py:137`).
   - verify: cached answer, then upload a doc → same question is a miss again.
5. **Signal it** — include `"cached": true|false` in the `answer()` return dict so the
   API/UI can show a "from cache" badge (optional, one field).
   - verify: response JSON carries the flag.

## Explicitly skipped (YAGNI)

- TTL expiry — document-change invalidation is the real correctness lever; add a TTL
  only if answers reference "today"-style volatile facts.
- Threshold tuner / adaptive thresholds / metrics dashboards — one env knob is enough
  until hit rate is worth measuring.
- Caching retrieved chunks or intermediate graph state — cache the final answer only.

## Success check

One `test_answer_cache.py`: put→get hit, unrelated→miss, clear→miss. No LLM/graph needed
(operate `answer_cache` directly against a temp Chroma dir).
