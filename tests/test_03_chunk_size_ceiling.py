"""Bug 3: SemanticChunker has no size ceiling of its own -- highly repetitive
text can make it group everything into one giant "chunk" since consecutive
sentence embeddings never diverge. Upload pathological duplicate-paragraph
text and confirm every resulting chunk is <= MAX_CHUNK_CHARS (2000 by default)."""
import subprocess
import uuid
from _common import new_chat, delete_chat, upload

MAX_CHUNK_CHARS = int(
    subprocess.run(
        ["docker", "exec", "docurag_dev", "python3", "-c",
         "from backend.config import Config; print(Config.MAX_CHUNK_CHARS)"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
)

# Same sentence repeated hundreds of times -> semantically near-identical
# throughout, so SemanticChunker would want to keep it all as one chunk.
paragraph = "The zebra ran quickly across the open savanna at dawn. "
pathological_text = (paragraph * 400).encode("utf-8")  # ~22,800 chars, one "topic"

chat_id = new_chat("chunk-ceiling-test")
fname = f"repetitive-{uuid.uuid4().hex[:8]}.txt"
try:
    r = upload(chat_id, fname, pathological_text)
    assert r.status_code == 200, f"upload should succeed, got {r.status_code}: {r.text}"
    chunks_reported = r.json()["chunks"]
    assert chunks_reported >= 1

    # Inspect the actual stored chunk lengths inside the container.
    check = subprocess.run(
        [
            "docker", "exec", "docurag_dev", "python3", "-c",
            "from backend.embedding_store import VectorStore\n"
            "vs = VectorStore()\n"
            "col = vs.get_collection()\n"
            f"data = col.get(where={{'source': '{fname}'}}, include=['documents'])\n"
            "lens = [len(d) for d in data['documents']]\n"
            "print(max(lens), len(lens))\n",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 0, check.stderr
    max_len, num_chunks = map(int, check.stdout.strip().split())
    assert num_chunks > 1, f"expected the fallback splitter to break this into multiple chunks, got {num_chunks}"
    assert max_len <= MAX_CHUNK_CHARS, f"found a chunk of {max_len} chars, exceeds ceiling {MAX_CHUNK_CHARS}"

    print(f"PASS: bug 3 chunk size ceiling ({num_chunks} chunks, max {max_len} chars <= {MAX_CHUNK_CHARS})")
finally:
    delete_chat(chat_id)
