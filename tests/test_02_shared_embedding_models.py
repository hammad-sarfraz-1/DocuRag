"""Bug 2: embeddings.py should load the underlying SentenceTransformer model
exactly ONCE per interface (2 total: Chroma's embedding fn + LangChain's),
not once per module that used to instantiate its own copy (3+ before the fix).
Checked two ways: container startup logs, and object identity inside the
running container's own process space."""
import subprocess

logs = subprocess.run(
    ["docker", "logs", "docurag_dev"], capture_output=True, text=True, timeout=30
).stdout

load_count = logs.count("Loading SentenceTransformer model from")
assert load_count == 2, f"expected exactly 2 model loads (one per embedding interface), got {load_count}"

# Cross-check via identity inside the live container: embedding_store and
# answer_cache must both be using the SAME CHROMA_EMBEDDING_FN singleton object.
check = subprocess.run(
    [
        "docker", "exec", "docurag_dev", "python3", "-c",
        "from backend.embedding_store import VectorStore\n"
        "from backend import answer_cache\n"
        "vs = VectorStore()\n"
        "assert vs.embedding_fn is answer_cache._embedding_fn, 'not the same singleton object'\n"
        "print('same-object-ok')\n",
    ],
    capture_output=True, text=True, timeout=60,
)
assert check.returncode == 0, f"identity check failed: {check.stdout}\n{check.stderr}"
assert "same-object-ok" in check.stdout, check.stdout

print(f"PASS: bug 2 shared embedding models (startup log shows {load_count} loads; singleton identity confirmed)")
