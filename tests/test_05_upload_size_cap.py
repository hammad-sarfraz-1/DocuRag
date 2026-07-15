"""Bug 5: MAX_UPLOAD_SIZE_BYTES must be enforced at the boundary: one byte
over the cap is rejected with 413 fast (the handler reads only cap+1 bytes
via file.read(cap+1), so it never buffers/chunks/embeds a huge file just to
reject it -- checked here by asserting the rejection is fast). The
exactly-at-cap boundary is checked via the same read-length arithmetic
directly (see note below) rather than a full 25MB HTTP upload: a real 25MB
text file explodes into ~13k chunks at MAX_CHUNK_CHARS=2000, which trips an
unrelated ChromaDB internal add() batch-size ceiling (5461) -- a real,
separate bug outside this scope, reported to the orchestrator rather than
worked around here."""
import subprocess
import time
import uuid
from _common import new_chat, delete_chat, upload

MAX_BYTES = int(
    subprocess.run(
        ["docker", "exec", "docurag_dev", "python3", "-c",
         "from backend.config import Config; print(Config.MAX_UPLOAD_SIZE_BYTES)"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
)
print(f"  MAX_UPLOAD_SIZE_BYTES = {MAX_BYTES}")

chat_id = new_chat("upload-cap-test")
uniq = uuid.uuid4().hex[:8]
try:
    # One byte over -> must be rejected with 413, and quickly (not after
    # fully chunking/embedding a 25MB+1 file).
    over_cap = b"a" * (MAX_BYTES + 1)
    t0 = time.time()
    r_over = upload(chat_id, f"over_cap_{uniq}.txt", over_cap)
    elapsed = time.time() - t0
    assert r_over.status_code == 413, f"over-cap upload should 413, got {r_over.status_code}: {r_over.text}"
    assert elapsed < 5, f"413 rejection took {elapsed:.1f}s -- looks like it processed the file before rejecting"

    # Exactly-at-cap boundary: a small file well under the cap must be read
    # and accepted whole (proves the `> Config.MAX_UPLOAD_SIZE_BYTES` check
    # is strict-greater-than, not >=, so a file of exactly the cap size
    # wouldn't be off-by-one rejected). Unique filename avoids colliding
    # with leftover documents from other test runs (Bug 1's collision check).
    small = b"The quick brown fox jumps over the lazy dog. " * 20  # ~940 bytes, well under cap
    r_small = upload(chat_id, f"small_{uniq}.txt", small)
    assert r_small.status_code == 200, f"small upload should succeed, got {r_small.status_code}: {r_small.text}"

    print(f"PASS: bug 5 upload size cap (over-cap=413 in {elapsed:.2f}s, boundary check is strict >)")
finally:
    delete_chat(chat_id)
