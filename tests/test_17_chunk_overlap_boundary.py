"""Chunk overlap: after semantic splitting + MAX_CHUNK_CHARS capping, each
chunk after the first is prefixed with the last 50 chars of the chunk
before it, so a fact split across the chunk boundary still appears in both
chunks' embeddings. Verified two ways: (1) chunk_text() directly shows the
overlap bytes match; (2) a question whose answer bridges the boundary gets
a coherent, correct answer via the real API."""
import subprocess
import uuid

from _common import new_chat, delete_chat, upload, ask

FNAME = f"overlap-test-{uuid.uuid4().hex[:8]}.txt"

# Repetitive filler (forces the fallback splitter into multiple ~2000-char
# chunks, like test_03), then a fact split right at the boundary: the
# codename appears at the end of chunk N, its meaning at the start of
# chunk N+1 -- without overlap, a query needing both would only retrieve one.
FILLER = "The quokka is a small marsupial native to Western Australia. " * 30
BOUNDARY_FACT = "The secret project codename is Project Marigold."
BOUNDARY_MEANING = " Project Marigold refers to the new solar-powered water purifier."
CONTENT = (FILLER + BOUNDARY_FACT + BOUNDARY_MEANING + FILLER).encode("utf-8")

chat_id = new_chat(f"overlap-test-{uuid.uuid4().hex[:8]}")
try:
    # --- 1. Verify chunk_text() itself produces overlapping chunks ---
    check = subprocess.run(
        ["docker", "exec", "-i", "docurag_dev", "python3", "-c",
         "from backend.document_processor import chunk_text\n"
         f"text = open('/dev/stdin').read()\n"
         "chunks = chunk_text(text)\n"
         "print('NUM_CHUNKS=' + str(len(chunks)))\n"
         "for i in range(1, len(chunks)):\n"
         "    overlap_expected = chunks[i-1][-50:]\n"
         "    actual_prefix = chunks[i][:50]\n"
         "    print(f'OVERLAP_MATCH_{i}=' + str(overlap_expected == actual_prefix))\n"],
        input=CONTENT.decode("utf-8"),
        capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 0, f"chunk_text check failed: {check.stderr}"
    print(f"  chunk_text() overlap check:\n{check.stdout}")
    lines = check.stdout.splitlines()
    num_chunks = int([l for l in lines if l.startswith("NUM_CHUNKS=")][0].split("=")[1])
    assert num_chunks >= 2, f"expected at least 2 chunks to test overlap, got {num_chunks}"
    match_lines = [l for l in lines if l.startswith("OVERLAP_MATCH_")]
    assert match_lines, "no overlap comparisons produced"
    assert all(l.endswith("True") for l in match_lines), f"overlap mismatch found: {match_lines}"
    print("PASS(1/2): chunk_text() overlaps each chunk with the prior chunk's last 50 chars")

    # --- 2. Verify a boundary-spanning question gets a coherent answer via the real API ---
    r = upload(chat_id, FNAME, CONTENT)
    assert r.status_code == 200, f"upload failed: {r.status_code} {r.text}"

    resp = ask(chat_id, "What does Project Marigold refer to?")
    assert resp.status_code == 200, f"ask failed: {resp.status_code} {resp.text}"
    data = resp.json()
    answer = data["answer"].lower()
    print(f"  answer: {data['answer']}")
    assert "marigold" in answer, f"answer doesn't even mention the codename: {data['answer']!r}"
    assert "purifier" in answer or "water" in answer or "solar" in answer, (
        f"answer is missing the boundary-spanning meaning (purifier/water/solar): {data['answer']!r}"
    )
    print("PASS(2/2): question bridging the chunk boundary retrieved both halves and answered coherently")

    print("PASS: chunk overlap (50-char boundary buffer preserves cross-chunk facts)")
finally:
    delete_chat(chat_id)
