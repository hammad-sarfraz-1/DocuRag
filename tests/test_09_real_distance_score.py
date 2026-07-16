"""Bug 9: retrieval must return the REAL vector distance from Chroma, not a
hardcoded/constant placeholder score. Upload a short, topically narrow
document, then call search_vector() directly (bypassing reranking/LLM,
which would obscure the raw score) once with a highly relevant query and
once with a clearly irrelevant one -- the two scores must differ
meaningfully, and neither should be a suspicious constant like 0.0."""
import subprocess
import uuid

from _common import new_chat, delete_chat, delete_document, upload

FNAME = f"distance-test-{uuid.uuid4().hex[:8]}.txt"
CONTENT = (
    b"The quokka is a small marsupial native to Western Australia. "
    b"It is often called the world's happiest animal because of its "
    b"friendly smiling facial expression. Quokkas are herbivores and "
    b"mostly nocturnal, feeding on grasses, leaves, and bark. " * 8
)

chat_id = new_chat(f"distance-test-{uuid.uuid4().hex[:8]}")
try:
    r = upload(chat_id, FNAME, CONTENT)
    assert r.status_code == 200, f"upload failed: {r.status_code} {r.text}"

    script = f"""
from backend.retrieval_tools import search_vector
relevant = search_vector({chat_id!r}, "What is a quokka and where does it live?", k=3)
irrelevant = search_vector({chat_id!r}, "How do I configure a Kubernetes ingress controller?", k=3)
r_score = relevant[0].score if relevant else None
i_score = irrelevant[0].score if irrelevant else None
print("RELEVANT_SCORE=" + repr(r_score))
print("IRRELEVANT_SCORE=" + repr(i_score))
print("RELEVANT_TEXT_HAS_QUOKKA=" + repr("quokka" in relevant[0].text.lower()) if relevant else "RELEVANT_TEXT_HAS_QUOKKA=None")
"""
    result = subprocess.run(
        ["docker", "exec", "docurag_dev", "python3", "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"in-container script failed: {result.stderr}"
    out = {}
    for line in result.stdout.splitlines():
        if "=" in line and line.split("=", 1)[0].isupper():
            k, v = line.split("=", 1)
            out[k] = v
    print(f"  raw output: {out}")

    relevant_score = eval(out["RELEVANT_SCORE"])
    irrelevant_score = eval(out["IRRELEVANT_SCORE"])
    has_quokka = eval(out.get("RELEVANT_TEXT_HAS_QUOKKA", "None"))

    assert relevant_score is not None, "no results at all for relevant query"
    assert irrelevant_score is not None, "no results at all for irrelevant query"
    assert has_quokka, "relevant query's top hit doesn't even contain 'quokka' -- test setup is wrong"

    # Chroma cosine distance: 0.0 = identical, ~1-2 = unrelated. Real scores
    # should not both be exactly 0.0 (a hardcoded placeholder would often be
    # exactly 0.0) and must differ meaningfully between relevant/irrelevant.
    assert not (relevant_score == 0.0 and irrelevant_score == 0.0), (
        "both scores are exactly 0.0 -- looks like a hardcoded placeholder, not a real distance"
    )
    assert relevant_score != irrelevant_score, (
        f"relevant and irrelevant queries got IDENTICAL scores ({relevant_score}) -- not a real per-query distance"
    )
    # Lower distance = more similar for the relevant (on-topic) query.
    assert relevant_score < irrelevant_score, (
        f"expected relevant query's distance ({relevant_score}) to be LOWER (more similar) "
        f"than irrelevant query's distance ({irrelevant_score})"
    )

    print(f"  relevant distance={relevant_score:.4f}  irrelevant distance={irrelevant_score:.4f}")
    print("PASS: bug 9 real distance score (scores differ meaningfully, relevant < irrelevant, not hardcoded)")
finally:
    delete_document(chat_id, FNAME)
    delete_chat(chat_id)
