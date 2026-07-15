"""Bug 15: a real processing failure (e.g. a .pdf that isn't actually a PDF,
so pypdf's parser genuinely raises) must surface to the API client as a
generic message -- no raw Python exception text, class name, or traceback
-- while the REAL exception detail still lands in docker logs for
debugging."""
import os
import subprocess
import uuid

from _common import new_chat, delete_chat, upload

FNAME = f"garbage-{uuid.uuid4().hex[:8]}.pdf"
# Random bytes with a .pdf extension -- not a valid PDF at all, guaranteed
# to make pypdf's PdfReader raise rather than silently return empty text.
GARBAGE_BYTES = os.urandom(2048)

# Leak indicators that must NEVER appear in the client-facing JSON response.
LEAK_MARKERS = [
    "Traceback (most recent call last)",
    "PdfReadError",
    "pypdf",
    ".py\", line",
    "site-packages",
]

chat_id = new_chat(f"generic-error-{uuid.uuid4().hex[:8]}")
try:
    r = upload(chat_id, FNAME, GARBAGE_BYTES, content_type="application/pdf")
    print(f"  upload response: {r.status_code} {r.text[:300]}")

    assert r.status_code == 400, f"expected a 400 for unparseable PDF, got {r.status_code}: {r.text}"

    body = r.text
    for marker in LEAK_MARKERS:
        assert marker not in body, f"raw exception detail leaked into API response: found {marker!r} in {body!r}"

    # The response body should read as a plain, generic message.
    data = r.json()
    detail = data.get("detail", "")
    assert isinstance(detail, str) and detail, f"expected a non-empty string 'detail', got {data!r}"
    assert FNAME in detail, f"expected the generic message to at least name the file, got {detail!r}"
    print(f"  client-facing detail: {detail!r}")

    # Meanwhile, docker logs must show the REAL exception for debugging.
    logs = subprocess.run(
        ["docker", "logs", "--tail", "100", "docurag_dev"],
        capture_output=True, text=True, timeout=30,
    )
    assert logs.returncode == 0
    combined = logs.stdout + logs.stderr
    assert FNAME in combined, f"expected docker logs to mention {FNAME}, found nothing"
    assert ("Traceback" in combined or "Error" in combined or "error" in combined.lower()), (
        "expected docker logs to contain real exception detail for this failed upload"
    )
    print("  confirmed docker logs contain real exception/traceback detail for this failure")

    print("PASS: bug 15 generic error messages (no raw exception leak to client, real detail in docker logs)")
finally:
    delete_chat(chat_id)
