"""Bug 1: same filename + different content across chats must be rejected
(400), not silently overwrite. Same filename + IDENTICAL content twice must
succeed (replace-in-place) -- proving the fix isn't "reject all repeats"."""
import uuid
from _common import new_chat, delete_chat, delete_document, upload

FNAME = f"collision-test-{uuid.uuid4().hex[:8]}.txt"

chat_a = new_chat("collision-a")
chat_b = new_chat("collision-b")
try:
    r1 = upload(chat_a, FNAME, b"Original content about zebras. " * 5)
    assert r1.status_code == 200, f"first upload should succeed, got {r1.status_code}: {r1.text}"

    # Different content, same filename, different chat -> must be rejected.
    # (Bug 15's generic-error-message fix means the client only sees a
    # generic 400, not the raw ValueError text -- that detail lives in
    # `docker logs` instead, so we only assert on the status code here.)
    r2 = upload(chat_b, FNAME, b"Totally different content about rockets. " * 5)
    assert r2.status_code == 400, f"colliding different content should 400, got {r2.status_code}: {r2.text}"

    # Re-upload the EXACT same content under the same name -> must succeed (replace-in-place).
    r3 = upload(chat_a, FNAME, b"Original content about zebras. " * 5)
    assert r3.status_code == 200, f"identical re-upload should succeed, got {r3.status_code}: {r3.text}"

    print("PASS: bug 1 filename collision (reject differing content, allow identical replace)")
finally:
    delete_chat(chat_a)
    delete_chat(chat_b)
