"""Bug 12: answer_cache.put() must key each entry with a fresh uuid4(), not
something derived (e.g. a counter or hash) that could collide under
concurrent writes. Fire several first-turn questions across several fresh
chats CONCURRENTLY (each triggers its own put()) and confirm the resulting
cache entry count matches the number of puts attempted -- no entry lost or
silently overwritten."""
import concurrent.futures
import subprocess
import uuid

from _common import new_chat, delete_chat, ask

N = 6
questions = [f"Concurrency probe question number {i} - {uuid.uuid4().hex[:6]}?" for i in range(N)]
chats = [new_chat(f"cache-put-race-{i}-{uuid.uuid4().hex[:8]}") for i in range(N)]


def fire(chat_id, question):
    r = ask(chat_id, question)
    return r.status_code


try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        results = list(pool.map(lambda args: fire(*args), zip(chats, questions)))

    assert all(code == 200 for code in results), f"some concurrent asks failed: {results}"

    # Each chat has its own cache collection (see answer_cache._collection),
    # so a lost/overwritten put would show up as that chat's collection
    # having 0 entries instead of 1.
    check_script = "\n".join(
        [
            "from backend import answer_cache",
            "counts = {}",
        ]
        + [f"counts[{c!r}] = answer_cache._collection({c!r}).count()" for c in chats]
        + ["print(counts)"]
    )
    result = subprocess.run(
        ["docker", "exec", "docurag_dev", "python3", "-c", check_script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"count-check script failed: {result.stderr}"
    counts = eval(result.stdout.strip().splitlines()[-1])
    print(f"  per-chat cache entry counts: {counts}")

    assert len(counts) == N, f"expected {N} chats' collections checked, got {len(counts)}"
    for chat_id, count in counts.items():
        assert count == 1, f"chat {chat_id} has {count} cache entries, expected exactly 1 (lost or duplicated put)"

    total = sum(counts.values())
    assert total == N, f"expected {N} total cache entries across all chats, got {total}"

    print(f"PASS: bug 12 cache put() UUID no collision ({N} concurrent puts, {total} entries, none lost/overwritten)")
finally:
    for c in chats:
        delete_chat(c)
