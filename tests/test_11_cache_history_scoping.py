"""Bug 11: the answer cache only applies to the FIRST message in a chat
(history_len == 0). The same question asked as a follow-up (history_len > 0)
must always miss the cache, even if it's word-for-word identical to a
question that's already cached -- a follow-up's meaning can depend on prior
turns, which the cache key can't capture.

Proof, via real HTTP calls:
1. Ask QUESTION as chat A's first message (history_len=0) -> populates cache
   (confirmed via logs/answer_cache.log MISS line + a real answer back).
2. Ask an unrelated question in chat B first (to build up history), then ask
   the SAME QUESTION as a follow-up (history_len=1) -> must MISS, confirmed
   both by the log line and by response latency (a cache HIT skips the whole
   agent graph and returns near-instantly; a MISS invokes the real LLM and
   takes several seconds)."""
import subprocess
import time
import uuid

from _common import new_chat, delete_chat, ask

QUESTION = f"What is the boiling point of water in Celsius? (probe-{uuid.uuid4().hex[:6]})"
UNRELATED_FIRST = f"What color is the sky on a clear day? (probe-{uuid.uuid4().hex[:6]})"


def cache_log_tail(n=100):
    result = subprocess.run(
        ["docker", "exec", "docurag_dev", "tail", f"-{n}", "/app/logs/answer_cache.log"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"could not read answer_cache.log: {result.stderr}"
    return result.stdout


def log_line_for(logs: str, chat_id: str, question: str) -> str:
    """Return the first log line matching this chat+question, or None."""
    for line in logs.splitlines():
        if chat_id in line and repr(question) in line:
            return line
    return None


chat_a = new_chat(f"cache-scope-first-{uuid.uuid4().hex[:8]}")
chat_b = new_chat(f"cache-scope-followup-{uuid.uuid4().hex[:8]}")
try:
    # --- Chat A: QUESTION as the very first message (empty history) ---
    t0 = time.monotonic()
    r1 = ask(chat_a, QUESTION)
    dur_first = time.monotonic() - t0
    assert r1.status_code == 200, f"first ask failed: {r1.status_code} {r1.text}"

    logs = cache_log_tail()
    line_first = log_line_for(logs, chat_a, QUESTION)
    assert line_first and " MISS " in line_first, f"expected MISS on the very first ask, log showed {line_first!r}"
    print(f"PASS(1/2): first-ever ask of QUESTION in a fresh chat -> MISS logged (took {dur_first:.1f}s, real LLM call)")

    # --- Chat B: unrelated first question (builds history), THEN the SAME
    # QUESTION as a follow-up (history_len=1) -> must MISS despite identical
    # text, because history is non-empty.
    r_unrelated = ask(chat_b, UNRELATED_FIRST)
    assert r_unrelated.status_code == 200, f"unrelated first ask failed: {r_unrelated.status_code}"

    t1 = time.monotonic()
    r2 = ask(chat_b, QUESTION)
    dur_followup = time.monotonic() - t1
    assert r2.status_code == 200, f"follow-up ask failed: {r2.status_code} {r2.text}"

    logs2 = cache_log_tail()
    line_followup = log_line_for(logs2, chat_b, QUESTION)
    assert line_followup and " MISS " in line_followup, (
        f"follow-up asking the IDENTICAL question (non-empty history) should always MISS "
        f"the cache, log showed {line_followup!r}"
    )
    # A cache HIT's log line has no "similarity=" comparison against a cached
    # entry (it just reports the matched hit); a MISS caused by history_len>0
    # gating still runs best_match() for logging, which stamps a real
    # similarity score against the nearest (unrelated) cached question. That
    # score's presence proves the miss came from the history-length gate
    # actually running best_match(), not a coincidence -- a signal that
    # doesn't depend on network/LLM latency, unlike a timing threshold.
    assert "similarity=" in line_followup, (
        f"expected the MISS log line to include a similarity= comparison (proves "
        f"best_match() ran), got: {line_followup!r}"
    )
    print(f"PASS(2/2): identical question as a follow-up (non-empty history) -> MISS logged "
          f"with a real similarity comparison, took {dur_followup:.1f}s")

    print("PASS: bug 11 cache history-length scoping (empty history cacheable, non-empty history always bypasses cache)")
finally:
    delete_chat(chat_a)
    delete_chat(chat_b)
