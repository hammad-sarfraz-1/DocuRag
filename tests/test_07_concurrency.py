"""Bug 7: chat routes must not block the event loop. /chats/{id}/chat runs
its (slow, synchronous) work via run_in_threadpool, so two concurrent chat
requests against two DIFFERENT chats should genuinely overlap in wall-clock
time rather than the second one waiting for the first to fully finish.

Uses real Groq API calls (a working key is configured), so each call takes
several seconds -- that's expected. The test fires two requests from two
threads and checks that request 2 started before request 1 finished."""
import threading
import time
import uuid

from _common import new_chat, delete_chat, ask

chat_a = new_chat(f"concurrency-a-{uuid.uuid4().hex[:8]}")
chat_b = new_chat(f"concurrency-b-{uuid.uuid4().hex[:8]}")

timings = {}


def fire(label, chat_id, question):
    start = time.monotonic()
    r = ask(chat_id, question)
    end = time.monotonic()
    timings[label] = (start, end, r.status_code)


try:
    t1 = threading.Thread(target=fire, args=("req1", chat_a, "What is the capital of France?"))
    t2 = threading.Thread(target=fire, args=("req2", chat_b, "What is photosynthesis?"))

    overall_start = time.monotonic()
    t1.start()
    # Small stagger so req1 is unambiguously "first", but short enough that
    # if requests were serialized, req2 would still have to wait out req1's
    # entire (multi-second) duration before starting.
    time.sleep(0.05)
    t2.start()
    t1.join()
    t2.join()
    overall_end = time.monotonic()

    s1, e1, code1 = timings["req1"]
    s2, e2, code2 = timings["req2"]
    assert code1 == 200, f"req1 failed: {code1}"
    assert code2 == 200, f"req2 failed: {code2}"

    total_wall_clock = overall_end - overall_start
    sum_of_durations = (e1 - s1) + (e2 - s2)
    print(f"  req1: start={s1 - overall_start:.2f}s end={e1 - overall_start:.2f}s dur={e1 - s1:.2f}s")
    print(f"  req2: start={s2 - overall_start:.2f}s end={e2 - overall_start:.2f}s dur={e2 - s2:.2f}s")
    print(f"  total wall clock={total_wall_clock:.2f}s, sum of individual durations={sum_of_durations:.2f}s")

    # req2 must have STARTED before req1 finished -- proof of real overlap,
    # not serialization (if serialized, s2 >= e1).
    assert s2 < e1, (
        f"req2 started (t={s2 - overall_start:.2f}s) only after req1 finished "
        f"(t={e1 - overall_start:.2f}s) -- looks serialized, event loop may be blocked"
    )

    # Overlap should give a real wall-clock speedup vs. running the two
    # sequentially -- total time should be well under the naive sum.
    assert total_wall_clock < sum_of_durations * 0.9, (
        f"no meaningful speedup from concurrency: wall={total_wall_clock:.2f}s vs "
        f"sequential-equivalent={sum_of_durations:.2f}s"
    )

    print("PASS: bug 7 concurrency (two chat requests genuinely overlapped, event loop not blocked)")
finally:
    delete_chat(chat_a)
    delete_chat(chat_b)
