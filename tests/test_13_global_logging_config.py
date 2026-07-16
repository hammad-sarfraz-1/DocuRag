"""Bug 13: logging.basicConfig in backend/app.py must configure the root
logger with a real format (timestamp, level, logger name, message), not
Python's bare default ("<message>" with no metadata). Trigger a real action
(create + delete a chat) and grep docker logs for a properly formatted
INFO line."""
import re
import subprocess
import uuid

from _common import new_chat, delete_chat

FORMATTED_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO chat\.[0-9a-f-]+ chat created$"
)

chat_name = f"logging-format-probe-{uuid.uuid4().hex[:8]}"
chat_id = new_chat(chat_name)
try:
    result = subprocess.run(
        ["docker", "logs", "--tail", "200", "docurag_dev"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"docker logs failed: {result.stderr}"
    lines = result.stdout.splitlines() + result.stderr.splitlines()

    # create_chat() logs "chat created" through this chat's own logger
    # (chat.<chat_id>), which propagates to the root handler configured by
    # logging.basicConfig in backend/app.py -- so this line's formatting
    # reflects the global config, not a one-off print.
    matches = [l for l in lines if chat_id in l and FORMATTED_LINE_RE.match(l)]
    print(f"  matched line for this chat: {matches[-1] if matches else None}")
    assert matches, (
        f"no log line for chat {chat_id} matched the expected "
        "'YYYY-MM-DD HH:MM:SS,mmm INFO chat.<id> chat created' format "
        "-- logging.basicConfig may not be configured with a real formatter"
    )

    # Also confirm it's not bare/default output (default logging has no
    # timestamp/level/logger prefix at all -- just the raw message text).
    bare_default_lines = [l for l in lines if l.strip() == "chat created"]
    assert not bare_default_lines, (
        f"found a bare unformatted log line with no timestamp/level prefix: {bare_default_lines}"
    )

    print("PASS: bug 13 global logging config (formatted timestamp+level+logger+message lines present in docker logs)")
finally:
    delete_chat(chat_id)
