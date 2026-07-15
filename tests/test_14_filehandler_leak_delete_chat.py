"""Bug 14: ChatEngine.delete_chat() must close and detach the per-chat
logging.FileHandler it created in create_chat(), or every deleted chat leaks
one open file descriptor forever (logging.getLogger() never forgets a
logger name, so the handler -- and its open fd -- would otherwise live for
the lifetime of the process).

Verified two ways against the REAL running server process (not a fresh
docker exec process, which wouldn't share the server's in-memory logger
registry):
1. Before/after file descriptor count in /proc/<server-pid>/fd for the
   specific chat's log file.
2. The server process's open fd list must not contain logs/<chat_id>.log
   after delete."""
import subprocess
import uuid

from _common import new_chat, delete_chat

CONTAINER = "docurag_dev"
SERVER_PID = "7"  # the actual uvicorn worker process inside the container (verified via docker exec ps/proc)


def dexec(*args):
    r = subprocess.run(["docker", "exec", CONTAINER, *args], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"docker exec {args} failed: {r.stderr}"
    return r.stdout


# Sanity: confirm PID 7 really is the uvicorn process before trusting fd checks against it.
cmdline = dexec("sh", "-c", f"tr '\\0' ' ' < /proc/{SERVER_PID}/cmdline")
assert "uvicorn" in cmdline, f"PID {SERVER_PID} doesn't look like the uvicorn process: {cmdline!r}"

chat_id = new_chat(f"filehandler-leak-{uuid.uuid4().hex[:8]}")
log_filename = f"{chat_id}.log"

# Chat creation opens this chat's log file (create_chat -> _chat_logger ->
# FileHandler) inside the running server process.
fds_after_create = dexec("sh", "-c", f"ls -l /proc/{SERVER_PID}/fd 2>/dev/null | grep -F {log_filename!r} || true")
print(f"  fd entries referencing {log_filename} right after create: {fds_after_create.strip() or '(none found)'}")
assert log_filename in fds_after_create, (
    f"expected the server process to have an open fd for {log_filename} after create_chat(), found none"
)

delete_chat(chat_id)

fds_after_delete = dexec("sh", "-c", f"ls -l /proc/{SERVER_PID}/fd 2>/dev/null | grep -F {log_filename!r} || true")
print(f"  fd entries referencing {log_filename} after delete_chat(): {fds_after_delete.strip() or '(none -- closed)'}")
assert log_filename not in fds_after_delete, (
    f"FILE DESCRIPTOR LEAK: {log_filename} is still open in the server process after delete_chat(): {fds_after_delete}"
)

print("PASS: bug 14 FileHandler leak fix (per-chat log fd opened on create, closed on delete_chat)")
