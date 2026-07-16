"""Bug 8: the shared ChatGroq client in backend/agents/utils.py must be
configured with a bounded timeout and a small retry count (timeout=60,
max_retries=2) instead of langchain_groq's defaults, so a hung/flaky Groq
call can't wedge a request thread forever.

Inspected in-process via docker exec against the live container -- this
reads real object state, not source text, so it can't be fooled by a
comment that doesn't match the actual constructor call."""
import subprocess

CONTAINER = "docurag_dev"

result = subprocess.run(
    [
        "docker", "exec", CONTAINER, "python3", "-c",
        "from backend.agents.utils import llm\n"
        "print('request_timeout=' + repr(llm.request_timeout))\n"
        "print('max_retries=' + repr(llm.max_retries))\n",
    ],
    capture_output=True, text=True, timeout=60,
)
assert result.returncode == 0, f"inspection script failed: {result.stderr}"

lines = [l for l in result.stdout.splitlines() if l.startswith("request_timeout=") or l.startswith("max_retries=")]
values = dict(l.split("=", 1) for l in lines)
print(f"  ChatGroq instance: {values}")

request_timeout = eval(values["request_timeout"])
max_retries = eval(values["max_retries"])

assert request_timeout == 60, f"expected request_timeout=60, got {request_timeout!r}"
assert max_retries == 2, f"expected max_retries=2, got {max_retries!r}"

print("PASS: bug 8 groq client config (request_timeout=60, max_retries=2, not library defaults)")
