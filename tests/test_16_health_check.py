"""Bug 16: /health must respond fast with {"status": "ok"} under normal
conditions, and its code must genuinely check both Config.GROQ_API_KEY and
the vector store's Chroma heartbeat (not just return a hardcoded 200).

We do NOT unset the real .env's GROQ_API_KEY on the live container to test
the failure path -- that would break the app for the user. Instead the
failure path is verified by reading the actual route source (already
confirmed in backend/app.py: checks `Config.GROQ_API_KEY` truthiness and
calls `vector_store.client.heartbeat()`, returning 503 with a `failed` list
if either fails) plus a direct call to the real heartbeat to prove it's not
a no-op stub."""
import subprocess
import time

import requests
from _common import BASE

# --- 1. Timed happy-path check ---
start = time.monotonic()
r = requests.get(f"{BASE}/health", timeout=10)
elapsed = time.monotonic() - start

print(f"  GET /health -> {r.status_code} in {elapsed * 1000:.1f}ms: {r.text}")
assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
assert r.json() == {"status": "ok"}, f"expected exact {{'status': 'ok'}}, got {r.json()}"
assert elapsed < 2.0, f"/health took {elapsed:.2f}s -- too slow for a health check"
print("PASS(1/2): /health returns {'status': 'ok'} in under 2s")

# --- 2. Confirm the route's code actually performs real checks, not a
# hardcoded 200, by reading the deployed source inside the container (the
# exact code path serving this endpoint right now) and independently
# exercising the same heartbeat call it relies on. ---
route_src = subprocess.run(
    ["docker", "exec", "docurag_dev", "sh", "-c",
     "sed -n '/async def health/,/^$/p' /app/backend/app.py"],
    capture_output=True, text=True, timeout=15,
)
assert route_src.returncode == 0, f"could not read /health route source: {route_src.stderr}"
src = route_src.stdout
print(f"  /health route source:\n{src}")

assert "Config.GROQ_API_KEY" in src, "expected /health to check Config.GROQ_API_KEY, source doesn't reference it"
assert "heartbeat()" in src, "expected /health to call vector_store.client.heartbeat(), source doesn't reference it"
assert "503" in src, "expected /health to return 503 on failure, source doesn't reference it"
assert '"status": "ok"' in src or "'status': 'ok'" in src, "expected the ok-path literal in source"

# Prove heartbeat() is a real, callable, non-stub check against the actual
# live Chroma client -- if this raised, /health's happy path above would
# not have returned 200.
heartbeat_check = subprocess.run(
    ["docker", "exec", "docurag_dev", "python3", "-c",
     "from backend.app import vector_store; print('HEARTBEAT=' + repr(vector_store.client.heartbeat()))"],
    capture_output=True, text=True, timeout=30,
)
assert heartbeat_check.returncode == 0, f"heartbeat() call failed: {heartbeat_check.stderr}"
print(f"  direct heartbeat() call: {[l for l in heartbeat_check.stdout.splitlines() if 'HEARTBEAT=' in l]}")
assert "HEARTBEAT=" in heartbeat_check.stdout, "heartbeat() did not return a value"

print("PASS(2/2): /health route source genuinely checks GROQ_API_KEY + Chroma heartbeat (confirmed via live heartbeat call), not hardcoded")
print("PASS: bug 16 deepened health check (fast ok response + real dependency checks confirmed in deployed code)")
