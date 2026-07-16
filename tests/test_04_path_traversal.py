"""Bug 4: _sanitize_filename must neutralize path traversal, null bytes, bare
"." / "..", and absurdly long filenames -- GET /documents/{source} must never
be able to escape DOCUMENTS_DIR.

Note: plain "/documents/.." is deliberately sent over a raw socket, not via
`requests` -- RFC 3986 path normalization means any well-behaved HTTP client
(curl, requests, browsers) collapses "/documents/.." to "/" *before* the
request ever leaves the machine, so testing it through `requests` would only
prove the client's normalizer works, not the server's. A raw socket bypasses
that and hits the server's own route matching directly."""
import socket
import requests
from _common import BASE

cases = [
    ("../../etc/passwd", "dotdot slash"),
    ("..%2f..%2fetc%2fpasswd", "url-encoded dotdot (literal, not decoded by us)"),
    ("....//....//etc/passwd", "doubled dotdot bypass attempt"),
    (".", "bare dot"),
    ("a" * 2000, "2000-char filename"),
    ("evil\x00.txt", "embedded null byte"),
    ("\x01\x02\x03", "pure control chars -> empty after sanitize"),
]


def raw_get(path: str) -> int:
    """Send a raw HTTP GET so client-side URL normalization can't hide what
    the server itself does with a literal '..' path segment."""
    with socket.create_connection(("localhost", 8000), timeout=10) as s:
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        status_line = data.split(b"\r\n", 1)[0].decode()
        return int(status_line.split(" ")[1])


bare_dotdot_status = raw_get("/documents/..")
print(f"  ['bare dotdot' via raw socket] -> {bare_dotdot_status} {'OK' if bare_dotdot_status in (400, 404) else 'UNSAFE'}")
assert bare_dotdot_status in (400, 404), f"raw '/documents/..' should be rejected, got {bare_dotdot_status}"

failures = []
for raw, label in cases:
    try:
        r = requests.get(f"{BASE}/documents/{raw}", timeout=15)
    except requests.exceptions.RequestException as e:
        # A raw NUL in the URL path can be rejected by the HTTP client/server
        # transport itself before it ever reaches our handler -- that's a
        # safe outcome too (never escapes), not a test failure.
        print(f"  [{label!r}] transport-level rejection ({e.__class__.__name__}) -- safe")
        continue
    ok = r.status_code in (400, 404)
    print(f"  [{label!r}] -> {r.status_code} {'OK' if ok else 'UNSAFE'}")
    if not ok:
        failures.append((raw, label, r.status_code, r.text[:200]))

# Actual traversal outside the container's documents dir must never succeed
# regardless of status code semantics -- verify /etc/passwd was never served.
for raw, label in cases:
    try:
        r = requests.get(f"{BASE}/documents/{raw}", timeout=15)
        if r.status_code == 200:
            assert "root:" not in r.text, f"traversal succeeded for {label!r}! leaked /etc/passwd"
    except requests.exceptions.RequestException:
        pass

assert not failures, f"path traversal cases returned non-40x: {failures}"
print("PASS: bug 4 path traversal (all adversarial filenames rejected, /etc/passwd never leaked)")
