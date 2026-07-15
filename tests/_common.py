"""Tiny shared helpers for the standalone bug-fix regression scripts in this
directory. Not a test framework — just avoids repeating the same three lines
of requests boilerplate in every script. Each test_*.py is still runnable
standalone: `python tests/test_01_filename_collision.py`.
"""
import requests

BASE = "http://localhost:8000"


def new_chat(name="test-chat"):
    r = requests.post(f"{BASE}/chats/new", data={"name": name}, timeout=30)
    r.raise_for_status()
    return r.json()["chat_id"]


def delete_chat(chat_id):
    requests.delete(f"{BASE}/chats/{chat_id}", timeout=30)


def upload(chat_id, filename, content_bytes, content_type="text/plain"):
    files = {"files": (filename, content_bytes, content_type)}
    return requests.post(f"{BASE}/chats/{chat_id}/upload", files=files, timeout=120)


def delete_document(chat_id, source):
    """Documents are shared across all chats (centralized store), so deleting
    a chat alone leaves the uploaded file behind -- tests must clean these up
    explicitly to avoid polluting the shared corpus other chats/tests see."""
    requests.delete(f"{BASE}/chats/{chat_id}/documents", params={"source": source}, timeout=30)


def ask(chat_id, question):
    return requests.post(f"{BASE}/chats/{chat_id}/chat", data={"question": question}, timeout=120)
