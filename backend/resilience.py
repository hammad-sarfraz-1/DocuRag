"""Shared timeout/retry/circuit-breaker helpers for external service calls
(Groq, Tavily, Chroma). Kept dependency-free and separate from
backend/agents/utils.py to avoid a circular import (retrieval_tools.py,
which agents/utils.py imports, needs this too)."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

# One shared pool for running blocking calls under a timeout. Threads, not
# processes — these calls are I/O-bound (network/disk), so no GIL contention
# to worry about, and a thread can be abandoned on timeout without cleanup.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="resilience")


class ServiceUnavailable(Exception):
    """Raised when a call to an external service exhausts all retries."""


def call_with_retry(fn, *, service: str, chat_id: str, timeout: float, attempts: int, backoff_base: float = 1.0):
    """Run fn() with a timeout, retrying on timeout or exception with
    exponential backoff (backoff_base, backoff_base*2, backoff_base*4, ...).
    Raises ServiceUnavailable if every attempt fails; logs each failure with
    chat_id, service name, and timestamp."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        future = _executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            last_exc = exc
            logger.warning(
                "chat=%s service=%s attempt=%d/%d timed out after %.1fs",
                chat_id, service, attempt, attempts, timeout,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "chat=%s service=%s attempt=%d/%d failed: %s",
                chat_id, service, attempt, attempts, exc,
            )
        if attempt < attempts:
            time.sleep(backoff_base * (2 ** (attempt - 1)))

    raise ServiceUnavailable(f"{service} unavailable after {attempts} attempts") from last_exc


class CircuitBreaker:
    """Trips after `threshold` consecutive failures; while tripped, calls are
    rejected immediately (no attempt made) until `cooldown` seconds pass."""

    def __init__(self, threshold: int, cooldown: float):
        self.threshold = threshold
        self.cooldown = cooldown
        self._consecutive_failures = 0
        self._tripped_at = None

    def is_open(self) -> bool:
        """True if the breaker is tripped and still cooling down."""
        if self._tripped_at is None:
            return False
        if time.monotonic() - self._tripped_at >= self.cooldown:
            # Cooldown elapsed — allow the next call through as a trial;
            # record_success/record_failure will re-trip or reset it.
            self._tripped_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self):
        self._consecutive_failures = 0
        self._tripped_at = None

    def record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold:
            self._tripped_at = time.monotonic()
