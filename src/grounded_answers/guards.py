"""Cheap input checks that run before retrieval or any model call.

The injection check is a tripwire for obvious attempts, not a defence. The real
defence is structural: the model only sees retrieved passages, and anything it
says without a valid citation is removed afterwards.
"""

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,30}\b(instructions|rules|prompt)\b",
    r"\bdisregard\b.{0,30}\b(instructions|rules|prompt|documents)\b",
    r"\bforget\b.{0,30}\b(instructions|rules|documents|context)\b",
    r"\b(system|hidden|developer)\s+prompt\b",
    r"\b(developer|jailbreak|dan)\s+mode\b",
    r"\byou are now\b",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)


class InvalidQuestionError(ValueError):
    pass


def check_question(question: str, max_chars: int) -> str:
    """Return the normalised question or raise `InvalidQuestionError`."""
    cleaned = " ".join(question.split())
    if not cleaned:
        raise InvalidQuestionError("question is empty")
    if len(cleaned) > max_chars:
        raise InvalidQuestionError(f"question is longer than {max_chars} characters")
    return cleaned


def looks_like_injection(question: str) -> bool:
    return _INJECTION.search(question) is not None


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """In-memory token bucket per key. Fine for one process; see README for multi-node."""

    def __init__(
        self,
        per_minute: int,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self._rate = per_minute / 60.0
        self._capacity = float(burst if burst is not None else per_minute)
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = _Bucket(self._capacity, now)
            bucket.tokens = min(self._capacity, bucket.tokens + (now - bucket.updated) * self._rate)
            bucket.updated = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return True
            return False

    def retry_after_s(self, key: str) -> int:
        with self._lock:
            bucket = self._buckets.get(key)
            missing = 1 - (bucket.tokens if bucket else self._capacity)
        return max(1, int(missing / self._rate + 0.999))
