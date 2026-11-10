"""Retrieval behind a small interface. BM25 is the default implementation.

BM25 is used because it is dependency free, deterministic and easy to reason
about on a corpus of a few dozen chunks. An embedding retriever only needs to
implement `Retriever.search`.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from grounded_answers.documents import Chunk
from grounded_answers.text import tokenize


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to `top_k` chunks, best first. Scores are only comparable within one
        retriever, which is why the refusal threshold is configured per deployment."""
        ...


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._term_freqs = [Counter(tokenize(c.searchable_text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_len = sum(self._lengths) / len(self._lengths)
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(self._chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def _score(self, terms: list[str], i: int) -> float:
        tf = self._term_freqs[i]
        norm = self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avg_len)
        score = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if f:
                score += self._idf[term] * f * (self._k1 + 1) / (f + norm)
        return score

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        terms = list(dict.fromkeys(tokenize(query)))  # dedupe, keep order
        if not terms or top_k <= 0:
            return []
        scored = [ScoredChunk(c, self._score(terms, i)) for i, c in enumerate(self._chunks)]
        scored = [s for s in scored if s.score > 0]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
