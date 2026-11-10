"""Language model behind an interface, with a real client and a deterministic fake.

The interface takes the question and the retrieved chunks rather than a raw
prompt string. That keeps prompt building in one place (`prompts.py`) and lets
the fake work on structured data instead of parsing text.
"""

from collections.abc import Sequence
from typing import Protocol

import httpx

from grounded_answers.documents import Chunk
from grounded_answers.prompts import INSUFFICIENT_CONTEXT, build_messages
from grounded_answers.text import tokenize


class LLMError(RuntimeError):
    """The model call failed. Callers should not retry blindly or invent an answer."""


class LLM(Protocol):
    def generate(self, question: str, chunks: Sequence[Chunk]) -> str: ...


class OpenAICompatibleLLM:
    """Calls any server that speaks the OpenAI chat completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not model:
            raise ValueError("GA_LLM_MODEL must be set to use the openai provider")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s)

    def generate(self, question: str, chunks: Sequence[Chunk]) -> str:
        payload = {
            "model": self._model,
            "messages": build_messages(question, chunks),
            "temperature": 0,
        }
        try:
            response = self._client.post(self._url, json=payload, headers=self._headers)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"model call failed: {exc}") from exc
        if not isinstance(content, str):
            raise LLMError("model returned no text content")
        return content.strip()


class FakeLLM:
    """Deterministic extractive stand-in for a model.

    It returns the single sentence from the context that shares the most content
    words with the question, cited with its chunk id, or declines when no
    sentence covers at least `min_overlap` of the question's content words.
    It exists to test the pipeline and guardrails offline, not to measure model
    quality. Ties go to the earlier chunk, which is the higher ranked one.
    """

    def __init__(self, min_overlap: float = 0.5) -> None:
        self._min_overlap = min_overlap

    def generate(self, question: str, chunks: Sequence[Chunk]) -> str:
        wanted = set(tokenize(question))
        if not wanted:
            return INSUFFICIENT_CONTEXT
        best: tuple[float, str, str] | None = None
        for chunk in chunks:
            for sentence in _sentences(chunk.text):
                overlap = len(wanted & set(tokenize(sentence))) / len(wanted)
                if best is None or overlap > best[0]:
                    best = (overlap, sentence, chunk.id)
        if best is None or best[0] < self._min_overlap:
            return INSUFFICIENT_CONTEXT
        _, sentence, chunk_id = best
        return f"{sentence} [{chunk_id}]"


class ScriptedLLM:
    """Returns a fixed reply. Used in tests to simulate a misbehaving model."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, question: str, chunks: Sequence[Chunk]) -> str:
        self.calls += 1
        return self.reply


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        out.extend(s.strip() for s in line.replace(". ", ".\n").split("\n") if s.strip())
    return out
