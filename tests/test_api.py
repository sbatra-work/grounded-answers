from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from grounded_answers.answer import AnswerService
from grounded_answers.api import create_app
from grounded_answers.config import Settings
from grounded_answers.documents import Chunk
from grounded_answers.guards import RateLimiter
from grounded_answers.llm import LLM, FakeLLM, LLMError
from grounded_answers.retrieval import BM25Retriever

KEY = {"X-API-Key": "test-key"}


def make_client(
    retriever: BM25Retriever, llm: LLM | None = None, per_minute: int = 100
) -> TestClient:
    settings = Settings(api_keys=frozenset({"test-key", "other-key"}), max_question_chars=200)
    service = AnswerService(retriever, llm or FakeLLM(), max_question_chars=200)
    return TestClient(create_app(service, settings, RateLimiter(per_minute)))


@pytest.fixture
def client(retriever: BM25Retriever) -> TestClient:
    return make_client(retriever)


def test_healthz_needs_no_key(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_answered_response_shape(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "How do I report a lost laptop?"}, headers=KEY)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert "extension 4400" in body["answer"]
    assert body["refusal_reason"] is None
    assert body["citations"] == [
        {
            "chunk_id": "security#3",
            "document": "Information Security Guidelines",
            "section": "Lost devices",
        }
    ]


def test_refusal_is_a_200_with_a_reason(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is the dress code?"}, headers=KEY)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["answer"] is None
    assert body["refusal_reason"] == "low_retrieval_score"
    assert body["citations"] == []


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_missing_or_unknown_key_is_401(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post("/ask", json={"question": "Annual leave?"}, headers=headers)
    assert response.status_code == 401


@pytest.mark.parametrize("question", ["", "   ", "x" * 201, "x" * 5000])
def test_bad_questions_are_422(client: TestClient, question: str) -> None:
    response = client.post("/ask", json={"question": question}, headers=KEY)
    assert response.status_code == 422


def test_rate_limit_is_per_key(retriever: BM25Retriever) -> None:
    client = make_client(retriever, per_minute=2)
    body = {"question": "How long is the probation period?"}
    assert client.post("/ask", json=body, headers=KEY).status_code == 200
    assert client.post("/ask", json=body, headers=KEY).status_code == 200
    limited = client.post("/ask", json=body, headers=KEY)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert client.post("/ask", json=body, headers={"X-API-Key": "other-key"}).status_code == 200


class BrokenLLM:
    def generate(self, question: str, chunks: Sequence[Chunk]) -> str:
        raise LLMError("upstream timeout with secret detail")


def test_model_failure_is_502_without_leaking_detail(retriever: BM25Retriever) -> None:
    client = make_client(retriever, llm=BrokenLLM())
    response = client.post("/ask", json={"question": "How long is probation?"}, headers=KEY)
    assert response.status_code == 502
    assert "secret" not in response.text
