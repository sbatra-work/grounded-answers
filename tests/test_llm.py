import json

import httpx
import pytest

from grounded_answers.documents import Chunk
from grounded_answers.llm import FakeLLM, LLMError, OpenAICompatibleLLM
from grounded_answers.prompts import INSUFFICIENT_CONTEXT, build_messages

CHUNKS = [
    Chunk("leave#1", "leave", "Leave", "Annual", "Staff receive 25 days of annual leave."),
    Chunk("leave#2", "leave", "Leave", "Sick", "Sick leave is 10 days. A note is needed."),
]


def test_fake_llm_quotes_best_sentence_with_its_chunk_id() -> None:
    reply = FakeLLM().generate("How many days of annual leave do staff receive?", CHUNKS)
    assert reply == "Staff receive 25 days of annual leave. [leave#1]"


def test_fake_llm_declines_when_context_does_not_cover_question() -> None:
    assert FakeLLM().generate("What is the dress code?", CHUNKS) == INSUFFICIENT_CONTEXT
    assert FakeLLM().generate("How many days?", []) == INSUFFICIENT_CONTEXT


def test_fake_llm_is_deterministic() -> None:
    question = "How many days of sick leave?"
    assert FakeLLM().generate(question, CHUNKS) == FakeLLM().generate(question, CHUNKS)


def test_prompt_contains_passages_with_ids_and_the_rules() -> None:
    system, user = build_messages("How many days?", CHUNKS)
    assert system["role"] == "system"
    assert INSUFFICIENT_CONTEXT in system["content"]
    assert '<passage id="leave#1"' in user["content"]
    assert user["content"].rstrip().endswith("</question>")


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_openai_client_sends_expected_request_and_returns_text() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " ok [leave#1] "}}]})

    llm = OpenAICompatibleLLM(
        "https://llm.test/v1/",
        "test-key",
        "test-model",
        client=_client(httpx.MockTransport(handler)),
    )
    assert llm.generate("How many days?", CHUNKS) == "ok [leave#1]"
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="not json"),
    ],
)
def test_openai_client_wraps_failures_in_llm_error(response: httpx.Response) -> None:
    transport = httpx.MockTransport(lambda request: response)
    llm = OpenAICompatibleLLM("https://llm.test/v1", "k", "m", client=_client(transport))
    with pytest.raises(LLMError):
        llm.generate("How many days?", CHUNKS)


def test_openai_client_requires_a_model_name() -> None:
    with pytest.raises(ValueError, match="GA_LLM_MODEL"):
        OpenAICompatibleLLM("https://llm.test/v1", "k", "")
