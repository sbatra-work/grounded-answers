"""The answer pipeline: check input, retrieve, decide, generate, verify.

Every exit path is an explicit `Answer` with a status and, for refusals, a
machine readable reason. Callers never get unverified model text.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from grounded_answers.citations import verify_citations
from grounded_answers.config import Settings
from grounded_answers.documents import Chunk, load_chunks
from grounded_answers.guards import check_question, looks_like_injection
from grounded_answers.llm import LLM, FakeLLM, OpenAICompatibleLLM
from grounded_answers.prompts import INSUFFICIENT_CONTEXT
from grounded_answers.retrieval import BM25Retriever, Retriever


class RefusalReason(StrEnum):
    SUSPECTED_INJECTION = "suspected_injection"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    MODEL_DECLINED = "model_declined"
    NO_VALID_CITATIONS = "no_valid_citations"


@dataclass(frozen=True)
class Answer:
    question: str
    text: str | None
    refusal: RefusalReason | None
    citations: list[Chunk] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    top_score: float = 0.0
    dropped_claims: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.refusal is not None


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        top_k: int = 4,
        min_retrieval_score: float = 3.0,
        max_question_chars: int = 500,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_retrieval_score
        self._max_chars = max_question_chars

    def ask(self, raw_question: str) -> Answer:
        question = check_question(raw_question, self._max_chars)  # raises InvalidQuestionError
        if looks_like_injection(question):
            return Answer(question, None, RefusalReason.SUSPECTED_INJECTION)

        hits = self._retriever.search(question, self._top_k)
        top_score = hits[0].score if hits else 0.0
        retrieved_ids = [h.chunk.id for h in hits]
        if top_score < self._min_score:
            return Answer(
                question, None, RefusalReason.LOW_RETRIEVAL_SCORE, [], retrieved_ids, top_score
            )

        chunks = [h.chunk for h in hits]
        raw = self._llm.generate(question, chunks)  # may raise LLMError
        if raw.strip() == INSUFFICIENT_CONTEXT:
            return Answer(
                question, None, RefusalReason.MODEL_DECLINED, [], retrieved_ids, top_score
            )

        report = verify_citations(raw, retrieved_ids)
        if not report.has_content:
            return Answer(
                question,
                None,
                RefusalReason.NO_VALID_CITATIONS,
                [],
                retrieved_ids,
                top_score,
                report.dropped_claims,
            )

        by_id = {c.id: c for c in chunks}
        return Answer(
            question,
            report.text,
            None,
            [by_id[i] for i in report.cited_ids],
            retrieved_ids,
            top_score,
            report.dropped_claims,
        )


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise ValueError(f"unknown GA_LLM_PROVIDER: {settings.llm_provider!r}")


def build_service(settings: Settings, llm: LLM | None = None) -> AnswerService:
    chunks = load_chunks(settings.docs_dir)
    return AnswerService(
        BM25Retriever(chunks),
        llm or build_llm(settings),
        top_k=settings.top_k,
        min_retrieval_score=settings.min_retrieval_score,
        max_question_chars=settings.max_question_chars,
    )
