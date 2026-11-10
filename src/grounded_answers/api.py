"""HTTP layer. Thin on purpose: validation, auth, rate limiting, and mapping to JSON."""

import logging
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from grounded_answers.answer import Answer, AnswerService, build_service
from grounded_answers.config import Settings
from grounded_answers.guards import InvalidQuestionError, RateLimiter
from grounded_answers.llm import LLMError

log = logging.getLogger("grounded_answers")

# Hard ceiling on the request body field. The configurable, stricter limit is
# enforced in the service so non-HTTP callers get it too.
_BODY_MAX_CHARS = 4000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_BODY_MAX_CHARS)


class Citation(BaseModel):
    chunk_id: str
    document: str
    section: str


class AskResponse(BaseModel):
    status: Literal["answered", "refused"]
    answer: str | None
    refusal_reason: str | None
    citations: list[Citation]
    top_score: float
    dropped_claims: int


def to_response(answer: Answer) -> AskResponse:
    return AskResponse(
        status="refused" if answer.refused else "answered",
        answer=answer.text,
        refusal_reason=answer.refusal.value if answer.refusal else None,
        citations=[
            Citation(chunk_id=c.id, document=c.doc_title, section=c.section)
            for c in answer.citations
        ],
        top_score=round(answer.top_score, 3),
        dropped_claims=len(answer.dropped_claims),
    )


def create_app(
    service: AnswerService | None = None,
    settings: Settings | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or build_service(settings)
    limiter = limiter or RateLimiter(settings.rate_limit_per_minute)
    app = FastAPI(title="grounded-answers", version="0.1.0")

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
        if not x_api_key or x_api_key not in settings.api_keys:
            raise HTTPException(status_code=401, detail="missing or unknown API key")
        if not limiter.allow(x_api_key):
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(limiter.retry_after_s(x_api_key))},
            )
        return x_api_key

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, _key: Annotated[str, Depends(require_key)]) -> AskResponse:
        try:
            answer = service.ask(body.question)
        except InvalidQuestionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            log.warning("llm error: %s", exc)
            raise HTTPException(status_code=502, detail="answer model unavailable") from exc
        # Log decisions, not content: question text can contain personal data.
        log.info(
            "ask status=%s reason=%s top_score=%.2f dropped=%d",
            "refused" if answer.refused else "answered",
            answer.refusal,
            answer.top_score,
            len(answer.dropped_claims),
        )
        return to_response(answer)

    return app
