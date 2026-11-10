"""Settings read from environment variables. No secrets have defaults."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("data/docs")
    llm_provider: str = "fake"  # "fake" or "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    top_k: int = 4
    min_retrieval_score: float = 3.0
    max_question_chars: int = 500
    rate_limit_per_minute: int = 30
    api_keys: frozenset[str] = field(default_factory=frozenset, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        d = cls()
        return cls(
            docs_dir=Path(e.get("GA_DOCS_DIR", str(d.docs_dir))),
            llm_provider=e.get("GA_LLM_PROVIDER", d.llm_provider),
            llm_base_url=e.get("GA_LLM_BASE_URL", d.llm_base_url),
            llm_api_key=e.get("GA_LLM_API_KEY", ""),
            llm_model=e.get("GA_LLM_MODEL", ""),
            top_k=int(e.get("GA_TOP_K", d.top_k)),
            min_retrieval_score=float(e.get("GA_MIN_RETRIEVAL_SCORE", d.min_retrieval_score)),
            max_question_chars=int(e.get("GA_MAX_QUESTION_CHARS", d.max_question_chars)),
            rate_limit_per_minute=int(e.get("GA_RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute)),
            api_keys=frozenset(k.strip() for k in e.get("GA_API_KEYS", "").split(",") if k.strip()),
        )
