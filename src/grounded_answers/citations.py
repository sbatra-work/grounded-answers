"""Check that every claim in a model answer cites a chunk we actually retrieved.

A "claim" is a sentence. Any sentence with no citation, or that cites an id
outside the retrieved set, is removed. The policy is strict on purpose: a
sentence that cites one real chunk and one invented one is dropped, because an
invented id is a sign the model is not reading the context.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass, field

CITATION = re.compile(r"\[([A-Za-z0-9_-]+#\d+)\]")
# Split after ., !, ? or ] when the next thing is not another citation.
_CLAIM_BOUNDARY = re.compile(r"(?<=[.!?\]])\s+(?!\[)")


@dataclass(frozen=True, slots=True)
class CitationReport:
    text: str  # the answer with unsupported claims removed
    cited_ids: list[str]  # unique, in order of first use
    dropped_claims: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.text)


def split_claims(answer: str) -> list[str]:
    return [c.strip() for c in _CLAIM_BOUNDARY.split(answer.strip()) if c.strip()]


def verify_citations(answer: str, allowed_ids: Collection[str]) -> CitationReport:
    allowed = set(allowed_ids)
    kept: list[str] = []
    dropped: list[str] = []
    cited: dict[str, None] = {}
    for claim in split_claims(answer):
        ids = CITATION.findall(claim)
        if ids and all(i in allowed for i in ids):
            kept.append(claim)
            cited.update(dict.fromkeys(ids))
        else:
            dropped.append(claim)
    return CitationReport(text=" ".join(kept), cited_ids=list(cited), dropped_claims=dropped)
