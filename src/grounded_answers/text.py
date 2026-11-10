"""Tokenisation shared by retrieval and the fake model, so both see text the same way."""

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from get has have how i if in is it
    many may me much my of on or our than that the their there this to was we what
    when where which who why will with you your""".split()  # noqa: SIM905 (easier to read)
)


def _stem(word: str) -> str:
    """Very light plural folding. Enough for a handbook, not a linguistics project."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase content words with stopwords and single characters removed."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]
