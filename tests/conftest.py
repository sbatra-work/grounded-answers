from pathlib import Path

import pytest

from grounded_answers.documents import Chunk, load_chunks
from grounded_answers.retrieval import BM25Retriever

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="session")
def chunks() -> list[Chunk]:
    return load_chunks(DOCS_DIR)


@pytest.fixture(scope="session")
def retriever(chunks: list[Chunk]) -> BM25Retriever:
    return BM25Retriever(chunks)
