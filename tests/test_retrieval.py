import pytest

from grounded_answers.documents import Chunk
from grounded_answers.retrieval import BM25Retriever
from grounded_answers.text import tokenize


def test_tokenize_drops_stopwords_and_folds_plurals() -> None:
    assert tokenize("What are the Core Hours?") == ["core", "hour"]
    assert tokenize("policies") == ["policy"]
    assert tokenize("class") == ["class"]


@pytest.mark.parametrize(
    ("question", "expected_top"),
    [
        ("How many days of annual leave do employees get?", "leave-policy#1"),
        ("What is the hotel cap per night?", "expenses#2"),
        ("How do I report a lost laptop?", "security#3"),
        ("How long is the probation period?", "onboarding#3"),
    ],
)
def test_relevant_chunk_ranks_first(
    retriever: BM25Retriever, question: str, expected_top: str
) -> None:
    hits = retriever.search(question, top_k=3)
    assert hits[0].chunk.id == expected_top
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_no_shared_terms_returns_nothing(retriever: BM25Retriever) -> None:
    assert retriever.search("Who won the football world cup?", top_k=3) == []


def test_off_topic_question_scores_well_below_on_topic(retriever: BM25Retriever) -> None:
    on_topic = retriever.search("How long is parental leave for primary caregivers?", 1)
    off_topic = retriever.search("What is the dress code?", 1)
    assert on_topic[0].score > 3.0
    assert not off_topic or off_topic[0].score < 3.0


def test_top_k_is_respected(retriever: BM25Retriever) -> None:
    assert len(retriever.search("leave days", top_k=2)) == 2
    assert retriever.search("leave days", top_k=0) == []


def test_rare_terms_outweigh_common_ones() -> None:
    chunks = [
        Chunk("a#1", "a", "A", "S", "leave leave leave policy"),
        Chunk("b#1", "b", "B", "S", "leave and sabbatical rules"),
        Chunk("c#1", "c", "C", "S", "leave request form"),
    ]
    hits = BM25Retriever(chunks).search("sabbatical leave", top_k=3)
    assert hits[0].chunk.id == "b#1"


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Retriever([])
