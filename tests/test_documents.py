from pathlib import Path

import pytest

from grounded_answers.documents import Chunk, chunk_markdown, load_chunks

DOC = """# Pet Policy

## Dogs

Dogs are welcome on Fridays.

They must be on a lead.

## Cats

Cats are not allowed in the workshop.
"""


def test_each_section_becomes_one_chunk_with_stable_ids() -> None:
    chunks = chunk_markdown("pets", DOC)
    assert [c.id for c in chunks] == ["pets#1", "pets#2"]
    assert chunks[0].doc_title == "Pet Policy"
    assert chunks[0].section == "Dogs"
    assert "on a lead" in chunks[0].text
    assert chunks[1].text == "Cats are not allowed in the workshop."


def test_long_sections_split_on_paragraphs() -> None:
    chunks = chunk_markdown("pets", DOC, max_words=6)
    assert [c.section for c in chunks] == ["Dogs", "Dogs", "Cats"]
    assert [c.id for c in chunks] == ["pets#1", "pets#2", "pets#3"]


def test_text_before_any_heading_is_kept() -> None:
    chunks = chunk_markdown("note", "Just a paragraph.")
    assert chunks == [Chunk("note#1", "note", "note", "Introduction", "Just a paragraph.")]


def test_load_chunks_reads_sample_documents(chunks: list[Chunk]) -> None:
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_id for c in chunks} == {
        "expenses",
        "leave-policy",
        "onboarding",
        "remote-work",
        "security",
    }


def test_load_chunks_fails_loudly_on_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path)
