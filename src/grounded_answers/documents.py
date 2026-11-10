"""Load markdown documents and split them into citable chunks.

Each `## ` section becomes one chunk, so a citation points a reader at a heading
they can find. Long sections are split on paragraph boundaries.
"""

from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_WORDS = 150


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str  # "<doc_id>#<n>", stable as long as the document's sections do not move
    doc_id: str
    doc_title: str
    section: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.doc_title}. {self.section}. {self.text}"


def _split_long(paragraphs: list[str], max_words: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    words = 0
    for para in paragraphs:
        n = len(para.split())
        if current and words + n > max_words:
            parts.append("\n\n".join(current))
            current, words = [], 0
        current.append(para)
        words += n
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(doc_id: str, markdown: str, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    title = doc_id
    sections: list[tuple[str, list[str]]] = []
    for raw in markdown.split("\n\n"):
        block = raw.strip()
        if not block:
            continue
        if block.startswith("# "):
            title = block[2:].strip()
        elif block.startswith("## "):
            heading, _, rest = block.partition("\n")
            sections.append((heading[3:].strip(), [rest.strip()] if rest.strip() else []))
        elif sections:
            sections[-1][1].append(block)
        else:
            sections.append(("Introduction", [block]))

    chunks: list[Chunk] = []
    for section, paragraphs in sections:
        for text in _split_long(paragraphs, max_words):
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{len(chunks) + 1}",
                    doc_id=doc_id,
                    doc_title=title,
                    section=section,
                    text=text,
                )
            )
    return chunks


def load_chunks(directory: Path) -> list[Chunk]:
    """Load every `*.md` file in `directory`. The file stem is the document id."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown documents found in {directory}")
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks
