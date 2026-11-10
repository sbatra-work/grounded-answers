# grounded-answers

A small RAG service that answers questions only from its own documents, cites the exact passages it used, and refuses when the documents don't support an answer.

**Status: the code is being moved into this repository now.** It includes:
- a FastAPI service with a clean module layout (ingest, retrieve, answer, citation check)
- guardrails: every claim must cite a retrieved passage, low-confidence questions are refused, and input size and rate are capped
- an eval harness covering answerable, unanswerable and prompt-injection questions
- pytest, ruff and mypy, run in GitHub Actions CI

Author: Shivam Batra
