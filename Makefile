.PHONY: install lint format typecheck test eval eval-real check run

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

eval:
	uv run grounded-answers-eval

eval-real:
	uv run grounded-answers-eval --real

check: lint typecheck test eval

run:
	GA_API_KEYS=$${GA_API_KEYS:-dev-key} uv run uvicorn grounded_answers.api:create_app --factory --reload
