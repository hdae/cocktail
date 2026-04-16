.PHONY: install api-only warmup test typecheck lint format clean

install:
	uv sync
	pnpm install

api-only:
	uv run uvicorn cocktail_server.main:app \
		--host 0.0.0.0 --port 8000 --reload \
		--app-dir apps/server/src

warmup:
	uv run python -m cocktail_server.scripts.warmup

test:
	uv run pytest

typecheck:
	uv run mypy apps/server/src

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
