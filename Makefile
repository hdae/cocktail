.PHONY: install api-only client warmup test typecheck lint format gen-openapi clean

install:
	uv sync
	pnpm install

api-only:
	uv run uvicorn cocktail_server.main:app \
		--host 0.0.0.0 --port 8000 --reload \
		--app-dir apps/server/src

client:
	pnpm --filter @cocktail/client dev

warmup:
	uv run python -m cocktail_server.scripts.warmup

test:
	uv run pytest

typecheck:
	uv run mypy apps/server/src
	pnpm --filter @cocktail/api-types typecheck
	pnpm --filter @cocktail/client typecheck

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

# `packages/api-types/openapi.json` に FastAPI のスキーマをダンプする。
# 将来 openapi-typescript を足すなら、ここから `pnpm exec openapi-typescript` を続ける。
gen-openapi:
	uv run python -m cocktail_server.scripts.export_openapi \
		--out packages/api-types/openapi.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
