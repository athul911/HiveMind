.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up down migrate seed test lint typecheck fmt install token logs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv into a local venv
	uv venv && uv pip install -e ".[dev]"

up: ## Start the full local stack
	$(COMPOSE) up -d --build

down: ## Stop the stack and remove volumes
	$(COMPOSE) down -v

logs: ## Tail API + worker logs
	$(COMPOSE) logs -f api worker

migrate: ## Apply database migrations (inside the api container)
	$(COMPOSE) run --rm api alembic upgrade head

seed: ## Hydrate built-in agents (happens on startup; this is a no-op trigger)
	$(COMPOSE) restart api worker

test: ## Run the test suite with coverage
	uv run pytest --cov=hivemind/core --cov=hivemind/api --cov-report=term-missing

test-integration: ## Run integration tests (needs the stack up)
	uv run pytest -m integration

lint: ## Lint with ruff
	uv run ruff check hivemind tests

fmt: ## Auto-format with ruff
	uv run ruff check --fix hivemind tests && uv run ruff format hivemind tests

typecheck: ## Type-check with mypy
	uv run mypy hivemind

token: ## Mint a dev JWT (HS256) for local API calls
	@uv run python scripts/mint_token.py
