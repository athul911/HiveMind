.DEFAULT_GOAL := help
COMPOSE := docker compose
DATA_SERVICES := postgres rabbitmq redis
APP_SERVICES := api worker

# Run Python/dev tools via the local .venv if it exists, else fall back to `uv run`.
ifneq (,$(wildcard .venv/bin/python))
RUN := . .venv/bin/activate &&
else
RUN := uv run
endif

.PHONY: help up down clean restart migrate seed seed-data logs \
        db-up db-down app-up app-down \
        test test-integration lint typecheck fmt install token proto

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies into .venv (uses uv if available, else python venv + pip)
	@if command -v uv >/dev/null 2>&1; then \
		uv venv && uv pip install -e ".[dev]"; \
	else \
		python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"; \
	fi

up: ## Start the full stack (auto-applies migrations before app starts)
	$(COMPOSE) up -d --build

down: ## Stop & remove containers — KEEPS the database volume (data persists)
	$(COMPOSE) down

clean: ## Stop and DELETE all volumes (full wipe — destroys the database)
	$(COMPOSE) down -v

restart: ## Restart only the app containers (api + worker), leaving data running
	$(COMPOSE) restart $(APP_SERVICES)

# ---- granular lifecycle: data plane vs app plane -------------------------
db-up: ## Start only Postgres + RabbitMQ + Redis
	$(COMPOSE) up -d $(DATA_SERVICES)

db-down: ## Stop the data services WITHOUT removing them (data persists)
	$(COMPOSE) stop $(DATA_SERVICES)

app-up: ## Build & start only the api + worker (runs migrations first)
	$(COMPOSE) up -d --build $(APP_SERVICES)

app-down: ## Stop only the api + worker (data services keep running)
	$(COMPOSE) stop $(APP_SERVICES)

logs: ## Tail API + worker logs
	$(COMPOSE) logs -f $(APP_SERVICES)

migrate: ## Apply database migrations manually (auto-run on `up`)
	$(COMPOSE) run --rm migrate

seed: ## Restart app containers (built-in agents hydrate on startup)
	$(COMPOSE) restart $(APP_SERVICES)

seed-data: ## Load a richer demo analytics dataset (customers/products/orders) for SQL testing
	$(COMPOSE) exec -T postgres psql -U hivemind -d hivemind < deploy/postgres/seed-demo.sql

test: ## Run the test suite with coverage
	$(RUN) pytest --cov=hivemind/core --cov=hivemind/api --cov-report=term-missing

test-integration: ## Run integration tests (needs the stack up)
	$(RUN) pytest -m integration

lint: ## Lint with ruff
	$(RUN) ruff check hivemind tests

fmt: ## Auto-format with ruff
	$(RUN) ruff check --fix hivemind tests && $(RUN) ruff format hivemind tests

typecheck: ## Type-check with mypy
	$(RUN) mypy hivemind

proto: ## Regenerate the gRPC executor stubs from the .proto
	$(RUN) python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
		hivemind/executor/proto/executor.proto

token: ## Mint a dev JWT (HS256) for local API calls
	@$(RUN) python scripts/mint_token.py
