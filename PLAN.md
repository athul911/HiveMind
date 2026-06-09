# HiveMind — Implementation Plan

> Multi-agent AI orchestration platform on **LangGraph** with OpenAI-compatible REST,
> SSE + RabbitMQ execution modes, Postgres persistence, and full observability.

This document is the build plan. For *why* each design decision was made, see
[DESIGN.md](DESIGN.md).

---

## 0. Locked decisions (from kickoff)

| Decision | Choice |
| --- | --- |
| Build sequencing | **Working vertical slice first** — full scaffold + a runnable end-to-end path, then widen. |
| Code-exec sandbox | **Container-per-execution** (ephemeral Docker: no net, read-only rootfs, dropped caps, cpu/mem/pid limits, non-root, wall-clock timeout). Abstracted behind a `Sandbox` interface with a hardened-subprocess fallback. |
| Auth | **HS256 shared-secret JWT default**, OAuth2 **JWKS/RS256 optional** via `OAUTH2_JWKS_URL`. |
| LLM providers | **All five fully implemented**: Anthropic, OpenAI, Azure OpenAI, vLLM, Ollama. OpenAI/Azure/vLLM share an OpenAI-compatible base adapter. |
| Skills | Markdown files w/ YAML frontmatter; **per-agent**, **injected into the system prompt at agent-load time** (with progressive-disclosure metadata to save tokens). |
| Package manager | **uv** + `pyproject.toml` (PEP 621). Python 3.11. |

---

## 1. Improvements made to the original spec

These are deliberate hardenings/changes over the original prompt. Rationale in DESIGN.md §"Spec deltas".

1. **Code execution is container-isolated, not "sandboxed subprocess".** A raw subprocess
   running arbitrary LLM-generated Python is not production-safe. We define a `Sandbox`
   abstraction with a Docker backend (default) and a hardened-subprocess backend (no-Docker
   fallback for laptops/CI).
2. **Least-privilege SQL.** The SQL tool connects with a **separate read-only role**, enforces
   a `statement_timeout`, a hard row cap, a schema allow-list, and rejects multi-statement /
   DDL / DML by default. LLM-authored SQL is treated as hostile.
3. **Skills subsystem** added end-to-end: `Skill` model, markdown loader w/ frontmatter,
   `SkillRegistry`, per-agent binding, progressive-disclosure injection, `skills` table,
   `/v1/skills` read endpoints.
4. **Official LangGraph Postgres checkpointer** (`langgraph-checkpoint-postgres`,
   `AsyncPostgresSaver`) is the source of truth for graph state. The spec's `checkpoints`
   table is reconciled: LangGraph owns its checkpoint tables; our `conversations`/`messages`
   remain the durable conversation log and `ephemeral_agents` the sub-agent registry.
5. **Redis Streams** (not pub/sub) for the per-task event channel — pub/sub cannot replay
   from an offset, which breaks the "reconnect and resume from last event" requirement. A
   Postgres `task_events` table is the durable backstop and replay source of record.
6. **Supervisor safety rails**: per-conversation recursion limit, max-iteration budget, and a
   token/cost budget guard to prevent runaway multi-agent loops.
7. **Cost & token accounting** persisted per request (`usage` rolled into `tasks` + emitted as
   OTel metric `hivemind.llm.tokens.used`).
8. **Idempotency keys** on `POST /v1/chat/completions` and on task dispatch + checkpoint writes.
9. **RFC 7807 problem+json** error bodies from a typed exception hierarchy carrying request context.
10. **Model-aware Anthropic adapter**: Opus 4.7/4.8 reject `temperature`/`top_p`/`top_k`/`budget_tokens`;
    the adapter strips unsupported sampling params per model and uses adaptive thinking + `effort`.

---

## 2. Build order (each step independently runnable/testable)

- [ ] **S1 — Scaffold**: `pyproject.toml` (uv), package layout, `config.py` (pydantic-settings),
      `.env.example`, `Makefile`, ruff/mypy/pytest config.
- [ ] **S2 — Cross-cutting**: `core/context.py` (contextvars request context),
      `observability/logging.py` (structlog JSON), `observability/tracing.py` (OTel),
      `core/errors.py` (exception hierarchy + RFC7807).
- [ ] **S3 — Persistence**: SQLAlchemy async models, repositories (CQS split), Alembic baseline.
- [ ] **S4 — LLM layer**: `LLMProvider` interface, factory, Anthropic + OpenAI-compatible
      (OpenAI/Azure/vLLM) + Ollama adapters; streaming + tool-use normalization to a common shape.
- [ ] **S5 — Tools**: `BaseTool`, `ToolRegistry`, JSON-schema validation, OTel spans;
      SQL tool, code-exec tool (+ `Sandbox` abstraction & Docker backend), web-search (stub),
      sub-agent spawner.
- [ ] **S6 — Skills**: model, markdown loader, `SkillRegistry`, prompt injection.
- [ ] **S7 — Agents**: immutable `Agent` model, `AgentFactory`, `AgentRegistry`, `AgentRepository`,
      startup hydration, built-in SQL specialist agent.
- [ ] **S8 — Graph**: LangGraph builder, supervisor node (LLM router: single/sequential/parallel/
      conditional), agent nodes, Postgres checkpointer wiring, safety rails.
- [ ] **S9 — Services**: `ConversationService`, `ArtifactStore`, `Scheduler` (TTL GC).
- [ ] **S10 — API**: pydantic schemas, auth/context/logging/rate-limit middleware, routers
      (chat, agents, tools, skills, tasks, conversations), `main.py` app factory + lifespan,
      graceful shutdown.
- [ ] **S11 — Workers + async mode**: RabbitMQ consumer, `events.py` task-event buffer
      (Redis Streams + `task_events`), SSE replay endpoint, mode-selection threshold.
- [ ] **S12 — Local infra**: `Dockerfile`, `docker-compose.yml` (api, worker, postgres,
      rabbitmq, redis, otel-collector, jaeger, prometheus, grafana), provisioning configs.
- [ ] **S13 — Tests**: unit (tools, factory, providers, supervisor, context) + integration
      (API via httpx, RabbitMQ dispatch + SSE replay, graph w/ mocked LLM); ≥80% on core/ & api/.
- [ ] **S14 — Helm**: `charts/hivemind` (api/worker Deployments + HPA/KEDA, cleanup CronJob,
      ConfigMap, Secret/ExternalSecret, ServiceMonitor, Ingress+TLS, PVC, values + values-local).

The **vertical slice** (this pass) targets S1–S12: a system that boots via
`make up`, authenticates, routes a chat message through the supervisor to the SQL
specialist agent, executes the SQL tool, streams SSE events, and persists everything —
plus the async/queue path. S13/S14 are scaffolded and filled iteratively.

---

## 3. Acceptance checks for the slice

1. `make up` brings up the full compose stack; `make migrate` applies schema.
2. `POST /v1/chat/completions` with a Bearer token returns an SSE stream of typed events
   (`node_start`, `tool_call`, `tool_result`, `message`, `done`).
3. `POST /v1/chat/completions` with `stream:false` returns a `task_id`; the worker executes
   it and `GET /v1/tasks/{id}/stream` replays buffered events and resumes live.
4. Built-in SQL specialist agent introspects schema and runs a parameterized read query
   through the least-privilege role, returning structured JSON.
5. Crash/restart: a conversation resumes from the LangGraph Postgres checkpoint; ephemeral
   sub-agents are restored.
6. Traces visible in Jaeger; metrics in Prometheus/Grafana; logs are structured JSON on stdout.
