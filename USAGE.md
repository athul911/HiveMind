# HiveMind — Usage Guide

How to run HiveMind, authenticate, talk to agents, create your own, add skills, and operate
it. For architecture see [DESIGN.md](DESIGN.md); for the build plan see [PLAN.md](PLAN.md).

---

## 1. Run it

### Option A — full local stack (recommended)

```bash
cp .env.example .env
make up          # api, worker, postgres, rabbitmq, redis, otel, jaeger, prometheus, grafana
                 # schema migrations run automatically (one-shot `migrate` service) before
                 # the app starts — no manual `make migrate` needed on first boot.
```

The compose stack defaults to **Ollama** as the LLM provider (zero API keys/cost) and the
**subprocess** sandbox. Run Ollama on your host (`ollama serve` + `ollama pull llama3.1`), or
switch providers (see §7). `AUTH_DISABLED=true` is set in compose so you can call the API
without a token while developing.

**Lifecycle / data persistence:**

| Command | Effect |
| --- | --- |
| `make up` | Start everything; auto-applies migrations first |
| `make down` | Stop & remove containers — **keeps the database volume** (data persists) |
| `make clean` | Stop and **delete all volumes** (full wipe — destroys the DB) |
| `make db-up` / `make db-down` | Start / stop just Postgres + RabbitMQ + Redis |
| `make app-up` / `make app-down` | Start / stop just the api + worker (data keeps running) |
| `make restart` | Restart the app containers only |
| `make migrate` | Apply migrations manually (rarely needed — `up` does it) |

> Migrations are idempotent (`alembic upgrade head`), so they're a no-op once the schema is
> current. `make down` no longer wipes data — use `make clean` for an intentional reset.

| Service | URL |
| --- | --- |
| API + Swagger UI | http://localhost:8000/docs |
| Jaeger (traces) | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana (dashboard) | http://localhost:3000 |
| RabbitMQ UI | http://localhost:15672 (hivemind/hivemind) |

Tail logs: `make logs`. Tear down: `make down`.

### Option B — local Python (no containers)

```bash
make install                                   # uv venv + deps
# bring up just the backing services you need (postgres/redis/rabbitmq) however you like
uvicorn hivemind.main:app --reload             # API
python -m hivemind.workers.consumer            # worker (separate shell, for queue mode)
```

---

## 2. Authenticate

Every endpoint except `/health*`, `/readyz`, `/metrics`, and the docs requires a
`Bearer` JWT.

- **Local / dev:** with `AUTH_DISABLED=true` any value works — send `Authorization: Bearer x`.
- **Real tokens (HS256):** mint one signed with `JWT_SECRET`:

  ```bash
  TOKEN=$(make -s token)            # or: python scripts/mint_token.py <user-id>
  curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/tools
  ```

- **OAuth2 / JWKS (RS256):** set `OAUTH2_JWKS_URL` to your IdP's JWKS endpoint; tokens are
  then verified against it by `kid`. The `sub` claim becomes the request's `user_id`.

> All examples below assume `export TOKEN=...` (use `x` when auth is disabled).

---

## 3. Chat — the main entrypoint

`POST /v1/chat/completions` is OpenAI-shaped. The supervisor routes your message to the
right agent(s) automatically; pin one with `agent_id` to skip routing.

### Interactive (SSE stream)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "How many sales rows are in the EMEA region?"}],
    "stream": true
  }'
```

You get a Server-Sent Events stream. Each line is `event: <type>` + `data: <json>`, where
the JSON is `{"type": ..., "data": {...}}`. Event types:

| Event | Meaning |
| --- | --- |
| `conversation` | First event — carries the `conversation_id` (and `request_id`) to reuse |
| `node_start` / `node_end` | A graph node (supervisor or agent) started/finished |
| `routing_decision` | The supervisor's chosen plan (`single`/`sequential`/`parallel`/`conditional`) |
| `text_delta` | An incremental chunk of the assistant's answer |
| `tool_call` / `tool_result` | A tool was invoked and returned |
| `condition_check` / `condition_result` | Conditional routing evaluated a branch |
| `usage` | Token usage for a model call |
| `agent_finished` | An agent turn ended; `data.stop_reason` explains why (`end_turn`/`length`/…) and whether it called a tool |
| `message` | The final assistant message (also persisted) |
| `done` | Terminal event; `data.final` is the full answer |
| `error` | Something failed; `data.detail` explains |

### Continue a conversation

Pass the same `conversation_id` back to keep history (history is persisted and, past
`CONVERSATION_HISTORY_LIMIT` turns, automatically compacted):

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"conversation_id": "11111111-1111-1111-1111-111111111111",
       "messages": [{"role": "user", "content": "Now break that down by product."}]}'
```

**Finding the conversation id on a first message:** if you omit `conversation_id`, the server
generates one and tells you immediately:

- **SSE mode:** the **first event** is `conversation`, carrying the id, and it's also returned
  as the `x-conversation-id` response header:
  ```
  event: conversation
  data: {"type":"conversation","data":{"conversation_id":"<uuid>","request_id":"<uuid>"}}
  ```
  Capture that id and pass it back as `conversation_id` on the next turn.
- **Queue mode:** the JSON response includes `conversation_id` (and `task_id`).

### Long-running (queue mode)

Set `stream: false` (or let a multi-agent workflow exceed `WORKFLOW_ASYNC_THRESHOLD_STEPS`)
and the call returns immediately with a `task_id`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Idempotency-Key: my-unique-key-123" \
  -d '{"messages": [{"role": "user", "content": "Run a full analysis."}], "stream": false}'
# → {"task_id": "...", "conversation_id": "...", "status": "queued",
#    "stream_url": "/v1/tasks/{id}/stream", "status_url": "/v1/tasks/{id}/status"}
```

`Idempotency-Key` makes a retried dispatch return the same task instead of double-running.

Then stream the task's events (reconnect-safe — resume with `?after=<seq>` or the
`Last-Event-ID` header):

```bash
curl -N "http://localhost:8000/v1/tasks/<task_id>/stream" -H "Authorization: Bearer $TOKEN"
# resume after a drop:
curl -N "http://localhost:8000/v1/tasks/<task_id>/stream?after=12" -H "Authorization: Bearer $TOKEN"
```

Or poll status:

```bash
curl "http://localhost:8000/v1/tasks/<task_id>/status" -H "Authorization: Bearer $TOKEN"
# → {"task_id": ..., "status": "completed", "result": {"final": "..."}, "usage": {...}}
```

### Reconnecting / recovering a task (returning user)

If a client loses the `task_id` (closed the app, switched device), it recovers from the
**`conversation_id`** it kept — list that conversation's tasks, newest first, and reconnect to
the most recent one:

```bash
curl "http://localhost:8000/v1/conversations/<conversation_id>/tasks" -H "Authorization: Bearer $TOKEN"
# → [{"task_id":"...","status":"running","created_at":"...","stream_url":"/v1/tasks/.../stream", ...}, ...]
# take element 0 (newest) and reconnect:
curl -N "http://localhost:8000/v1/tasks/<task_id>/stream" -H "Authorization: Bearer $TOKEN"
```

This is **status-independent on purpose**: you don't need to catch the task while it's
"running". Because every event is durably logged to `task_events`, reconnecting to a task that
already **completed** replays its full event history (then ends) — so the user sees the result
either way. (If you didn't keep the `conversation_id` either, that's the one id to persist
client-side — it's returned in the queue response and as the first SSE `conversation` event.)

### Ownership & one-turn-at-a-time

- **Private by default.** A conversation belongs to the user who created it (from the JWT
  `sub`). Another user gets `403` on `GET`/`DELETE`/tasks/stream for it. (An operator with the
  `admin_scope` may access any conversation when `RBAC_ENABLED`.)
- **A conversation is locked while a turn runs.** If you send a new message while a task is in
  progress, you get `409 Conflict` ("a turn is already in progress"). Wait for it to finish, or
  cancel it:
  ```bash
  curl -X POST "http://localhost:8000/v1/tasks/<task_id>/cancel" -H "Authorization: Bearer $TOKEN"
  # → {"task_id":"...","status":"cancelled", ...}   # unlocks the conversation
  ```
  Cancellation is cooperative — a worker mid-run finishes its current step, but the task is
  marked cancelled and the conversation unlocks immediately so you can send the next query. A
  crashed holder's lock is auto-released by the cleanup scheduler after
  `CONVERSATION_LOCK_STALE_SECONDS`.

### Conversation lifecycle

```bash
curl http://localhost:8000/v1/conversations/<id> -H "Authorization: Bearer $TOKEN"   # history
curl -X DELETE http://localhost:8000/v1/conversations/<id> -H "Authorization: Bearer $TOKEN"  # end + cleanup
```

> **OpenAI SDK note:** the request shape is OpenAI-compatible, but the SSE stream emits
> HiveMind's typed events, **not** OpenAI `chat.completion.chunk` frames — so the stock
> `openai` SDK won't parse the stream. Consume the SSE events directly (any SSE client).

---

## 4. Built-ins you get out of the box

- **SQL specialist agent** (`sql-specialist`) is provisioned at startup — ask data questions
  in plain English and it introspects the schema and runs safe read-only queries. The compose
  stack seeds a `demo_sales` table to try it against.
- **Tools:** `sql_query` (read-only, parser-guarded), `code_exec` (sandboxed Python),
  `web_search` (stub backend), `spawn_subagent` (dynamic ephemeral specialists).
- **Skill:** `postgres-optimization`.

Browse them:

```bash
curl http://localhost:8000/v1/agents  -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/v1/tools   -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/v1/skills  -H "Authorization: Bearer $TOKEN"
```

---

## 5. Create your own agent

Agents are **immutable** (a change = a new version). Bind any subset of registered tools and
skills, and pin an LLM config:

```bash
curl -X POST http://localhost:8000/v1/agents \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "name": "researcher",
    "description": "Researches topics and summarizes findings with sources.",
    "system_prompt": "You are a meticulous research assistant. Cite sources.",
    "tool_names": ["web_search", "code_exec"],
    "skill_names": [],
    "llm_config": {"provider": "anthropic", "model": "claude-opus-4-8", "max_tokens": 4096}
  }'
```

Creating with an unknown tool/skill name is rejected (`400`). Once created, the agent is in
the registry immediately and the supervisor can route to it. Pin it explicitly with
`"agent_id": "<id>"` in a chat request. Decommission with `DELETE /v1/agents/{id}`.

> If `RBAC_ENABLED=true`, `POST`/`DELETE /v1/agents` require the `ADMIN_SCOPE` (e.g.
> `hivemind:admin`) in your token's `scope`/`scopes`/`permissions`/`roles` claim.

LLM config fields: `provider` (`anthropic`|`openai`|`azure`|`vllm`|`ollama`), `model`,
`max_tokens`, optional `temperature`/`top_p` (ignored for Opus 4.7/4.8 which use adaptive
thinking), and `extra` (e.g. `{"effort": "high"}` for Anthropic).

---

## 6. Add skills (Markdown)

A skill is a Markdown file with YAML frontmatter, dropped into the `skills/` directory
(`SKILLS_DIR`). It's loaded at startup and bound to agents by name; the body is injected into
the bound agent's system prompt (with progressive disclosure).

```markdown
---
name: incident-response
description: How to triage and write up production incidents.
version: 1
---

# Incident Response Playbook
1. Establish impact and severity before anything else.
2. ...
```

Save as `skills/incident-response.md`, restart the API (`make seed` or `docker compose
restart api worker`), then reference it: `"skill_names": ["incident-response"]` when creating
an agent. List loaded skills at `GET /v1/skills`.

---

## 7. Choose an LLM provider

Set the default in `.env` (per-agent config can override):

```bash
# Local, free:
LLM_DEFAULT_PROVIDER=ollama
LLM_DEFAULT_MODEL=llama3.1

# Anthropic:
LLM_DEFAULT_PROVIDER=anthropic
LLM_DEFAULT_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI:
LLM_DEFAULT_PROVIDER=openai
LLM_DEFAULT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

Azure (`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_API_KEY`, model = deployment name) and vLLM
(`VLLM_BASE_URL`) are also supported. All providers are wrapped with retry + a circuit
breaker, so a flaky upstream degrades gracefully.

---

## 8. Operate it

- **Health/readiness:** `GET /health` (liveness), `GET /readyz` (checks Postgres + Redis +
  RabbitMQ; `503` until all are reachable).
- **Metrics:** `GET /metrics` (Prometheus). Custom metrics: `hivemind_workflow_duration_seconds`,
  `hivemind_tool_calls_total`, `hivemind_llm_tokens_used_total`. A Grafana dashboard is
  pre-provisioned.
- **Traces:** spans for graph nodes, tool calls, LLM calls, and routing decisions show up in
  Jaeger.
- **Logs:** structured JSON to stdout, enriched with `request_id`/`conversation_id`/`user_id`/
  `agent_id`; secret-bearing fields are auto-redacted.
- **Cleanup:** a scheduler GCs expired conversations, ephemeral sub-agents, and their
  artifacts (also runnable as a one-shot for the K8s CronJob).

### Key configuration knobs (`.env`)

| Var | Purpose |
| --- | --- |
| `WORKFLOW_ASYNC_THRESHOLD_STEPS` | When to switch SSE → queue mode |
| `SUPERVISOR_TOKEN_BUDGET` | Cumulative per-conversation token ceiling |
| `SUBAGENT_MAX_DEPTH` | Max `spawn_subagent` recursion depth |
| `CONVERSATION_HISTORY_LIMIT` | Turns kept verbatim before compaction |
| `SANDBOX_BACKEND` | `docker` (prod) or `subprocess` (laptop/CI) |
| `SQL_TOOL_DATABASE_URL` | Least-privilege read-only DSN for the SQL tool |
| `RATE_LIMIT_PER_MINUTE` | Per-user request limit |
| `RBAC_ENABLED` / `ADMIN_SCOPE` | Gate agent management on a scope |
| `PROMPT_CACHE_ENABLED` | Anthropic prompt caching on stable prefixes |

See `.env.example` for the full list.

---

## 9. Deploy on Kubernetes

```bash
helm install hivemind charts/hivemind -f charts/hivemind/values.yaml
# local cluster (k3d/minikube): reduced replicas, no autoscaling, subprocess sandbox
helm install hivemind charts/hivemind -f charts/hivemind/values-local.yaml
```

The chart ships API + worker Deployments (HPA on CPU/RPS, KEDA on RabbitMQ queue depth), the
cleanup CronJob, ConfigMap + Secret/ExternalSecret, a TLS Ingress, an artifact PVC, and a
Prometheus ServiceMonitor. Supply secrets via `secrets.data`, an `existingSecret`, or
`externalSecret`.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `relation "agents" does not exist` | Schema not migrated. `make up` now auto-runs migrations; if you started services piecemeal, run `make migrate`. (Don't use `make clean` between runs — it wipes the DB; use `make down`.) |
| `401 application/problem+json` | Missing/invalid `Authorization: Bearer` header; set `AUTH_DISABLED=true` for local, or mint a token. |
| `400 Unsafe SQL Rejected` | The SQL tool only allows a single read-only `SELECT`/`WITH`. |
| Chat hangs with no tokens | Provider unreachable (e.g. Ollama not running) — check `make logs`; the circuit breaker will fast-fail after repeated failures. |
| `/readyz` returns 503 | One of Postgres/Redis/RabbitMQ is down; the body lists which. |
| `429` | Per-user rate limit hit (`RATE_LIMIT_PER_MINUTE`). |
| Code-exec tool errors about Docker | Set `SANDBOX_BACKEND=subprocess` if Docker isn't available. |
| Agent create `400 Unknown tools/skills` | Bind only names from `GET /v1/tools` and `GET /v1/skills`. |
