# HiveMind

Production-grade multi-agent AI orchestration on **LangGraph**, with OpenAI-compatible
REST endpoints, SSE + RabbitMQ execution modes, PostgreSQL persistence and crash-recovery
checkpointing, container-isolated code execution, a per-agent skills system, and full
OpenTelemetry observability.

- **Usage:** [USAGE.md](USAGE.md) · **Plan:** [PLAN.md](PLAN.md) · **Design:** [DESIGN.md](DESIGN.md)

## Highlights

- **Supervisor routing** over registered agents: single / sequential / parallel dispatch.
- **Immutable agents** = persona + tool subset + skill subset + LLM config; hydrated from
  Postgres on startup. A built-in **SQL specialist** is provisioned automatically.
- **5 LLM providers**: Anthropic, OpenAI, Azure OpenAI, vLLM, Ollama (one common interface).
- **Tools**: least-privilege read-only SQL, container-sandboxed Python code execution
  (artifact references, not raw data), web search, and a dynamic sub-agent spawner.
- **Skills** as markdown + frontmatter, bound per-agent and injected into the system prompt
  with progressive disclosure.
- **Two execution modes**: SSE for interactive turns; RabbitMQ queue for long-running
  workflows with **reconnect-safe** event replay (Postgres log + Redis Streams).
- **Crash recovery** via the official LangGraph `AsyncPostgresSaver`; ephemeral sub-agents
  are checkpointed and restorable; a scheduler GCs expired records.
- **Observability**: structlog JSON logs + OTel traces/metrics → Jaeger + Prometheus +
  Grafana. RFC 7807 problem+json errors. Request context via `contextvars`.

## Quickstart (local)

```bash
cp .env.example .env
make up           # api, worker, postgres, rabbitmq, redis, otel, jaeger, prometheus, grafana
make migrate      # apply the schema

# mint a dev token and call the API
TOKEN=$(make -s token)
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"How many sales rows are in EMEA?"}]}'
```

For zero-cost local runs set `LLM_DEFAULT_PROVIDER=ollama` (default in compose) and run
Ollama on the host. For Anthropic/OpenAI set the relevant API key in `.env`.

| Service | URL |
| --- | --- |
| API (docs) | http://localhost:8000/docs |
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| RabbitMQ UI | http://localhost:15672 |

## API surface

```
POST   /v1/chat/completions        # SSE stream, or {task_id} when stream:false / long workflows
GET    /v1/conversations/{id}
DELETE /v1/conversations/{id}       # end + cleanup
POST   /v1/agents                  # create (immutable)
GET    /v1/agents | /v1/agents/{id}
DELETE /v1/agents/{id}
GET    /v1/tools  | /v1/tools/{name}
GET    /v1/skills | /v1/skills/{name}
GET    /v1/tasks/{id}/stream       # SSE replay + live tail (reconnect-safe)
GET    /v1/tasks/{id}/status
```

All non-health endpoints require a Bearer JWT (HS256 by default; OAuth2 JWKS/RS256 when
`OAUTH2_JWKS_URL` is set).

## Development

```bash
make install      # uv venv + editable install with dev extras
make test         # pytest + coverage (>=80% on core/ and api/)
make lint typecheck
```

## Deployment

Kubernetes via the Helm chart in [`charts/hivemind`](charts/hivemind) — API + worker
Deployments (HPA / KEDA autoscaling), a cleanup CronJob, ConfigMap/Secret, ServiceMonitor,
TLS Ingress, and an artifact PVC. See its [values.yaml](charts/hivemind/values.yaml) and
`values-local.yaml` for a reduced local-cluster profile.
