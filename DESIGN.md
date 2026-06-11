# HiveMind — Design Document

Production-grade multi-agent orchestration on LangGraph. This doc covers architecture,
component contracts, data model, execution flow, and the security/observability posture.
Build sequencing lives in [PLAN.md](PLAN.md).

---

## 1. System overview

```
                    ┌────────────────────────────────────────────────────────┐
   Client ── JWT ──▶│  FastAPI (api/)                                          │
                    │  middleware: auth → context(contextvars) → log → rate    │
                    │  routers: chat / agents / tools / skills / tasks / convo │
                    └───────────────┬───────────────────────┬─────────────────┘
                          short/interactive          long-running / stream:false
                                    │                         │
                              SSE (in-proc)            dispatch task → RabbitMQ
                                    │                         │
                    ┌───────────────▼─────────────┐   ┌───────▼──────────────────┐
                    │  GraphRunner (core/graph)    │   │  Worker (workers/)        │
                    │  supervisor → agent nodes    │   │  consumes task, runs same │
                    │  tools, sub-agents           │   │  GraphRunner, buffers     │
                    │  LangGraph AsyncPostgresSaver│   │  events → Redis Streams + │
                    └───────┬──────────────┬───────┘   │  task_events (replay)     │
                            │              │           └───────────┬───────────────┘
                ┌───────────▼──┐   ┌───────▼────────┐              │ SSE replay
                │  Postgres    │   │  Tools          │   GET /v1/tasks/{id}/stream
                │  (asyncpg)   │   │  sql / code /   │
                │  conversations│   │  web / spawner │
                │  messages     │   │  ArtifactStore │
                │  agents/skills│   └────────────────┘
                │  tasks/events │
                │  ephemeral_*  │   Observability: structlog(JSON)+OTel → OTLP → Jaeger/Prom
                │  langgraph_*  │
                └──────────────┘
```

Everything I/O is `async`. No global mutable state — shared resources (engine,
registries, settings, broker) are created in the app/worker lifespan and handed out
via FastAPI `Depends` or explicit injection into the worker.

---

## 2. Core abstractions & contracts

### 2.1 Request context (`core/context.py`)
A frozen `RequestContext` (request_id, conversation_id, user_id, agent_id, task_id,
trace_id) stored in a `ContextVar`. Set by middleware (API) or on task pickup (worker).
`bind_context()` / `get_context()` / `clear_context()`. Structlog and OTel both read it,
so every log line and span automatically carries the full context. `contextvars` propagate
across `await` and `asyncio.Task` copies — no bleed between concurrent requests.

### 2.2 LLM provider (`core/llm/`)
```python
class LLMProvider(Protocol):
    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMStreamEvent]: ...
    async def complete(self, req: LLMRequest) -> LLMResponse: ...
```
`LLMRequest` normalizes messages, tool schemas, system prompt, and `LLMConfig`
(provider, model, temperature, max_tokens, extra). Each adapter maps to/from its SDK and
emits a common `LLMStreamEvent` union (`text_delta`, `thinking_delta`, `tool_call`,
`usage`, `done`). `LLMProviderFactory.create(config)` returns the right adapter; clients
are pooled per (provider, base_url, key).

- **Anthropic** — `anthropic.AsyncAnthropic`, `messages.stream()`. Model-aware: for
  `claude-opus-4-7`/`-4-8` it omits `temperature`/`top_p`/`top_k`/`budget_tokens` (they 400),
  uses `thinking={"type":"adaptive"}` + `output_config.effort`. Default model `claude-opus-4-8`.
- **OpenAI-compatible base** — `openai.AsyncOpenAI`; shared by **OpenAI**, **Azure**
  (`AsyncAzureOpenAI`, deployment-as-model), **vLLM** (`base_url` override).
- **Ollama** — native `/api/chat` HTTP via `httpx`, streaming NDJSON; tool-calls normalized.

All adapters: streaming-first, OTel span per call, emit `hivemind.llm.tokens.used`.

### 2.3 Tools (`core/tools/`)
```python
class BaseTool(ABC):
    name: str; description: str; input_schema: dict   # JSON Schema
    async def run(self, args: dict, ctx: RequestContext) -> ToolResult: ...
```
`ToolRegistry` holds tools by unique name, validates `args` against `input_schema`
(jsonschema) before dispatch, wraps every call in an OTel span + structured log, and
increments `hivemind.tool.calls.total`. Agents bind a **subset** of registered tools.
Plugin pattern: tools self-register via `@register_tool` / entry points.

`ToolResult` is either inline JSON or an **artifact reference**
(`{"type":"artifact_ref","path":...,"size_bytes":...,"mime_type":...}`) — large outputs go
to the artifact store and only the ref travels in the prompt.

Built-in tools:
- **`sql_query`** — least-privilege read-only role, `statement_timeout`, row cap, schema
  allow-list, single-statement/SELECT-only guard, parameterized. Schema introspection helper.
- **`code_exec`** — generates/runs Python in a `Sandbox` (Docker backend default). Artifacts
  written to mounted `conversation_id/task_id/code_exec/`; returns artifact refs.
- **`web_search`** — pluggable; stub returns deterministic shaped results, real backend behind
  an interface.
- **`spawn_subagent`** — instantiates an ephemeral sub-agent (see §4.4).

### 2.4 Sandbox (`core/tools/sandbox/`)
```python
class Sandbox(Protocol):
    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult: ...
```
- **DockerSandbox** (default): ephemeral container, `--network none`, `--read-only`,
  `--cap-drop ALL`, `--pids-limit`, `--memory`, `--cpus`, non-root user, tmpfs `/tmp`,
  artifact dir bind-mounted rw, hard wall-clock kill.
- **SubprocessSandbox** (fallback): `resource` RLIMITs (CPU, AS, NOFILE, NPROC), scrubbed env,
  cwd jailed to artifact dir, timeout. Weaker isolation — flagged in logs; gated by config.
Selected by `SANDBOX_BACKEND` (`docker` | `subprocess`).

### 2.5 Skills (`core/skills/`)
A skill is a markdown file with YAML frontmatter:
```markdown
---
name: postgres-optimization
description: When to use EXPLAIN/ANALYZE and index strategies for slow queries.
version: 1
---
<full instructional body…>
```
`SkillLoader` parses frontmatter + body; `SkillRegistry` indexes by name. Agents bind skill
names at creation. **Progressive disclosure**: at agent-load the system prompt gets the
skills' `name`+`description` (cheap), and the full body is appended for bound skills (or
lazily, behind a `read_skill` capability for large sets). Stored in the `skills` table and/or
loaded from a `skills/` directory on disk.

### 2.6 Agent (`core/agents/`)
Immutable value object:
```python
@dataclass(frozen=True)
class Agent:
    id: str; name: str; system_prompt: str
    tool_names: tuple[str, ...]; skill_names: tuple[str, ...]
    llm_config: LLMConfig; version: int; created_at: datetime
```
Mutation = new version (new row). `AgentFactory` composes the effective system prompt
(persona + bound skills) and resolves tools/skills from the registries. `AgentRegistry` is
the in-memory index; `AgentRepository` persists/loads. On API and worker startup all
persisted agents are **hydrated** into the registry; the **SQL specialist** is provisioned
if absent.

### 2.7 Graph (`core/graph/`)
LangGraph `StateGraph` over a typed `GraphState` (messages, route plan, scratch, budgets).
Nodes: `supervisor` (LLM router) → one or more `agent` nodes → optional `aggregator`.
The **supervisor** prompts the routing LLM with the registered agents' descriptions and the
conversation, returning a structured route: `single` | `sequential[]` | `parallel[]` |
`conditional`. Safety rails: LangGraph `recursion_limit`, a max-iteration counter, and a
token/cost budget in state that aborts with a typed error when exceeded.
Checkpointing: `AsyncPostgresSaver` (langgraph-checkpoint-postgres) snapshots state after
each node. Queue tasks key the checkpoint by `thread_id == task_id`, so a worker crash +
RabbitMQ redelivery **resumes** the interrupted run from its last checkpoint (the worker
continues the event `seq` from `tasks.last_event_seq`, and already-finished/cancelled tasks
are dropped idempotently). SSE runs in-process (`thread_id == conversation_id`) and is not
resumed. The token budget in state is re-seeded each turn, so it is enforced **per turn**,
not cumulatively across the conversation.

### 2.8 Execution modes
`GraphRunner.run()` yields typed events. Mode chosen by `ModeSelector`:
- **SSE**: API holds the connection, iterates `GraphRunner` in-process, serializes each event
  as SSE. Threshold: estimated steps ≤ `WORKFLOW_ASYNC_THRESHOLD_STEPS` **and** `stream != false`.
- **Queue**: API persists a `tasks` row (`queued`), publishes to RabbitMQ, returns `task_id`
  immediately. Worker runs the same `GraphRunner`, writing each event to **Redis Streams**
  (live fan-out) **and** `task_events` (durable replay). `GET /v1/tasks/{id}/stream` replays
  `task_events` from the client's last offset, then tails the Redis stream — reconnect-safe.

---

## 3. Data model (Postgres, async SQLAlchemy + Alembic)

| Table | Purpose / key columns |
| --- | --- |
| `agents` | immutable defs: `id, name, version, system_prompt, tool_names[], skill_names[], llm_config(jsonb), immutable(bool), created_at` |
| `skills` | `id, name, version, description, body, created_at` |
| `conversations` | `id(uuid), user_id, agent_id, status, created_at, updated_at, ttl_expires_at` |
| `messages` | `id, conversation_id, role, content, tool_calls(jsonb), tool_results(jsonb), created_at` |
| `ephemeral_agents` | `id, parent_conversation_id, definition(jsonb), checkpoint(jsonb), created_at, expires_at` |
| `tasks` | `task_id, conversation_id, status, usage(jsonb), last_event_seq, created_at, completed_at, idempotency_key` |
| `task_events` | `id, task_id, seq, event_type, payload(jsonb), created_at` (replay log; `(task_id, seq)` unique) |
| `langgraph_*` | created/owned by `AsyncPostgresSaver` (checkpoints, writes) |

Repositories encapsulate all access (Repository pattern); business logic never issues raw
SQL/ORM queries. Reads and writes are split into distinct repo methods (CQS). Checkpoint and
task writes are **idempotent** (upsert on natural keys) to survive retries.

---

## 4. Key flows

### 4.1 Chat (SSE)
1. middleware validates JWT, binds `RequestContext`.
2. `ConversationService` loads/creates the conversation + history.
3. `ModeSelector` → SSE. `GraphRunner` resumes graph state from checkpoint and streams.
4. supervisor routes → agent node calls LLM (streaming) → may call tools → results appended.
5. each event serialized as SSE; final assistant message persisted to `messages`.

### 4.2 Chat (queue)
As above through step 2; `ModeSelector` → queue. Create `tasks` row (idempotent on
`Idempotency-Key`), publish, return `{task_id}`. Worker consumes, runs `GraphRunner`,
buffers events. Client streams `/v1/tasks/{id}/stream`.

### 4.3 Supervisor routing
LLM sees `[{agent_id, name, description}]` + conversation → returns a route plan. Plans:
single dispatch; sequential pipeline (output feeds next); parallel fan-out + `aggregator`
merge; conditional branch keyed on an intermediate result. Bounded by iteration/budget rails.

### 4.4 Ephemeral sub-agents
`spawn_subagent` builds an `Agent`-shaped definition, **checkpoints it to `ephemeral_agents`
immediately** (definition + initial checkpoint), attaches a node to the running graph, and
ties its lifecycle to the parent conversation. On resume after crash, sub-agents are restored
from `ephemeral_agents` and re-attached. The `Scheduler` GCs expired sub-agents + tasks
(TTL `EPHEMERAL_AGENT_TTL_SECONDS`).

---

## 5. Security posture

- **AuthN**: Bearer JWT. HS256 (shared secret) default; OAuth2 JWKS/RS256 when `OAUTH2_JWKS_URL`
  set (cached JWKS, `kid` lookup). Claims → `RequestContext.user_id`. Rate limiting keyed on
  `user_id` (Redis token bucket).
- **AuthZ (RBAC)**: agent-management routes require the configured `admin_scope` when
  `RBAC_ENABLED`; scopes read from `scope`/`scopes`/`permissions`/`roles` claims.
- **SQL tool**: dedicated read-only role, statement timeout, row cap, schema allow-list, and a
  **parser-based** guard (sqlglot) that requires exactly one read-only `SELECT`/`WITH` and
  rejects any mutating/DDL/DCL node — far more robust than keyword regex. Treats model SQL as
  hostile; always parameterized.
- **Code exec**: container isolation (no net, RO rootfs, dropped caps, resource limits, non-root,
  timeout). Subprocess fallback explicitly weaker, config-gated, with best-effort RLIMITs.
- **Sub-agents**: bounded recursion via `SUBAGENT_MAX_DEPTH`; spawning is idempotent and
  checkpoint-backed (deterministic id; a completed checkpoint is restored rather than re-run).
- **Artifacts**: namespaced `conversation_id/task_id/tool_name/`; traversal blocked (filename
  rejection + jail check); tools return refs, not raw bytes; GC'd with the conversation.
- **Errors**: RFC 7807 problem+json; internal details never leaked; every error carries
  `request_id`/`conversation_id`.
- **Secrets**: from env via pydantic-settings; a structlog redaction processor scrubs
  secret-bearing keys from every log line; K8s via Secret/ExternalSecret. CORS configurable.

---

## 6. Observability

- **Logging**: structlog → JSON to stdout; every entry has timestamp, level, logger,
  request_id, conversation_id, user_id, agent_id, event. Per-component levels via env.
- **Tracing/metrics**: OpenTelemetry SDK + auto-instrumentation (FastAPI, SQLAlchemy, HTTPX).
  Custom spans: graph node exec, tool invocation, LLM call, routing decision. Metrics:
  `hivemind.workflow.duration`, `hivemind.tool.calls.total`, `hivemind.llm.tokens.used`.
  OTLP export to the collector → Jaeger (traces) + Prometheus (metrics) → Grafana dashboard.

---

## 7. Design principles applied

Dependency inversion (Protocol-based providers/tools/repos), Repository pattern, CQS,
fail-fast typed errors with context, idempotent task/checkpoint writes, graceful shutdown
(drain in-flight on SIGTERM; worker finishes current task), no global mutable state (DI),
async throughout.

---

## 8. Spec deltas (what changed and why)

See PLAN.md §1 for the list. Headlines: container-isolated code exec (subprocess is unsafe),
parser-based least-privilege SQL, first-class Skills subsystem, official LangGraph Postgres
checkpointer as state source of truth, Redis **Streams** for replayable task events, supervisor
budget rails, cost accounting, idempotency keys, RFC7807 errors, model-aware Anthropic adapter.

**Hardening pass (post-slice):** conditional routing (single/sequential/parallel/conditional);
**per-conversation** cumulative token budget enforced in the graph; idempotent,
checkpoint-backed sub-agent restore + depth limit; full readiness checks (PG/RabbitMQ/Redis);
Anthropic prompt caching on stable prefixes; conversation compaction (windowing + summary);
artifact GC; LLM retry + circuit breaker; RBAC scope on agent management; log secret-redaction;
CORS.
