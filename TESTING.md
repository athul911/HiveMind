# HiveMind — Real End-to-End Test Flow

A manual, phased walkthrough that exercises the **running system** with a real LLM — not the
unit tests. Each step lists the command and what you should see. Assumes Docker is running.

> Conventions: set `export BASE=http://localhost:8000`. Most phases use `AUTH_DISABLED=true`
> (compose default) so no token is needed; Phase 10 covers auth/RBAC explicitly.

---

## Phase 0 — Configure a real LLM and launch

Edit `.env` (copy from `.env.example` if you haven't):

```bash
# Anthropic
LLM_DEFAULT_PROVIDER=anthropic
LLM_DEFAULT_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=sk-ant-...

# …or OpenAI
# LLM_DEFAULT_PROVIDER=openai
# LLM_DEFAULT_MODEL=gpt-4o
# OPENAI_API_KEY=sk-...

AUTH_DISABLED=true          # keep simple for phases 1–9
SANDBOX_BACKEND=subprocess  # runs code-exec inside the container; no docker socket needed
```

```bash
make up         # builds, runs migrations automatically, starts api + worker + infra
make logs       # watch until you see api.started / worker.started (Ctrl-C to stop tailing)
```

A real cloud LLM is recommended — tool-calling (SQL/code) is reliable on Claude/GPT-4o.
Small local Ollama models often won't call tools dependably.

---

## Phase 1 — Smoke test

```bash
export BASE=http://localhost:8000
curl -s $BASE/health                       # {"status":"ok"}
curl -s $BASE/readyz | jq                   # {"status":"ready","checks":{postgres/redis/rabbitmq:"ok"}}
```

If `/readyz` is 503, one dependency is down — the body says which. Open the Swagger UI at
$BASE/docs.

---

## Phase 2 — Catalog (what's registered)

```bash
curl -s $BASE/v1/tools  | jq '.[].name'     # sql_query, code_exec, web_search, spawn_subagent
curl -s $BASE/v1/skills | jq '.[].name'     # postgres-optimization
curl -s $BASE/v1/agents | jq '.[] | {id,name}'   # sql-specialist provisioned at startup
```

Grab the SQL agent id for later:

```bash
SQL_AGENT=$(curl -s $BASE/v1/agents | jq -r '.[] | select(.name=="sql-specialist") | .id')
echo $SQL_AGENT
```

---

## Phase 3 — Load demo data

```bash
make seed-data
# → prints: customers=60 products=12 orders=800 completed_revenue=<n>
```

This creates `customers`, `products`, `orders` and grants the read-only SQL role access.

---

## Phase 4 — SQL specialist over SSE (schema introspection + safe query)

Ask a data question, pinning the SQL agent so routing is deterministic. `-N` keeps the SSE
stream open:

```bash
curl -N $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"agent_id\":\"$SQL_AGENT\",
       \"messages\":[{\"role\":\"user\",
         \"content\":\"What are the top 5 products by completed revenue, and which region buys the most?\"}]}"
```

**Expect**, as SSE events: `node_start` (supervisor) → `routing_decision` → `node_start`
(agent) → one or more `tool_call` (`sql_query`, possibly with `introspect:true` first) →
`tool_result` (rows as JSON) → `text_delta` chunks → `message` → `done`. The final answer
should name real products/regions from the seed data.

Try a few more:
- "Show monthly completed revenue for the last 6 months."
- "What's the average order value by customer segment?"
- "Which customers have the highest lifetime spend?"

---

## Phase 5 — Code execution tool (artifacts)

```bash
curl -N $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"agent_id\":\"$SQL_AGENT\",
       \"messages\":[{\"role\":\"user\",
         \"content\":\"Use Python to compute the mean and standard deviation of [10,12,23,23,16,23,21,16] and write the result to a file results.txt.\"}]}"
```

**Expect** a `tool_call` for `code_exec` and a `tool_result` whose `content.artifacts`
contains an `artifact_ref` pointing at `results.txt`, plus a `download_url`. Confirm the file
landed on the volume:

```bash
docker compose exec api ls -R /data/artifacts | head -30
```

The `download_url` is **owner-authenticated** — fetch it with your bearer token (the same
user who created the conversation). It is not anonymously shareable:

```bash
curl -L "<download_url from the artifact_ref>" -H "Authorization: Bearer $TOKEN" -o results.txt
```

---

## Phase 6 — Conversation continuity

Start a conversation and reuse its id for a follow-up that depends on prior context:

```bash
CID=$(uuidgen)
curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$CID\",\"agent_id\":\"$SQL_AGENT\",
       \"messages\":[{\"role\":\"user\",\"content\":\"How many orders are in the APAC region?\"}]}"

curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$CID\",\"agent_id\":\"$SQL_AGENT\",
       \"messages\":[{\"role\":\"user\",\"content\":\"Now break that down by product category.\"}]}"

# Inspect the stored transcript:
curl -s $BASE/v1/conversations/$CID | jq '.messages[] | {role, content: .content[0:80]}'
```

**Expect** the second answer to understand "that" = APAC orders, and the transcript to show
the persisted user/assistant turns.

---

## Phase 7 — Create a custom agent + multi-agent routing

Create a second agent so the supervisor has a real routing choice:

```bash
curl -s -X POST $BASE/v1/agents -H "Content-Type: application/json" -d '{
  "name": "analyst",
  "description": "Explains business metrics and writes short narrative summaries. No DB access.",
  "system_prompt": "You are a concise business analyst. Explain findings in plain language.",
  "tool_names": [],
  "llm_config": {"provider": "anthropic", "model": "claude-opus-4-8", "max_tokens": 1024}
}' | jq '{id,name}'
```

Now ask WITHOUT pinning an agent — let the supervisor route:

```bash
# A pure data question should route to sql-specialist:
curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"How many completed orders are there?"}]}'

# An explanatory question should route to the analyst:
curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"In one paragraph, what does average order value tell a business?"}]}'

# A compound ask may route sequential/parallel/conditional:
curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Find the top region by revenue, then explain why that matters."}]}'
```

**Expect** the `routing_decision` event's `data.plan.mode` to vary
(`single`/`sequential`/`parallel`/`conditional`) and `data.plan.agents` to list the chosen
agent ids.

---

## Phase 8 — Ephemeral sub-agents

Create an orchestrator that can spawn sub-agents, then give it a task that benefits from one:

```bash
ORCH=$(curl -s -X POST $BASE/v1/agents -H "Content-Type: application/json" -d '{
  "name": "orchestrator",
  "description": "Breaks work into focused subtasks and delegates to specialist sub-agents.",
  "system_prompt": "When a subtask is self-contained, use the spawn_subagent tool to delegate it. Provide the sub-agent a clear system_prompt and task.",
  "tool_names": ["spawn_subagent"],
  "llm_config": {"provider":"anthropic","model":"claude-opus-4-8","max_tokens":2048}
}' | jq -r .id)

curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"agent_id\":\"$ORCH\",
       \"messages\":[{\"role\":\"user\",\"content\":\"Spawn a sub-agent to draft a 3-line product tagline for 'Aurora Headphones', then return it.\"}]}"
```

**Expect** a `tool_call` for `spawn_subagent` and a `tool_result` with the sub-agent's output.
Verify it was checkpointed:

```bash
docker compose exec postgres psql -U hivemind -d hivemind \
  -c "SELECT id, parent_conversation_id, checkpoint->>'result' AS result FROM ephemeral_agents ORDER BY created_at DESC LIMIT 5;"
```

You should see a row with a non-null `result` (the idempotent, checkpoint-backed restore
record).

---

## Phase 9 — Async / queue mode + reconnect-safe replay

Send with `stream:false` to get a `task_id` (the worker runs it):

```bash
RESP=$(curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -H "Idempotency-Key: e2e-test-1" \
  -d "{\"agent_id\":\"$SQL_AGENT\",\"stream\":false,
       \"messages\":[{\"role\":\"user\",\"content\":\"Give a full breakdown of revenue by region and category.\"}]}")
echo $RESP | jq
TASK=$(echo $RESP | jq -r .task_id)
```

Stream the task's events (resumable):

```bash
curl -N "$BASE/v1/tasks/$TASK/stream"            # full event stream from seq 0
curl -N "$BASE/v1/tasks/$TASK/stream?after=3"    # resume after seq 3 (reconnect-safe)
curl -s  "$BASE/v1/tasks/$TASK/status" | jq      # {status:"completed", result:{final:...}, usage:{...}}
```

**Expect** `status` to progress to `completed` and `result.final` to hold the answer.
Re-sending the same `Idempotency-Key` returns the **same** `task_id` without re-running.

---

## Phase 10 — Guardrails (each needs a small `.env` change + `make app-up`)

**SQL safety** (no env change): ask the SQL agent to mutate data — it must refuse at the tool
layer.

```bash
curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"agent_id\":\"$SQL_AGENT\",
       \"messages\":[{\"role\":\"user\",\"content\":\"Delete all rows from the orders table.\"}]}"
```
**Expect** a `tool_result` with `is_error` / an "Unsafe SQL Rejected" message — the read-only
role + parser guard block it.

**Per-conversation token budget:** set `SUPERVISOR_TOKEN_BUDGET=50` in `.env`, `make app-up`,
then send any chat. **Expect** an `error` event of type `BudgetExceededError`. Restore the
value afterward.

**Rate limiting:** set `RATE_LIMIT_PER_MINUTE=5`, `make app-up`, then:
```bash
for i in $(seq 1 8); do curl -s -o /dev/null -w "%{http_code}\n" $BASE/v1/tools; done
```
**Expect** the first 5 → `200`, then `429`.

**RBAC:** set `AUTH_DISABLED=false` and `RBAC_ENABLED=true`, `make app-up`. Agent management
now needs the admin scope:
```bash
TOKEN=$(make -s token)                       # plain token, no admin scope
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE/v1/agents \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"x","system_prompt":"p","llm_config":{"provider":"anthropic","model":"claude-opus-4-8"}}'
# → 403
```
Mint a token with the scope and retry (the token must carry `scope: "hivemind:admin"` —
extend `scripts/mint_token.py` or use your IdP). Reset `.env` when done.

---

## Phase 11 — Observability

- **Traces:** open Jaeger at http://localhost:16686, service `hivemind`. You'll see spans for
  `graph.run`, `supervisor.route`, `agent`, `tool.invoke`, and `llm.call`.
- **Metrics:** http://localhost:9090 (Prometheus) — query `hivemind_tool_calls_total`,
  `hivemind_llm_tokens_used_total`, `hivemind_workflow_duration_seconds_count`.
- **Dashboard:** http://localhost:3000 (Grafana, anonymous admin) → HiveMind dashboard.
- **Logs:** `make logs` — structured JSON lines carrying `request_id`/`conversation_id`/
  `agent_id`; confirm no secrets appear (they're redacted).

---

## Phase 12 — Persistence & crash recovery

```bash
make restart                                  # bounce api + worker only
curl -s $BASE/v1/conversations/$CID | jq '.messages | length'   # history survived
```

**Expect** the conversation transcript from Phase 6 to still be there. Then prove the DB
survives teardown:

```bash
make down                                     # stops containers, KEEPS data
make up
curl -s $BASE/v1/conversations/$CID | jq '.status'   # still present
```

**Crash-recovery resume (queue mode).** Start a long queue task, hard-kill the worker
mid-run, then bring it back — RabbitMQ redelivers the unacked task and the worker **resumes**
from the checkpoint instead of restarting:

```bash
TASK=$(curl -s $BASE/v1/chat/completions -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stream":false,"messages":[{"role":"user","content":"Run a multi-step analysis over the orders table."}]}' \
  | jq -r .task_id)
docker compose kill -s SIGKILL worker          # hard crash mid-task (no graceful ack)
docker compose up -d worker                     # comes back; RabbitMQ redelivers the task
# tail the same task — it continues to a terminal event, no duplicate restart from scratch
curl -sN "$BASE/v1/tasks/$TASK/stream" -H "Authorization: Bearer $TOKEN" | grep -m1 '^event: done'
curl -s "$BASE/v1/tasks/$TASK/status" -H "Authorization: Bearer $TOKEN" | jq '.status'  # completed
```

(Look for `conversation.resume` in the worker logs. Event `seq` continues from where it left
off; an already-`completed`/`cancelled` task that gets redelivered is dropped idempotently.)

---

## Phase 13 — Cleanup & teardown

- **TTL GC:** the scheduler GCs expired conversations, ephemeral agents, and artifacts on its
  interval (`CLEANUP_INTERVAL_SECONDS`). To force a conversation's cleanup now:
  ```bash
  curl -s -X DELETE $BASE/v1/conversations/$CID   # ends it + drops its ephemeral agents
  ```
- **Stop, keep data:** `make down`
- **Full wipe:** `make clean` (destroys the database volume — next `make up` re-seeds the
  read-only role + the tiny `demo_sales` table; re-run `make seed-data` for the rich set).

---

## Quick checklist

- [ ] `/health` + `/readyz` green
- [ ] SQL agent introspects schema and answers data questions (Phase 4)
- [ ] Code-exec produces an artifact reference (Phase 5)
- [ ] Conversation context carries across turns (Phase 6)
- [ ] Supervisor routes between agents; modes vary (Phase 7)
- [ ] Sub-agent spawned + checkpointed (Phase 8)
- [ ] Queue mode returns a task_id; stream replays; status completes (Phase 9)
- [ ] Unsafe SQL rejected; budget/rate/RBAC guards fire (Phase 10)
- [ ] Traces/metrics/logs visible (Phase 11)
- [ ] Data survives restart and `make down`/`up` (Phase 12)
