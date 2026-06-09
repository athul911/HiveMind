"""Supervisor routing.

The supervisor reasons over the registered agents' descriptions and the conversation to
choose a route: a single agent, a sequential pipeline, or a parallel fan-out. It uses the
LLM in JSON mode and falls back to a safe default (the first/SQL agent) if parsing fails or
no agents match.
"""

from __future__ import annotations

import json

from hivemind.core.graph.deps import GraphDeps
from hivemind.core.llm.base import LLMConfig, LLMRequest, Message
from hivemind.observability.logging import get_logger
from hivemind.observability.tracing import span

logger = get_logger("hivemind.supervisor")

_ROUTER_SYSTEM = """\
You are the routing supervisor for a multi-agent system. Given the available agents and the
user's latest message, decide which agent(s) should handle it.

Respond with ONLY a JSON object, no prose:
{
  "mode": "single" | "sequential" | "parallel" | "conditional",
  "agents": ["<agent_id>", ...],
  "condition": "<only for conditional mode: the check on the first agent's result>",
  "reasoning": "<one sentence>"
}

- "single": one agent answers.
- "sequential": agents run in order, each building on the previous output.
- "parallel": agents run independently and their outputs are aggregated.
- "conditional": the first agent runs; if "condition" holds for its result, the remaining
  agents run as a follow-up. Use this when a second agent is only needed in some cases.
Choose the fewest agents necessary. Use the exact agent_id values provided.
"""


async def decide_route(deps: GraphDeps, user_message: str) -> dict:
    table = deps.agents.routing_table()
    if not table:
        return {"mode": "single", "agents": [], "reasoning": "no agents registered"}
    if len(table) == 1:
        return {"mode": "single", "agents": [table[0]["agent_id"]], "reasoning": "only one agent"}

    catalog = json.dumps(table, indent=2)
    config = LLMConfig(
        provider=deps.settings.llm_default_provider,
        model=deps.settings.llm_default_model,
        max_tokens=512,
    )
    provider = deps.llm_factory.create(config)
    request = LLMRequest(
        config=config,
        system=_ROUTER_SYSTEM,
        messages=[
            Message(role="user", content=f"Available agents:\n{catalog}\n\nUser: {user_message}")
        ],
    )
    with span("supervisor.route"):
        try:
            response = await provider.complete(request)
            plan = _parse_plan(response.text, table)
        except Exception as exc:
            logger.warning("supervisor.route_failed", error=str(exc))
            plan = {"mode": "single", "agents": [table[0]["agent_id"]], "reasoning": "fallback"}
    valid_ids = {row["agent_id"] for row in table}
    plan["agents"] = [a for a in plan.get("agents", []) if a in valid_ids]
    if not plan["agents"]:
        plan = {"mode": "single", "agents": [table[0]["agent_id"]], "reasoning": "fallback"}
    logger.info("supervisor.routed", mode=plan["mode"], agents=plan["agents"])
    return plan


def _parse_plan(text: str, table: list[dict]) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in router response")
    plan = json.loads(text[start : end + 1])
    if "mode" not in plan or "agents" not in plan:
        raise ValueError("router response missing required keys")
    return plan
