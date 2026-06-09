"""Condition evaluation for the supervisor's conditional routing mode.

Given a natural-language condition and an agent's output, a cheap LLM call decides whether
the condition holds (so the graph can branch on intermediate results). Returns the boolean
plus the tokens consumed (so it counts against the conversation budget).
"""

from __future__ import annotations

from hivemind.core.graph.deps import GraphDeps
from hivemind.core.llm.base import LLMConfig, LLMRequest, Message

_SYSTEM = (
    "You are a router. Decide whether the CONDITION is satisfied by the RESULT. "
    "Reply with exactly one word: YES or NO."
)


async def evaluate_condition(deps: GraphDeps, condition: str, result: str) -> tuple[bool, int]:
    config = LLMConfig(
        provider=deps.settings.llm_default_provider,
        model=deps.settings.llm_default_model,
        max_tokens=8,
    )
    provider = deps.llm_factory.create(config)
    request = LLMRequest(
        config=config,
        system=_SYSTEM,
        messages=[Message(role="user", content=f"CONDITION: {condition}\n\nRESULT:\n{result}")],
    )
    try:
        response = await provider.complete(request)
    except Exception:
        return False, 0
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return response.text.strip().upper().startswith("YES"), tokens
