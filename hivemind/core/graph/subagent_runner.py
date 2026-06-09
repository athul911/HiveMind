"""Concrete :class:`SubAgentRunner` used by the spawn_subagent tool.

Lifecycle, made crash-safe and idempotent:

1. The ephemeral agent's id is **deterministic** — derived from (conversation, system prompt,
   task) — so the same logical sub-agent always maps to the same row.
2. On spawn we look it up. If a **completed checkpoint** already exists (e.g. the parent task
   was redelivered after a crash by RabbitMQ's at-least-once delivery), we **restore** the
   prior result instead of re-running expensive work — this is the checkpoint-backed resume.
3. Otherwise we checkpoint the definition, run the sub-agent (under a depth-incremented
   context so nested spawns are bounded), and re-checkpoint the result.

Depth is capped by ``SUBAGENT_MAX_DEPTH`` to prevent unbounded recursive spawning.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from hivemind.core.context import RequestContext, bind_context, reset_context
from hivemind.core.errors import ValidationError
from hivemind.core.graph.agent_runtime import run_agent_turn
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.llm.base import LLMConfig, Message
from hivemind.db.repository import EphemeralAgentRepository
from hivemind.db.session import Database
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.subagent")


def _deterministic_id(conversation_id: str, system_prompt: str, task: str) -> str:
    digest = hashlib.sha256(f"{conversation_id}\x00{system_prompt}\x00{task}".encode()).hexdigest()
    return f"eph_{digest[:32]}"


class SubAgentRunnerImpl:
    def __init__(self, deps: GraphDeps, db: Database) -> None:
        self._deps = deps
        self._db = db

    async def run_subagent(self, definition: dict, task: str, ctx: RequestContext) -> dict:
        settings = self._deps.settings
        if ctx.subagent_depth >= settings.subagent_max_depth:
            raise ValidationError(
                "Maximum sub-agent spawn depth reached.",
                depth=ctx.subagent_depth,
                max_depth=settings.subagent_max_depth,
            )

        conversation_id = ctx.conversation_id or "sync"
        ephemeral_id = _deterministic_id(conversation_id, definition["system_prompt"], task)

        # Restore-on-resume: if a completed checkpoint exists, reuse it (idempotent).
        async with self._db.session() as session:
            existing = await EphemeralAgentRepository(session).get(ephemeral_id)
        cp = existing.checkpoint if existing is not None else None
        if cp and cp.get("result") is not None:
            logger.info("subagent.restored", ephemeral_id=ephemeral_id)
            return cp

        record = {**definition, "task": task}
        async with self._db.session() as session:
            await EphemeralAgentRepository(session).checkpoint(
                ephemeral_id, conversation_id, record, None, settings.ephemeral_agent_ttl_seconds
            )
        logger.info(
            "subagent.spawned",
            ephemeral_id=ephemeral_id,
            depth=ctx.subagent_depth + 1,
            tools=definition.get("tool_names", []),
        )

        llm_config = LLMConfig(
            provider=settings.llm_default_provider,
            model=definition.get("model") or settings.llm_default_model,
            max_tokens=2048,
        )
        tool_names = [t for t in definition.get("tool_names", []) if self._deps.tools.has(t)]

        def _noop_emit(_ev) -> None:
            return None

        # Run under a depth-incremented context so any nested spawn is bounded.
        child_ctx = replace(ctx, subagent_depth=ctx.subagent_depth + 1, agent_id=ephemeral_id)
        token = bind_context(child_ctx)
        try:
            final, _appended, usage = await run_agent_turn(
                deps=self._deps,
                system_prompt=definition["system_prompt"],
                tool_names=tool_names,
                llm_config=llm_config,
                conversation=[Message(role="user", content=task)],
                ctx=child_ctx,
                emit=_noop_emit,
                token_budget=settings.supervisor_token_budget,
            )
        finally:
            reset_context(token)

        result = {
            "ephemeral_id": ephemeral_id,
            "result": final,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        }
        async with self._db.session() as session:
            await EphemeralAgentRepository(session).checkpoint(
                ephemeral_id, conversation_id, record, result, settings.ephemeral_agent_ttl_seconds
            )
        return result
