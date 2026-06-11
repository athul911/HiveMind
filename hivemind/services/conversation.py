"""Conversation lifecycle management.

Loads/creates conversations, reconstructs history, drives the :class:`GraphRunner`, and
persists the user and final assistant messages. Emitted events flow to the caller (the SSE
endpoint or the worker), which forwards them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from hivemind.core.graph import events
from hivemind.core.graph.runner import GraphRunner
from hivemind.core.llm.base import Message
from hivemind.db.repository import ConversationRepository, MessageRepository
from hivemind.db.session import Database
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.conversation")


class ConversationService:
    def __init__(
        self,
        db: Database,
        runner: GraphRunner,
        *,
        ttl_seconds: int,
        deps=None,  # GraphDeps; enables LLM-backed compaction. Optional for tests.
        history_limit: int = 40,
        compaction_enabled: bool = True,
    ) -> None:
        self._db = db
        self._runner = runner
        self._ttl = ttl_seconds
        self._deps = deps
        self._history_limit = history_limit
        self._compaction_enabled = compaction_enabled

    async def load_history(self, conversation_id: str) -> list[Message]:
        """Reconstruct the durable user/assistant turns for prompting, with compaction.

        The ``messages`` table is the durable conversation log of user and final-assistant
        turns. Intermediate tool-call exchanges within a run are not persisted here — they
        live in the LangGraph checkpoint (used for crash recovery/resume).

        To keep the prompt within the context window on long conversations, history beyond
        ``history_limit`` turns is **compacted**: the oldest turns are replaced by a single
        summary message (windowing-only if compaction is disabled or no LLM is available).
        """
        async with self._db.session() as session:
            rows = await MessageRepository(session).history(conversation_id)
        history = [Message(role=row.role, content=row.content) for row in rows]
        if len(history) <= self._history_limit:
            return history
        return await self._compact(history)

    async def _compact(self, history: list[Message]) -> list[Message]:
        keep = self._history_limit
        older, recent = history[:-keep], history[-keep:]
        summary = await self._summarize(older)
        logger.info("conversation.compacted", dropped=len(older), kept=len(recent))
        if not summary:
            return recent  # windowing fallback
        marker = Message(role="user", content=f"[Summary of earlier conversation]\n{summary}")
        return [marker, *recent]

    async def _summarize(self, messages: list[Message]) -> str:
        if not (self._compaction_enabled and self._deps is not None):
            return ""
        from hivemind.core.llm.base import LLMConfig, LLMRequest

        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        config = LLMConfig(
            provider=self._deps.settings.llm_default_provider,
            model=self._deps.settings.llm_default_model,
            max_tokens=512,
        )
        request = LLMRequest(
            config=config,
            system="Summarize this conversation excerpt in a few sentences, preserving "
            "facts, decisions, and open questions. Be concise.",
            messages=[Message(role="user", content=transcript)],
        )
        try:
            response = await self._deps.llm_factory.create(config).complete(request)
            return response.text.strip()
        except Exception as exc:
            logger.warning("conversation.summarize_failed", error=str(exc))
            return ""

    async def ensure_conversation(
        self, conversation_id: str, user_id: str, agent_id: str | None
    ) -> None:
        async with self._db.session() as session:
            await ConversationRepository(session).get_or_create(
                conversation_id, user_id, agent_id, self._ttl
            )

    async def stream(
        self,
        *,
        conversation_id: str,
        user_id: str,
        agent_id: str | None,
        user_message: str,
        mode: str = "sse",
        task_id: str | None = None,
    ) -> AsyncIterator[events.GraphEvent]:
        """Run a turn, yielding events; persists the user message up front and the final
        assistant message at the end.

        For queue tasks the checkpoint thread is keyed by ``task_id``, so a redelivered task
        (after a worker crash) **resumes** its interrupted run from the checkpoint instead of
        starting over — and we skip re-appending the user message / reloading history, since
        the checkpoint already holds them. SSE runs in-process and is never resumed.
        """
        await self.ensure_conversation(conversation_id, user_id, agent_id)
        thread_id = task_id if (mode == "queue" and task_id) else conversation_id

        if mode == "queue" and task_id and await self._runner.is_resumable(thread_id):
            logger.info("conversation.resume", conversation_id=conversation_id, task_id=task_id)
            stream = self._runner.resume(thread_id=thread_id, mode=mode)
        else:
            history = await self.load_history(conversation_id)
            async with self._db.session() as session:
                await MessageRepository(session).add(conversation_id, "user", user_message)
            stream = self._runner.run(
                thread_id=thread_id, history=history, user_message=user_message, mode=mode
            )

        final_text = ""
        async for event in stream:
            if event.type == "done":
                final_text = event.data.get("final", "")
            yield event

        if final_text:
            async with self._db.session() as session:
                await MessageRepository(session).add(
                    conversation_id, "assistant", final_text, agent_id=agent_id
                )

    async def end(self, conversation_id: str) -> None:
        """Mark a conversation ended and trigger ephemeral-agent cleanup."""
        from hivemind.db.repository import EphemeralAgentRepository

        async with self._db.session() as session:
            await ConversationRepository(session).set_status(conversation_id, "ended")
            await EphemeralAgentRepository(session).delete_for_conversation(conversation_id)
        logger.info("conversation.ended", conversation_id=conversation_id)
