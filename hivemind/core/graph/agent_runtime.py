"""Single-agent execution turn: stream LLM, run tools, loop until done.

Reused by both top-level agent nodes and ephemeral sub-agents. Independent of LangGraph —
it takes an ``emit`` callback so it is unit-testable with a fake. Implements the standard
agentic loop: call the model, if it requests tools execute them and feed results back,
repeat until the model stops calling tools (or the per-turn tool budget is hit).
"""

from __future__ import annotations

from collections.abc import Callable

from hivemind.core.context import RequestContext
from hivemind.core.errors import BudgetExceededError
from hivemind.core.graph import events
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.llm.base import (
    DoneEvent,
    LLMRequest,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
    UsageEvent,
)

Emit = Callable[[events.GraphEvent], None]

_MAX_TOOL_ROUNDS = 8


async def run_agent_turn(
    *,
    deps: GraphDeps,
    system_prompt: str,
    tool_names: list[str],
    llm_config,
    conversation: list[Message],
    ctx: RequestContext,
    emit: Emit,
    token_budget: int | None = None,
) -> tuple[str, list[Message], Usage]:
    """Run one agent to completion. Returns (final_text, appended_messages, usage).

    ``conversation`` is the running message list (mutated copy returned). ``emit`` receives
    text deltas, tool_call, and tool_result events for streaming.
    """
    provider = deps.llm_factory.create(llm_config)
    tool_schemas = deps.tools.schemas_for(tool_names)
    messages = list(conversation)
    appended: list[Message] = []
    total_usage = Usage()
    final_text = ""

    for _ in range(_MAX_TOOL_ROUNDS):
        request = LLMRequest(
            config=llm_config,
            messages=list(messages),  # snapshot: don't let later appends mutate this call
            system=system_prompt,
            tools=tool_schemas,
            tool_choice="auto",
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = "end_turn"
        async for ev in provider.stream(request):
            if isinstance(ev, TextDelta):
                text_parts.append(ev.text)
                emit(events.text_delta(ev.text, agent_id=ctx.agent_id))
            elif isinstance(ev, ToolCallEvent):
                tool_calls.append(ev.tool_call)
            elif isinstance(ev, UsageEvent):
                total_usage = total_usage + ev.usage
                emit(events.usage(ev.usage.input_tokens, ev.usage.output_tokens))
            elif isinstance(ev, DoneEvent):
                stop_reason = ev.stop_reason

        if (
            token_budget is not None
            and total_usage.output_tokens + total_usage.input_tokens > token_budget
        ):
            raise BudgetExceededError(
                "Token budget exceeded for this workflow.",
                tokens_used=total_usage.input_tokens + total_usage.output_tokens,
            )

        assistant_text = "".join(text_parts)
        assistant_msg = Message(role="assistant", content=assistant_text, tool_calls=tool_calls)
        messages.append(assistant_msg)
        appended.append(assistant_msg)

        if not tool_calls:
            final_text = assistant_text
            # Surface why the turn ended so callers can spot truncation ("length") or a model
            # that answered in plain text without calling an expected tool ("end_turn"/"stop").
            emit(
                events.GraphEvent(
                    "agent_finished",
                    {
                        "agent_id": ctx.agent_id,
                        "stop_reason": stop_reason,
                        "had_tool_calls": False,
                    },
                )
            )
            break

        # Execute requested tools and feed results back.
        for call in tool_calls:
            emit(events.tool_call(call.name, call.arguments, call.id))
            result = await deps.tools.execute(call.name, call.arguments, ctx)
            payload = result.to_payload()
            emit(events.tool_result(call.name, payload, call.id))
            tool_msg = Message(
                role="tool",
                content=_stringify(payload),
                tool_call_id=call.id,
                name=call.name,
            )
            messages.append(tool_msg)
            appended.append(tool_msg)

    return final_text, appended, total_usage


def _stringify(payload: dict) -> str:
    import json

    return json.dumps(payload, default=str)
