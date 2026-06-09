"""OpenAI-compatible provider adapter.

Shared by **OpenAI**, **Azure OpenAI**, and **vLLM** — they all speak the OpenAI
Chat Completions API. The only differences are client construction (base URL, key,
Azure deployment) which are handled in the factory; the streaming/tool-call translation
is identical.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from hivemind.core.errors import LLMProviderError
from hivemind.core.llm._aggregate import aggregate_stream
from hivemind.core.llm.base import (
    DoneEvent,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from hivemind.observability.tracing import record_llm_tokens, span


class OpenAICompatibleProvider:
    """Wraps an ``openai.AsyncOpenAI``-compatible client.

    Args:
        client: a constructed ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` instance.
        name: provider label for metrics/spans (``openai`` | ``azure`` | ``vllm``).
    """

    def __init__(self, client: Any, *, name: str = "openai") -> None:
        self._client = client
        self.name = name

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        cfg = request.config
        messages = _to_openai_messages(request.messages, request.system)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
            kwargs["tool_choice"] = "auto" if request.tool_choice == "auto" else request.tool_choice

        with span("llm.call", **{"llm.provider": self.name, "llm.model": cfg.model}):
            try:
                tool_acc: dict[int, dict[str, Any]] = {}
                stop_reason = "stop"
                usage = Usage()
                stream = await self._client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if chunk.usage is not None:
                        usage = Usage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if getattr(delta, "content", None):
                        yield TextDelta(text=delta.content)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        acc = tool_acc.setdefault(
                            tc.index, {"id": tc.id, "name": "", "arguments": ""}
                        )
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments
                    if choice.finish_reason:
                        stop_reason = choice.finish_reason

                for acc in tool_acc.values():
                    yield ToolCallEvent(
                        tool_call=ToolCall(
                            id=acc["id"] or f"call_{acc['name']}",
                            name=acc["name"],
                            arguments=_safe_json(acc["arguments"]),
                        )
                    )
                record_llm_tokens(
                    self.name,
                    cfg.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                yield UsageEvent(usage=usage)
                yield DoneEvent(stop_reason=stop_reason)
            except Exception as exc:
                raise LLMProviderError(f"{self.name} stream failed: {exc}") from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await aggregate_stream(self.stream(request))


def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
