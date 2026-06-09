"""Anthropic provider adapter (Claude).

Model-aware: Opus 4.7 / 4.8 reject ``temperature``/``top_p``/``top_k`` and the legacy
``budget_tokens`` thinking config (they 400). For those models we omit sampling params and
use adaptive thinking + the ``effort`` knob. Default model: ``claude-opus-4-8``.
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
    ThinkingDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from hivemind.observability.tracing import record_llm_tokens, span

# Models that reject sampling params and use adaptive thinking only.
_ADAPTIVE_ONLY_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8")


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        *,
        default_model: str = "claude-opus-4-8",
        prompt_cache: bool = True,
    ) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self._default_model = default_model
        self._prompt_cache = prompt_cache

    def _is_adaptive_only(self, model: str) -> bool:
        return any(model.startswith(p) for p in _ADAPTIVE_ONLY_PREFIXES)

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        cfg = request.config
        model = cfg.model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": cfg.max_tokens,
            "messages": _to_anthropic_messages(request.messages),
        }
        if request.system:
            # Prompt caching: the system prompt is a stable prefix reused across turns.
            # Marking it with cache_control lets Anthropic serve it from cache (~0.1x cost,
            # lower latency). The cache key is a prefix match, so a frozen system prompt is
            # essential — we never interpolate per-request data into it.
            if self._prompt_cache:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": request.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = request.system
        if request.tools:
            tools = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
            # Cache the (stable) tool definitions too — they render before system.
            if self._prompt_cache and tools:
                tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": request.tool_choice}

        if self._is_adaptive_only(model):
            # Sampling params and budget_tokens are rejected; use adaptive thinking.
            kwargs["thinking"] = {"type": "adaptive"}
            effort = cfg.extra.get("effort", "high")
            kwargs["output_config"] = {"effort": effort}
        else:
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                kwargs["top_p"] = cfg.top_p
        return kwargs

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        kwargs = self._build_kwargs(request)
        with span("llm.call", **{"llm.provider": self.name, "llm.model": kwargs["model"]}):
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        delta = _translate_delta(event)
                        if delta is not None:
                            yield delta
                    final = await stream.get_final_message()
                # Tool-use blocks carry fully-accumulated input on the final message.
                for block in final.content:
                    if getattr(block, "type", None) == "tool_use":
                        yield ToolCallEvent(
                            tool_call=ToolCall(
                                id=block.id, name=block.name, arguments=_coerce_input(block.input)
                            )
                        )
                usage = Usage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                )
                record_llm_tokens(
                    self.name,
                    kwargs["model"],
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                yield UsageEvent(usage=usage)
                yield DoneEvent(stop_reason=final.stop_reason or "end_turn")
            except LLMProviderError:
                raise
            except Exception as exc:
                raise LLMProviderError(f"Anthropic stream failed: {exc}") from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await aggregate_stream(self.stream(request))


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate normalized messages to the Anthropic messages format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _translate_delta(event: Any) -> LLMStreamEvent | None:
    """Map an Anthropic stream event to a text/thinking delta (or None)."""
    if getattr(event, "type", None) != "content_block_delta":
        return None
    delta = event.delta
    if delta.type == "text_delta":
        return TextDelta(text=delta.text)
    if delta.type == "thinking_delta":
        return ThinkingDelta(text=delta.thinking)
    return None


def _coerce_input(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}
