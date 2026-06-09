"""Ollama provider adapter — native ``/api/chat`` streaming over HTTP.

Lets the whole system run locally with zero API keys/cost. Tool-calls are normalized
from Ollama's ``message.tool_calls`` shape into the common :class:`ToolCall`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

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


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        cfg = request.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": _to_ollama_messages(request.messages, request.system),
            "stream": True,
            "options": {"num_predict": cfg.max_tokens},
        }
        if cfg.temperature is not None:
            payload["options"]["temperature"] = cfg.temperature
        if request.tools:
            payload["tools"] = [
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

        with span("llm.call", **{"llm.provider": self.name, "llm.model": cfg.model}):
            try:
                usage = Usage()
                async with self._client.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message") or {}
                        if msg.get("content"):
                            yield TextDelta(text=msg["content"])
                        for tc in msg.get("tool_calls", []) or []:
                            fn = tc.get("function", {})
                            yield ToolCallEvent(
                                tool_call=ToolCall(
                                    id=fn.get("name", "call"),
                                    name=fn.get("name", ""),
                                    arguments=_coerce_args(fn.get("arguments")),
                                )
                            )
                        if chunk.get("done"):
                            usage = Usage(
                                input_tokens=chunk.get("prompt_eval_count", 0),
                                output_tokens=chunk.get("eval_count", 0),
                            )
                record_llm_tokens(
                    self.name,
                    cfg.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                yield UsageEvent(usage=usage)
                yield DoneEvent(stop_reason="stop")
            except Exception as exc:
                raise LLMProviderError(f"Ollama stream failed: {exc}") from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await aggregate_stream(self.stream(request))

    async def aclose(self) -> None:
        await self._client.aclose()


def _to_ollama_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _coerce_args(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}
