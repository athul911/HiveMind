"""OpenAI-compatible provider adapter (OpenAI, Azure OpenAI, vLLM, and other gateways).

Speaks the OpenAI Chat Completions wire format over plain HTTP (httpx) and parses the SSE
stream **leniently**. This matters because "OpenAI-compatible" endpoints diverge from the
exact schema in small ways the official SDK rejects with strict pydantic validation — e.g.
Google's Gemini compatibility endpoint omits the ``index`` field on streaming
``tool_calls`` deltas and returns ``finish_reason: "stop"`` alongside a tool call. Hand-rolling
the parse keeps us robust across OpenAI, Azure, vLLM, OpenRouter, LiteLLM, and Google.

Tool-call accumulation keys on ``index`` when present (OpenAI streams a call across many
deltas by index), otherwise on the call ``id`` (Google sends one complete delta with an id),
so both shapes assemble correctly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

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

AuthStyle = Literal["bearer", "azure"]


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        name: str = "openai",
        auth: AuthStyle = "bearer",
        api_version: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""
        self.name = name
        self._auth = auth
        self._api_version = api_version
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    def _endpoint_and_headers(self, model: str) -> tuple[str, dict[str, str]]:
        headers = {"Content-Type": "application/json"}
        if self._auth == "azure":
            url = (
                f"{self._base_url}/openai/deployments/{model}/chat/completions"
                f"?api-version={self._api_version}"
            )
            headers["api-key"] = self._api_key
        else:
            url = f"{self._base_url}/chat/completions"
            headers["Authorization"] = f"Bearer {self._api_key}"
        return url, headers

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        cfg = request.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": _to_openai_messages(request.messages, request.system),
            "max_tokens": cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if cfg.temperature is not None:
            payload["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
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
            payload["tool_choice"] = request.tool_choice
        return payload

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        cfg = request.config
        url, headers = self._endpoint_and_headers(cfg.model)
        payload = self._payload(request)

        with span("llm.call", **{"llm.provider": self.name, "llm.model": cfg.model}):
            try:
                tool_acc: dict[Any, dict[str, Any]] = {}
                stop_reason = "stop"
                usage = Usage()
                async with self._client.stream(
                    "POST", url, headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise LLMProviderError(f"{self.name} {resp.status_code}: {body[:500]}")
                    async for raw in resp.aiter_lines():
                        line = raw.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if chunk.get("usage"):
                            u = chunk["usage"]
                            usage = Usage(
                                input_tokens=u.get("prompt_tokens", 0) or 0,
                                output_tokens=u.get("completion_tokens", 0) or 0,
                            )
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield TextDelta(text=delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            _accumulate_tool_call(tool_acc, tc)
                        if choice.get("finish_reason"):
                            stop_reason = choice["finish_reason"]

                for acc in tool_acc.values():
                    if not acc["name"]:
                        continue
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
            except LLMProviderError:
                raise
            except Exception as exc:
                raise LLMProviderError(f"{self.name} stream failed: {exc}") from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await aggregate_stream(self.stream(request))

    async def aclose(self) -> None:
        await self._client.aclose()


def _accumulate_tool_call(acc_map: dict[Any, dict[str, Any]], tc: dict[str, Any]) -> None:
    """Accumulate a streaming tool-call delta.

    OpenAI sends a tool call across multiple deltas keyed by ``index``; Google sends one
    complete delta with an ``id`` and no ``index``. Key on whichever correlates the deltas.
    """
    key = tc.get("index")
    if key is None:
        key = tc.get("id") or len(acc_map)
    acc = acc_map.setdefault(key, {"id": tc.get("id"), "name": "", "arguments": ""})
    if tc.get("id"):
        acc["id"] = tc["id"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        acc["name"] = fn["name"]
    if fn.get("arguments"):
        acc["arguments"] += fn["arguments"]


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
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
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
