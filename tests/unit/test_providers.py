"""LLM provider adapter tests using mocked SDK clients / httpx transport (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from hivemind.config import Settings
from hivemind.core.errors import LLMProviderError
from hivemind.core.llm.base import (
    LLMConfig,
    LLMRequest,
    Message,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from hivemind.core.llm.factory import LLMProviderFactory
from hivemind.core.llm.ollama_provider import OllamaProvider
from hivemind.core.llm.openai_provider import OpenAICompatibleProvider


def _req(tools=False) -> LLMRequest:
    schema = []
    if tools:
        from hivemind.core.llm.base import ToolSchema

        schema = [ToolSchema(name="add", description="add", input_schema={"type": "object"})]
    return LLMRequest(
        config=LLMConfig(provider="x", model="m", temperature=0.2),
        messages=[Message(role="user", content="hi")],
        system="sys",
        tools=schema,
    )


# ---- Ollama (httpx MockTransport) -----------------------------------------


async def test_ollama_stream_parses_ndjson_and_tools():
    lines = [
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"message": {"tool_calls": [{"function": {"name": "add", "arguments": {"a": 1}}}]}},
        {"done": True, "prompt_eval_count": 11, "eval_count": 7},
    ]
    body = "\n".join(json.dumps(line) for line in lines)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider("http://ollama", client=client)
    out = [ev async for ev in provider.stream(_req(tools=True))]
    texts = [e.text for e in out if isinstance(e, TextDelta)]
    assert "".join(texts) == "Hello"
    assert any(isinstance(e, ToolCallEvent) and e.tool_call.name == "add" for e in out)
    usage = next(e for e in out if isinstance(e, UsageEvent))
    assert usage.usage.input_tokens == 11
    await client.aclose()


async def test_ollama_complete_aggregates():
    body = json.dumps({"message": {"content": "done"}, "done": True})
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    )
    provider = OllamaProvider("http://ollama", client=client)
    resp = await provider.complete(_req())
    assert resp.text == "done"
    await client.aclose()


async def test_ollama_error_wrapped():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    provider = OllamaProvider("http://ollama", client=client)
    with pytest.raises(LLMProviderError):
        [ev async for ev in provider.stream(_req())]
    await client.aclose()


# ---- OpenAI-compatible (fake client) --------------------------------------


class _FakeChunk:
    def __init__(self, content=None, tool=None, finish=None, usage=None):
        delta = SimpleNamespace(content=content, tool_calls=tool)
        self.choices = (
            [SimpleNamespace(delta=delta, finish_reason=finish)]
            if (content or tool or finish)
            else []
        )
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class _FakeOpenAIClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


async def test_openai_stream_text_tools_usage():
    tool_delta = [
        SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name="add", arguments='{"a":1}'),
        )
    ]
    chunks = [
        _FakeChunk(content="Hi "),
        _FakeChunk(content="there"),
        _FakeChunk(tool=tool_delta),
        _FakeChunk(finish="stop", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
    ]
    provider = OpenAICompatibleProvider(_FakeOpenAIClient(chunks), name="openai")
    out = [ev async for ev in provider.stream(_req(tools=True))]
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == "Hi there"
    tc = next(e for e in out if isinstance(e, ToolCallEvent))
    assert tc.tool_call.name == "add" and tc.tool_call.arguments == {"a": 1}


# ---- Anthropic (fake stream client) ---------------------------------------


class _FakeAnthropicStream:
    def __init__(self, deltas, final):
        self._deltas = deltas
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for d in self._deltas:
                yield d

        return gen()

    async def get_final_message(self):
        return self._final


def test_anthropic_adaptive_only_strips_sampling(monkeypatch):
    from hivemind.core.llm import anthropic_provider as ap

    captured = {}

    class _Messages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            final = SimpleNamespace(
                content=[],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )
            return _FakeAnthropicStream([], final)

    class _Client:
        def __init__(self, *a, **k): ...

        messages = _Messages()

    monkeypatch.setattr(ap, "AsyncAnthropic", _Client, raising=False)
    # Patch the lazy import inside __init__ by injecting into the module the class would import.
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Client)
    provider = ap.AnthropicProvider(api_key=None, default_model="claude-opus-4-8")
    req = LLMRequest(
        config=LLMConfig(provider="anthropic", model="claude-opus-4-8", temperature=0.9),
        messages=[Message(role="user", content="hi")],
    )
    kwargs = provider._build_kwargs(req)
    assert "temperature" not in kwargs  # stripped for opus-4-8
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "high"


def test_anthropic_non_adaptive_keeps_temperature():
    from hivemind.core.llm.anthropic_provider import AnthropicProvider

    # No client construction needed for _build_kwargs; build a bare instance.
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._default_model = "claude-3-5-sonnet"  # type: ignore[attr-defined]
    req = LLMRequest(
        config=LLMConfig(provider="anthropic", model="claude-3-5-sonnet", temperature=0.5),
        messages=[Message(role="user", content="hi")],
    )
    kwargs = provider._build_kwargs(req)
    assert kwargs["temperature"] == 0.5
    assert "thinking" not in kwargs


# ---- Factory ---------------------------------------------------------------


def test_factory_builds_ollama_and_caches():
    factory = LLMProviderFactory(Settings(ollama_base_url="http://o"))
    p1 = factory.create(LLMConfig(provider="ollama", model="llama3.1"))
    p2 = factory.create(LLMConfig(provider="ollama", model="llama3.1"))
    assert p1 is p2
    assert p1.name == "ollama"


def test_factory_unknown_provider_raises():
    factory = LLMProviderFactory(Settings())
    with pytest.raises(LLMProviderError):
        factory.create(LLMConfig(provider="bogus", model="m"))


def test_factory_azure_requires_endpoint():
    factory = LLMProviderFactory(Settings(azure_openai_endpoint=None))
    with pytest.raises(LLMProviderError):
        factory.create(LLMConfig(provider="azure", model="m"))
