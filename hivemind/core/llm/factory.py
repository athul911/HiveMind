"""LLM provider factory with per-provider client pooling.

Instantiates the correct adapter for a given :class:`LLMConfig`. Clients are cached by
provider so we reuse connection pools across requests. The factory depends only on the
:class:`LLMProvider` protocol — callers never import concrete adapters.
"""

from __future__ import annotations

from hivemind.config import Settings
from hivemind.core.errors import LLMProviderError
from hivemind.core.llm.base import LLMConfig, LLMProvider


class LLMProviderFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, LLMProvider] = {}

    def create(self, config: LLMConfig) -> LLMProvider:
        provider = config.provider.lower()
        if provider in self._cache:
            return self._cache[provider]
        instance = self._wrap_resilient(self._build(provider))
        self._cache[provider] = instance
        return instance

    def _wrap_resilient(self, inner: LLMProvider) -> LLMProvider:
        from hivemind.core.llm.resilience import ResilientProvider

        s = self._settings
        return ResilientProvider(
            inner,
            max_retries=s.llm_max_retries,
            base_delay=s.llm_retry_base_delay_s,
            breaker_threshold=s.circuit_breaker_threshold,
            breaker_reset_s=s.circuit_breaker_reset_s,
        )

    def _build(self, provider: str) -> LLMProvider:
        s = self._settings
        if provider == "anthropic":
            from hivemind.core.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                s.anthropic_api_key,
                default_model=s.llm_default_model,
                prompt_cache=s.prompt_cache_enabled,
            )

        if provider == "openai":
            from openai import AsyncOpenAI

            from hivemind.core.llm.openai_provider import OpenAICompatibleProvider

            client = AsyncOpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)
            return OpenAICompatibleProvider(client, name="openai")

        if provider == "azure":
            from openai import AsyncAzureOpenAI

            from hivemind.core.llm.openai_provider import OpenAICompatibleProvider

            if not s.azure_openai_endpoint:
                raise LLMProviderError("AZURE_OPENAI_ENDPOINT is not configured.")
            client = AsyncAzureOpenAI(
                azure_endpoint=s.azure_openai_endpoint,
                api_key=s.azure_openai_api_key,
                api_version=s.azure_openai_api_version,
            )
            return OpenAICompatibleProvider(client, name="azure")

        if provider == "vllm":
            from openai import AsyncOpenAI

            from hivemind.core.llm.openai_provider import OpenAICompatibleProvider

            client = AsyncOpenAI(api_key=s.vllm_api_key, base_url=s.vllm_base_url)
            return OpenAICompatibleProvider(client, name="vllm")

        if provider == "ollama":
            from hivemind.core.llm.ollama_provider import OllamaProvider

            return OllamaProvider(s.ollama_base_url)

        raise LLMProviderError(f"Unknown LLM provider: {provider!r}")
