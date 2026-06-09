"""Centralized configuration loaded from the environment via pydantic-settings.

All runtime configuration flows through :class:`Settings`. Nothing reads ``os.environ``
directly. Settings are created once at process startup and injected as a dependency,
keeping configuration out of global mutable state.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SandboxBackend = Literal["docker", "subprocess"]
LLMProviderName = Literal["anthropic", "openai", "azure", "vllm", "ollama"]


class Settings(BaseSettings):
    """Process-wide configuration. Field names map to UPPER_SNAKE env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- service identity -------------------------------------------------
    service_name: str = "hivemind"
    environment: str = "local"
    log_level: str = "INFO"

    # ---- persistence ------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://hivemind:hivemind@localhost:5432/hivemind",
        description="Async SQLAlchemy DSN (asyncpg driver).",
    )
    # Separate least-privilege read-only DSN used by the SQL tool. Falls back to
    # database_url if unset, but production must set a distinct read-only role.
    sql_tool_database_url: str | None = None
    sql_tool_statement_timeout_ms: int = 10_000
    sql_tool_max_rows: int = 1_000
    sql_tool_allowed_schemas: list[str] = Field(default_factory=lambda: ["public"])

    # ---- broker / cache ---------------------------------------------------
    rabbitmq_url: str = "amqp://hivemind:hivemind@localhost:5672/"
    rabbitmq_task_queue: str = "hivemind.tasks"
    redis_url: str = "redis://localhost:6379/0"

    # ---- auth -------------------------------------------------------------
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_audience: str | None = None
    jwt_issuer: str | None = None
    oauth2_jwks_url: str | None = None  # when set, RS256/JWKS verification is used
    auth_disabled: bool = False  # local/dev escape hatch only

    rate_limit_per_minute: int = 120

    # ---- artifacts --------------------------------------------------------
    artifact_base_path: str = "/data/artifacts"

    # ---- sandbox ----------------------------------------------------------
    sandbox_backend: SandboxBackend = "docker"
    sandbox_image: str = "python:3.11-slim"
    sandbox_timeout_s: int = 30
    sandbox_memory: str = "256m"
    sandbox_cpus: float = 1.0
    sandbox_pids_limit: int = 128

    # ---- llm --------------------------------------------------------------
    llm_default_provider: LLMProviderName = "anthropic"
    llm_default_model: str = "claude-opus-4-8"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"

    # ---- orchestration ----------------------------------------------------
    workflow_async_threshold_steps: int = 3
    supervisor_max_iterations: int = 12
    supervisor_token_budget: int = 200_000  # per-conversation cumulative token ceiling
    ephemeral_agent_ttl_seconds: int = 3_600
    cleanup_interval_seconds: int = 300
    subagent_max_depth: int = 2  # how deep spawn_subagent may recurse

    # ---- conversation memory ----------------------------------------------
    conversation_history_limit: int = 40  # turns kept verbatim before compaction
    conversation_compaction_enabled: bool = True

    # ---- llm resilience ---------------------------------------------------
    llm_max_retries: int = 2
    llm_retry_base_delay_s: float = 0.5
    circuit_breaker_threshold: int = 5  # consecutive failures before a provider trips
    circuit_breaker_reset_s: float = 30.0
    prompt_cache_enabled: bool = True

    # ---- authz / cors -----------------------------------------------------
    rbac_enabled: bool = False  # when true, agent-management routes require admin_scope
    admin_scope: str = "hivemind:admin"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ---- skills -----------------------------------------------------------
    skills_dir: str = "skills"

    # ---- observability ----------------------------------------------------
    otel_exporter_otlp_endpoint: str | None = None
    otel_enabled: bool = True
    prometheus_port: int = 9464

    @field_validator("sql_tool_allowed_schemas", "cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        # Allow comma-separated env values for these list fields.
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def effective_sql_tool_dsn(self) -> str:
        return self.sql_tool_database_url or self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
