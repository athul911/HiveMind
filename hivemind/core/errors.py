"""Typed exception hierarchy with RFC 7807 problem+json mapping.

Every HiveMind error carries request context so failures are traceable. The API layer
converts these into ``application/problem+json`` responses; internal details are never
leaked to clients.
"""

from __future__ import annotations

from typing import Any

from hivemind.core.context import get_context


class HiveMindError(Exception):
    """Base class for all domain errors.

    Attributes:
        status_code: HTTP status to surface.
        error_type: stable machine-readable slug (the problem ``type`` suffix).
        title: short human-readable summary.
        detail: longer explanation (safe to show to clients).
        extra: additional problem members.
    """

    status_code: int = 500
    error_type: str = "internal-error"
    title: str = "Internal Server Error"

    def __init__(self, detail: str | None = None, **extra: Any) -> None:
        self.detail = detail or self.title
        self.extra = extra
        super().__init__(self.detail)

    def to_problem(self) -> dict[str, Any]:
        """Render as an RFC 7807 problem detail object."""
        problem: dict[str, Any] = {
            "type": f"https://hivemind.dev/errors/{self.error_type}",
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
            **self.extra,
        }
        ctx = get_context()
        if ctx is not None:
            if ctx.request_id:
                problem["request_id"] = ctx.request_id
            if ctx.conversation_id:
                problem["conversation_id"] = ctx.conversation_id
        return problem


class ValidationError(HiveMindError):
    status_code = 400
    error_type = "validation-error"
    title = "Validation Error"


class AuthenticationError(HiveMindError):
    status_code = 401
    error_type = "authentication-error"
    title = "Authentication Failed"


class AuthorizationError(HiveMindError):
    status_code = 403
    error_type = "authorization-error"
    title = "Not Authorized"


class NotFoundError(HiveMindError):
    status_code = 404
    error_type = "not-found"
    title = "Resource Not Found"


class ConflictError(HiveMindError):
    status_code = 409
    error_type = "conflict"
    title = "Conflict"


class RateLimitError(HiveMindError):
    status_code = 429
    error_type = "rate-limited"
    title = "Too Many Requests"


class ImmutableAgentError(ConflictError):
    error_type = "immutable-agent"
    title = "Agent Is Immutable"


class ToolExecutionError(HiveMindError):
    status_code = 422
    error_type = "tool-execution-error"
    title = "Tool Execution Failed"


class SandboxError(HiveMindError):
    status_code = 422
    error_type = "sandbox-error"
    title = "Sandboxed Execution Failed"


class UnsafeSQLError(ValidationError):
    error_type = "unsafe-sql"
    title = "Unsafe SQL Rejected"


class BudgetExceededError(HiveMindError):
    status_code = 429
    error_type = "budget-exceeded"
    title = "Workflow Budget Exceeded"


class LLMProviderError(HiveMindError):
    status_code = 502
    error_type = "llm-provider-error"
    title = "Upstream LLM Provider Error"
