"""Tool base class and result type.

Tools are async, declare a JSON Schema for inputs, and return either inline JSON content
or an artifact reference. The registry validates inputs against the schema before
dispatch, so tools may assume well-formed args.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hivemind.core.context import RequestContext


@dataclass(slots=True)
class ToolResult:
    """Result of a tool invocation.

    Either ``content`` (small, inline JSON-serializable) or ``artifact`` (a reference to
    data written to the artifact store) should be set. ``is_error`` flags failures that
    should be surfaced to the model rather than raised.
    """

    content: Any = None
    artifact: dict[str, Any] | None = None
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        if self.artifact is not None:
            return {"artifact": self.artifact, "is_error": self.is_error, **self.metadata}
        return {"content": self.content, "is_error": self.is_error, **self.metadata}


class BaseTool(ABC):
    """Abstract base for all tools.

    Subclasses set ``name``, ``description``, ``input_schema`` and implement :meth:`run`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: RequestContext) -> ToolResult:
        """Execute the tool with validated ``args`` in request context ``ctx``."""
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
