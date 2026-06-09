"""Web search tool.

Ships with a deterministic stub backend so the system runs without external API keys.
A real backend is plugged in by implementing :class:`SearchBackend` and passing it to the
tool — the tool itself never changes.
"""

from __future__ import annotations

from typing import Protocol

from hivemind.core.context import RequestContext
from hivemind.core.tools.base import BaseTool, ToolResult


class SearchBackend(Protocol):
    async def search(self, query: str, *, limit: int) -> list[dict]: ...


class StubSearchBackend:
    """Returns shaped placeholder results — replace with a real provider in production."""

    async def search(self, query: str, *, limit: int) -> list[dict]:
        return [
            {
                "title": f"Result {i + 1} for {query!r}",
                "url": f"https://example.com/search?q={query}&r={i + 1}",
                "snippet": f"Placeholder snippet {i + 1} for query {query!r}.",
            }
            for i in range(min(limit, 3))
        ]


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for up-to-date information and return ranked results."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self._backend = backend or StubSearchBackend()

    async def run(self, args: dict, ctx: RequestContext) -> ToolResult:
        results = await self._backend.search(args["query"], limit=args.get("limit", 5))
        return ToolResult(content={"results": results})
