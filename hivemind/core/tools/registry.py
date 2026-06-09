"""Centralized tool registry.

Tools register globally by unique name. The registry validates inputs against each
tool's JSON Schema before dispatch, wraps every invocation in an OTel span and a
structured log line, and records the ``hivemind.tool.calls.total`` metric. Agents bind a
subset of registered tools by name.
"""

from __future__ import annotations

import time

import jsonschema

from hivemind.core.context import RequestContext
from hivemind.core.errors import NotFoundError, ToolExecutionError, ValidationError
from hivemind.core.llm.base import ToolSchema
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.observability.logging import get_logger
from hivemind.observability.tracing import record_tool_call, span

logger = get_logger("hivemind.tools")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValidationError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(f"Tool not found: {name}", tool_name=name)
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas_for(self, names: list[str]) -> list[ToolSchema]:
        """Return LLM tool schemas for the named subset (skips unknown names)."""
        out: list[ToolSchema] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                out.append(
                    ToolSchema(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                )
        return out

    async def execute(self, name: str, args: dict, ctx: RequestContext) -> ToolResult:
        """Validate args against the tool schema, then dispatch with tracing/logging."""
        tool = self.get(name)
        try:
            jsonschema.validate(instance=args, schema=tool.input_schema)
        except jsonschema.ValidationError as exc:
            raise ValidationError(f"Invalid arguments for tool {name}: {exc.message}") from exc

        started = time.perf_counter()
        with span("tool.invoke", **{"tool.name": name}):
            log = logger.bind(tool=name)
            log.info("tool.start", args_keys=list(args.keys()))
            try:
                result = await tool.run(args, ctx)
            except ToolExecutionError:
                record_tool_call(name, success=False)
                raise
            except Exception as exc:
                record_tool_call(name, success=False)
                log.error("tool.error", error=str(exc))
                raise ToolExecutionError(f"Tool {name} failed: {exc}") from exc
            duration = time.perf_counter() - started
            record_tool_call(name, success=not result.is_error)
            log.info("tool.done", duration_s=round(duration, 4), is_error=result.is_error)
            return result
