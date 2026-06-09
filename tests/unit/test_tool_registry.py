from __future__ import annotations

import pytest
from hivemind.core.context import RequestContext
from hivemind.core.errors import NotFoundError, ToolExecutionError, ValidationError
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.core.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo a message."
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }

    async def run(self, args, ctx):
        return ToolResult(content={"echo": args["message"]})


class BoomTool(BaseTool):
    name = "boom"
    description = "Always fails."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, args, ctx):
        raise RuntimeError("kaboom")


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(EchoTool())
    r.register(BoomTool())
    return r


async def test_execute_validates_and_runs(registry: ToolRegistry):
    result = await registry.execute("echo", {"message": "hi"}, RequestContext())
    assert result.content == {"echo": "hi"}


async def test_execute_rejects_invalid_args(registry: ToolRegistry):
    with pytest.raises(ValidationError):
        await registry.execute("echo", {"wrong": 1}, RequestContext())


async def test_execute_unknown_tool(registry: ToolRegistry):
    with pytest.raises(NotFoundError):
        await registry.execute("nope", {}, RequestContext())


async def test_execute_wraps_tool_errors(registry: ToolRegistry):
    with pytest.raises(ToolExecutionError):
        await registry.execute("boom", {}, RequestContext())


def test_duplicate_registration_rejected(registry: ToolRegistry):
    with pytest.raises(ValidationError):
        registry.register(EchoTool())


def test_schemas_for_skips_unknown(registry: ToolRegistry):
    schemas = registry.schemas_for(["echo", "ghost"])
    assert [s.name for s in schemas] == ["echo"]
