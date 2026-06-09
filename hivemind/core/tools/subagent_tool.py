"""Sub-agent spawner tool.

Lets any agent dynamically instantiate an ephemeral sub-agent (system prompt, tool subset,
LLM config) during a workflow. The sub-agent's definition is **checkpointed to Postgres
immediately** on creation (so it can be restored after a crash) and its lifecycle is tied
to the parent conversation. Execution is delegated to an injected :class:`SubAgentRunner`
to avoid a circular dependency on the graph/agent layers.
"""

from __future__ import annotations

from typing import Protocol

from hivemind.core.context import RequestContext
from hivemind.core.tools.base import BaseTool, ToolResult


class SubAgentRunner(Protocol):
    async def run_subagent(self, definition: dict, task: str, ctx: RequestContext) -> dict:
        """Persist the ephemeral agent, run it on ``task``, return a result payload."""
        ...


class SpawnSubAgentTool(BaseTool):
    name = "spawn_subagent"
    description = (
        "Spawn an ephemeral specialist sub-agent to handle a focused subtask. Provide its "
        "system prompt, the subset of tools it may use, and the task to perform. The "
        "sub-agent runs to completion and returns its result."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "system_prompt": {"type": "string", "description": "Sub-agent persona/instructions."},
            "task": {"type": "string", "description": "The subtask for the sub-agent to perform."},
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subset of tool names the sub-agent may use.",
                "default": [],
            },
            "model": {"type": "string", "description": "Optional model override."},
        },
        "required": ["system_prompt", "task"],
        "additionalProperties": False,
    }

    def __init__(self, runner: SubAgentRunner) -> None:
        self._runner = runner

    async def run(self, args: dict, ctx: RequestContext) -> ToolResult:
        definition = {
            "system_prompt": args["system_prompt"],
            "tool_names": args.get("tool_names", []),
            "model": args.get("model"),
        }
        result = await self._runner.run_subagent(definition, args["task"], ctx)
        return ToolResult(content=result)
