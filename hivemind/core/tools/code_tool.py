"""Code generation & execution tool.

Runs arbitrary Python in a :class:`Sandbox` (Docker by default). Any files the code writes
to the working directory (``/artifacts`` in-container) are captured and returned as artifact
references — the tool returns filesystem references, not raw data, to minimize token usage.
stdout/stderr are truncated to a small inline preview.
"""

from __future__ import annotations

from pathlib import Path

from hivemind.core.context import RequestContext
from hivemind.core.errors import SandboxError
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.core.tools.sandbox.base import Sandbox
from hivemind.services.artifact_store import ArtifactStore

_STDOUT_PREVIEW = 4_000


class CodeExecTool(BaseTool):
    name = "code_exec"
    description = (
        "Execute Python code in an isolated sandbox (no network access). Write any output "
        "files to the current working directory; they are saved and returned as artifact "
        "references. Pre-installed: the standard library. Returns stdout/stderr preview "
        "plus references to generated files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox: Sandbox, artifacts: ArtifactStore, *, timeout_s: int = 30) -> None:
        self._sandbox = sandbox
        self._artifacts = artifacts
        self._timeout_s = timeout_s

    async def run(self, args: dict, ctx: RequestContext) -> ToolResult:
        code = args["code"]
        artifact_dir = self._artifacts.namespace_dir(
            ctx.conversation_id or "sync", ctx.task_id, self.name
        )
        try:
            result = await self._sandbox.run(
                code, artifact_dir=artifact_dir, timeout_s=self._timeout_s
            )
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox execution failed: {exc}") from exc

        produced = _collect_artifacts(self._artifacts, artifact_dir)
        payload = {
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout[:_STDOUT_PREVIEW],
            "stderr": result.stderr[:_STDOUT_PREVIEW],
            "artifacts": produced,
        }
        return ToolResult(content=payload, is_error=result.exit_code != 0)


def _collect_artifacts(store: ArtifactStore, artifact_dir: Path) -> list[dict]:
    refs: list[dict] = []
    for path in sorted(artifact_dir.iterdir()):
        if path.name == "_code.py" or not path.is_file():
            continue
        refs.append(store.describe(path).to_dict())
    return refs
