"""Tests for the subprocess sandbox, code-exec & web-search tools, and supervisor routing."""

from __future__ import annotations

import sys

import pytest
from hivemind.config import Settings
from hivemind.core.agents.agent import Agent
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.context import RequestContext
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.graph.supervisor import decide_route
from hivemind.core.llm.base import LLMConfig
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.code_tool import CodeExecTool
from hivemind.core.tools.registry import ToolRegistry
from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.core.tools.sandbox.subprocess_sandbox import SubprocessSandbox
from hivemind.core.tools.web_search_tool import StubSearchBackend, WebSearchTool
from hivemind.services.artifact_store import ArtifactStore

from tests.conftest import ScriptedFactory, ScriptedProvider

# ---- Subprocess sandbox (executes real child processes) -------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX preexec/RLIMIT only")
async def test_subprocess_sandbox_runs_code(tmp_path):
    sandbox = SubprocessSandbox()
    result = await sandbox.run("print('hello sandbox')", artifact_dir=tmp_path, timeout_s=10)
    assert result.exit_code == 0
    assert "hello sandbox" in result.stdout
    assert not result.timed_out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX preexec/RLIMIT only")
async def test_subprocess_sandbox_times_out(tmp_path):
    sandbox = SubprocessSandbox()
    result = await sandbox.run(
        "import time\ntime.sleep(5)", artifact_dir=tmp_path, timeout_s=1
    )
    assert result.timed_out
    assert result.exit_code == 124


# ---- Code-exec tool (fake sandbox + real artifact store) ------------------

class _FakeSandbox:
    def __init__(self, result: SandboxResult, *, write_file: bool = False) -> None:
        self._result = result
        self._write_file = write_file

    async def run(self, code, *, artifact_dir, timeout_s):
        if self._write_file:
            (artifact_dir / "out.csv").write_text("a,b\n1,2\n")
        return self._result


async def test_code_exec_tool_returns_artifacts(tmp_path):
    store = ArtifactStore(str(tmp_path))
    sandbox = _FakeSandbox(SandboxResult(0, "ok", ""), write_file=True)
    tool = CodeExecTool(sandbox, store, timeout_s=5)
    result = await tool.run({"code": "print('x')"}, RequestContext(conversation_id="c1"))
    assert result.is_error is False
    assert result.content["exit_code"] == 0
    assert any(a["path"].endswith("out.csv") for a in result.content["artifacts"])


async def test_code_exec_tool_flags_nonzero_exit(tmp_path):
    store = ArtifactStore(str(tmp_path))
    sandbox = _FakeSandbox(SandboxResult(1, "", "boom"))
    tool = CodeExecTool(sandbox, store, timeout_s=5)
    result = await tool.run({"code": "raise SystemExit(1)"}, RequestContext(conversation_id="c1"))
    assert result.is_error is True
    assert "boom" in result.content["stderr"]


# ---- Web search tool -------------------------------------------------------

async def test_web_search_stub_returns_results():
    tool = WebSearchTool(StubSearchBackend())
    result = await tool.run({"query": "hivemind", "limit": 2}, RequestContext())
    assert len(result.content["results"]) == 2
    assert "hivemind" in result.content["results"][0]["title"]


# ---- Supervisor routing ----------------------------------------------------

def _deps(provider, agents):
    tools = ToolRegistry()
    skills = SkillRegistry()
    registry = AgentRegistry()
    for a in agents:
        registry.add(a)
    return GraphDeps(
        settings=Settings(otel_enabled=False),
        agents=registry,
        agent_factory=AgentFactory(tools, skills),
        llm_factory=ScriptedFactory(provider),
        tools=tools,
    )


def _agent(name: str) -> Agent:
    return Agent(
        name=name,
        description=f"{name} agent",
        system_prompt="p",
        llm_config=LLMConfig(provider="scripted", model="m"),
    )


async def test_supervisor_no_agents():
    plan = await decide_route(_deps(ScriptedProvider([]), []), "hi")
    assert plan["agents"] == []


async def test_supervisor_single_agent_shortcircuits():
    a = _agent("solo")
    plan = await decide_route(_deps(ScriptedProvider([]), [a]), "hi")
    assert plan == {"mode": "single", "agents": [a.id], "reasoning": "only one agent"}


async def test_supervisor_parses_llm_plan():
    a, b = _agent("alpha"), _agent("beta")
    from hivemind.core.llm.base import DoneEvent, TextDelta, Usage, UsageEvent

    plan_json = f'{{"mode":"parallel","agents":["{a.id}","{b.id}"],"reasoning":"both"}}'
    provider = ScriptedProvider([[TextDelta(plan_json), UsageEvent(Usage(1, 1)), DoneEvent("end")]])
    plan = await decide_route(_deps(provider, [a, b]), "do both")
    assert plan["mode"] == "parallel"
    assert set(plan["agents"]) == {a.id, b.id}


async def test_supervisor_falls_back_on_bad_llm_output():
    a, b = _agent("alpha"), _agent("beta")
    from hivemind.core.llm.base import DoneEvent, TextDelta, Usage, UsageEvent

    turn = [TextDelta("not json"), UsageEvent(Usage(1, 1)), DoneEvent("end")]
    provider = ScriptedProvider([turn])
    plan = await decide_route(_deps(provider, [a, b]), "?")
    assert plan["agents"] and plan["agents"][0] in {a.id, b.id}
