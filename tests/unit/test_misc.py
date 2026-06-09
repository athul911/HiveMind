from __future__ import annotations

import pytest
from hivemind.config import Settings
from hivemind.core.graph.supervisor import _parse_plan
from hivemind.core.llm.base import LLMConfig
from hivemind.services.artifact_store import ArtifactStore
from hivemind.services.mode_selector import ModeSelector

_TABLE = [{"agent_id": "a1", "name": "x", "description": "y"}]


def test_parse_plan_extracts_json():
    text = 'sure, here:\n{"mode": "single", "agents": ["a1"], "reasoning": "ok"}\nthanks'
    plan = _parse_plan(text, _TABLE)
    assert plan["mode"] == "single"
    assert plan["agents"] == ["a1"]


def test_parse_plan_rejects_non_json():
    with pytest.raises(ValueError):
        _parse_plan("no json here", _TABLE)


def test_mode_selector_forces_queue_when_not_streaming():
    sel = ModeSelector(Settings(workflow_async_threshold_steps=3))
    assert sel.select(stream=False, agent_count=1) == "queue"


def test_mode_selector_sse_for_small_streaming():
    sel = ModeSelector(Settings(workflow_async_threshold_steps=3))
    assert sel.select(stream=True, agent_count=2) == "sse"


def test_mode_selector_queue_over_threshold():
    sel = ModeSelector(Settings(workflow_async_threshold_steps=3))
    assert sel.select(stream=True, agent_count=5) == "queue"


def test_llm_config_roundtrip():
    cfg = LLMConfig(provider="anthropic", model="claude-opus-4-8", extra={"effort": "high"})
    again = LLMConfig.from_dict(cfg.to_dict())
    assert again.provider == "anthropic"
    assert again.extra["effort"] == "high"


def test_artifact_store_write_and_traversal_block(tmp_path):
    store = ArtifactStore(str(tmp_path))
    ref = store.write_text("conv1", "task1", "code_exec", "out.txt", "hello")
    assert ref.size_bytes == 5
    assert store.read_ref(ref.path) == b"hello"
    with pytest.raises(ValueError):
        store.write_text("conv1", "task1", "code_exec", "../escape.txt", "x")
