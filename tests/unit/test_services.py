"""Service- and worker-layer tests using fakes + repository monkeypatching (no live infra)."""

from __future__ import annotations

import pytest
from hivemind.core.graph import events
from hivemind.services import conversation as convo_mod
from hivemind.services import scheduler as sched_mod
from hivemind.services.conversation import ConversationService
from hivemind.services.scheduler import CleanupScheduler
from hivemind.workers import events as events_mod
from hivemind.workers.dispatcher import TaskDispatcher
from hivemind.workers.events import TaskEventBuffer

from tests import fakes


@pytest.fixture(autouse=True)
def _reset():
    fakes.reset_fakes()
    yield
    fakes.reset_fakes()


# ---- ConversationService ---------------------------------------------------


async def test_conversation_stream_persists_user_and_final(monkeypatch):
    monkeypatch.setattr(convo_mod, "ConversationRepository", fakes.FakeConversationRepo)
    monkeypatch.setattr(convo_mod, "MessageRepository", fakes.FakeMessageRepo)

    runner = fakes.FakeRunner(
        [
            events.text_delta("hi"),
            events.message("assistant", "final answer"),
            events.done("final answer"),
        ]
    )
    svc = ConversationService(fakes.FakeDatabase(), runner, ttl_seconds=60)

    collected = [
        ev
        async for ev in svc.stream(
            conversation_id="c1", user_id="u1", agent_id=None, user_message="hello"
        )
    ]
    assert any(e.type == "done" for e in collected)
    # SSE turn: thread keyed by conversation id, fresh run (not a resume).
    assert runner.calls[0]["thread_id"] == "c1"
    assert not runner.resumed
    stored = fakes.FakeMessageRepo.store["c1"]
    assert stored[0].role == "user" and stored[0].content == "hello"
    assert stored[-1].role == "assistant" and stored[-1].content == "final answer"


async def test_conversation_resume_skips_user_append(monkeypatch):
    monkeypatch.setattr(convo_mod, "ConversationRepository", fakes.FakeConversationRepo)
    monkeypatch.setattr(convo_mod, "MessageRepository", fakes.FakeMessageRepo)
    fakes.FakeMessageRepo.store.pop("cr", None)

    # A redelivered, resumable queue task: continue from the checkpoint, don't re-append.
    runner = fakes.FakeRunner([events.done("recovered")], resumable=True)
    svc = ConversationService(fakes.FakeDatabase(), runner, ttl_seconds=60)

    collected = [
        ev
        async for ev in svc.stream(
            conversation_id="cr",
            user_id="u1",
            agent_id=None,
            user_message="hello",
            mode="queue",
            task_id="task-cr",
        )
    ]
    assert any(e.type == "done" for e in collected)
    assert runner.resumed and runner.resumed[0]["thread_id"] == "task-cr"  # keyed by task_id
    assert not runner.calls  # fresh run path not taken
    # No user message persisted on resume (the checkpoint already has it); only the assistant.
    stored = fakes.FakeMessageRepo.store.get("cr", [])
    assert all(m.role != "user" for m in stored)
    assert stored and stored[-1].content == "recovered"


async def test_conversation_load_history_maps_turns(monkeypatch):
    monkeypatch.setattr(convo_mod, "MessageRepository", fakes.FakeMessageRepo)
    fakes.FakeMessageRepo.store["c2"] = [
        fakes.FakeMessage("user", "q1"),
        fakes.FakeMessage("assistant", "a1"),
    ]
    svc = ConversationService(fakes.FakeDatabase(), fakes.FakeRunner([]), ttl_seconds=60)
    history = await svc.load_history("c2")
    assert [m.content for m in history] == ["q1", "a1"]


async def test_conversation_end_sets_status_and_gc(monkeypatch):
    monkeypatch.setattr(convo_mod, "ConversationRepository", fakes.FakeConversationRepo)
    # end() imports EphemeralAgentRepository locally, so patch it at its source module.
    monkeypatch.setattr("hivemind.db.repository.EphemeralAgentRepository", fakes.FakeEphemeralRepo)
    svc = ConversationService(fakes.FakeDatabase(), fakes.FakeRunner([]), ttl_seconds=60)
    await svc.end("c3")
    assert fakes.FakeConversationRepo.statuses["c3"] == "ended"
    assert "c3" in fakes.FakeEphemeralRepo.deleted


# ---- TaskDispatcher --------------------------------------------------------


async def test_dispatcher_publishes_and_returns_task_id(monkeypatch):
    monkeypatch.setattr("hivemind.workers.dispatcher.TaskRepository", fakes.FakeTaskRepo)
    broker = fakes.FakeBroker()
    dispatcher = TaskDispatcher(fakes.FakeDatabase(), broker)
    task_id = await dispatcher.dispatch(
        conversation_id="c1", user_id="u1", agent_id=None, user_message="hi"
    )
    assert task_id
    assert broker.published[0]["conversation_id"] == "c1"


async def test_dispatcher_idempotent_skips_republish(monkeypatch):
    monkeypatch.setattr("hivemind.workers.dispatcher.TaskRepository", fakes.FakeTaskRepo)
    # Pre-seed a non-queued task so create_idempotent returns it as already-dispatched.
    fakes.FakeTaskRepo.tasks["t-existing"] = fakes.FakeTask("t-existing", status="running")

    class _Repo(fakes.FakeTaskRepo):
        async def create_idempotent(self, *a, **k):
            return fakes.FakeTaskRepo.tasks["t-existing"]

    monkeypatch.setattr("hivemind.workers.dispatcher.TaskRepository", _Repo)
    broker = fakes.FakeBroker()
    dispatcher = TaskDispatcher(fakes.FakeDatabase(), broker)
    task_id = await dispatcher.dispatch(
        conversation_id="c1", user_id="u1", agent_id=None, user_message="hi", idempotency_key="k"
    )
    assert task_id == "t-existing"
    assert broker.published == []  # not republished


# ---- CleanupScheduler ------------------------------------------------------


async def test_scheduler_run_once(monkeypatch):
    monkeypatch.setattr(sched_mod, "EphemeralAgentRepository", fakes.FakeEphemeralRepo)
    monkeypatch.setattr(sched_mod, "ConversationRepository", fakes.FakeConversationRepo)
    scheduler = CleanupScheduler(fakes.FakeDatabase(), interval_seconds=300)
    result = await scheduler.run_once()
    assert result["ephemeral_deleted"] == 2
    assert result["conversations_expired"] == 0


async def test_scheduler_gcs_checkpoints_for_expired_conversations(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(sched_mod, "EphemeralAgentRepository", fakes.FakeEphemeralRepo)
    monkeypatch.setattr(sched_mod, "ConversationRepository", fakes.FakeConversationRepo)
    monkeypatch.setattr(sched_mod, "TaskRepository", fakes.FakeTaskRepo)
    fakes.FakeConversationRepo.expired = [SimpleNamespace(id="cx")]
    fakes.FakeTaskRepo.tasks.clear()
    fakes.FakeTaskRepo.tasks["tx"] = fakes.FakeTask("tx", conversation_id="cx", status="completed")

    deleted: list[str] = []

    async def gc(thread_id: str) -> None:
        deleted.append(thread_id)

    scheduler = CleanupScheduler(fakes.FakeDatabase(), interval_seconds=300, checkpoint_gc=gc)
    try:
        result = await scheduler.run_once()
    finally:
        fakes.FakeConversationRepo.expired = []
        fakes.FakeTaskRepo.tasks.clear()

    assert deleted == ["tx"]  # the expired conversation's task checkpoint was GC'd
    assert result["checkpoints_deleted"] == 1
    assert result["conversations_expired"] == 1


async def test_runner_delete_checkpoint_is_best_effort():
    from hivemind.core.graph.runner import GraphRunner

    runner = GraphRunner.__new__(GraphRunner)  # skip graph compilation; we only test the wrapper

    class _Saver:
        deleted: list[str] = []

        async def adelete_thread(self, thread_id):
            _Saver.deleted.append(thread_id)

    runner._checkpointer = _Saver()
    await runner.delete_checkpoint("t-1")
    assert _Saver.deleted == ["t-1"]

    class _Boom:
        async def adelete_thread(self, thread_id):
            raise RuntimeError("saver down")

    runner._checkpointer = _Boom()
    await runner.delete_checkpoint("t-2")  # swallowed — never fatal to the caller


# ---- TaskEventBuffer -------------------------------------------------------


async def test_event_buffer_publish_and_replay(monkeypatch):
    monkeypatch.setattr(events_mod, "TaskEventRepository", _FakeTaskEventRepo)
    redis = fakes.FakeRedis()
    buffer = TaskEventBuffer(fakes.FakeDatabase(), redis)
    await buffer.publish("t1", 1, events.text_delta("a"))
    await buffer.publish("t1", 2, events.done("final"))

    # Replay reads the durable log (here, the fake) then tails the live stream.
    out = [ev async for _seq, ev in buffer.replay_and_tail("t1", after_seq=0)]
    types = [e.type for e in out]
    assert "text_delta" in types and "done" in types


class _FakeTaskEventRepo:
    """In-memory durable event log keyed by task_id, used by the replay path."""

    log: dict[str, list[tuple[int, str, dict]]] = {}

    def __init__(self, _session) -> None: ...

    async def append(self, task_id, seq, event_type, payload):
        _FakeTaskEventRepo.log.setdefault(task_id, []).append((seq, event_type, payload))

    async def replay(self, task_id, after_seq=0):
        from types import SimpleNamespace

        return [
            SimpleNamespace(seq=s, event_type=t, payload=p)
            for (s, t, p) in _FakeTaskEventRepo.log.get(task_id, [])
            if s > after_seq
        ]
