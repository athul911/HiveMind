"""Artifact downloads: path-based ids, jail enforcement, and owner-authenticated endpoint."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from hivemind.api.routes import artifacts as artifacts_route
from hivemind.config import Settings
from hivemind.main import create_app
from hivemind.services.artifact_store import ArtifactStore
from httpx import ASGITransport, AsyncClient

from tests import fakes

# ---- store unit tests ----


def test_store_emits_download_url(tmp_path):
    store = ArtifactStore(str(tmp_path), public_base_url="http://host:8000/")
    ref = store.write_text("c1", "t1", "code_exec", "out.txt", "hello")
    d = ref.to_dict()
    assert d["download_url"].startswith("http://host:8000/v1/artifacts/")
    artifact_id = d["download_url"].rsplit("/", 1)[-1]
    assert store.resolve_id(artifact_id).read_bytes() == b"hello"


def test_store_no_public_url_means_no_url(tmp_path):
    store = ArtifactStore(str(tmp_path))
    ref = store.write_text("c1", "t1", "code_exec", "out.txt", "hello")
    assert "download_url" not in ref.to_dict()


def test_resolve_id_rejects_traversal(tmp_path):
    store = ArtifactStore(str(tmp_path), public_base_url="http://h")
    evil = base64.urlsafe_b64encode(b"../../etc/passwd").decode().rstrip("=")
    with pytest.raises(ValueError):
        store.resolve_id(evil)


def test_owner_conversation_id_is_namespace_root(tmp_path):
    store = ArtifactStore(str(tmp_path), public_base_url="http://h")
    ref = store.write_text("conv-7", "t1", "code_exec", "out.txt", "x")
    assert store.owner_conversation_id(Path(ref.path)) == "conv-7"


# ---- endpoint tests ----


class _FakeConvoRepo:
    owner = "local-dev"
    missing = False

    def __init__(self, _session) -> None: ...

    async def get(self, conversation_id):
        if _FakeConvoRepo.missing:
            return None
        return SimpleNamespace(id=conversation_id, user_id=_FakeConvoRepo.owner, status="active")


def _build(tmp_path, monkeypatch, *, auth_disabled=True):
    settings = Settings(
        auth_disabled=auth_disabled,
        environment="test",
        otel_enabled=False,
        artifact_base_path=str(tmp_path),
        public_base_url="http://test",
    )
    store = ArtifactStore(str(tmp_path), public_base_url="http://test")
    ref = store.write_text("conv-7", "t1", "code_exec", "report.txt", "the answer is 42")
    artifact_id = ref.to_dict()["download_url"].rsplit("/", 1)[-1]
    monkeypatch.setattr(artifacts_route, "ConversationRepository", _FakeConvoRepo)
    app = create_app(settings)
    app.state.context = SimpleNamespace(artifacts=store, settings=settings, db=fakes.FakeDatabase())
    return app, artifact_id


async def test_download_serves_file_for_owner(tmp_path, monkeypatch):
    _FakeConvoRepo.owner, _FakeConvoRepo.missing = "local-dev", False
    app, artifact_id = _build(tmp_path, monkeypatch)
    bad = base64.urlsafe_b64encode(b"../../etc/passwd").decode().rstrip("=")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ok = await ac.get(f"/v1/artifacts/{artifact_id}")
        missing = await ac.get(f"/v1/artifacts/{bad}")
    assert ok.status_code == 200
    assert ok.text == "the answer is 42"
    assert 'filename="report.txt"' in ok.headers.get("content-disposition", "")
    assert missing.status_code == 404  # traversal / malformed id


async def test_download_denied_for_non_owner(tmp_path, monkeypatch):
    _FakeConvoRepo.owner, _FakeConvoRepo.missing = "someone-else", False
    app, artifact_id = _build(tmp_path, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/v1/artifacts/{artifact_id}")
    assert resp.status_code == 403


async def test_download_denied_when_owner_unknown(tmp_path, monkeypatch):
    _FakeConvoRepo.owner, _FakeConvoRepo.missing = "local-dev", True  # no conversation row
    app, artifact_id = _build(tmp_path, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/v1/artifacts/{artifact_id}")
    assert resp.status_code == 403


async def test_download_requires_bearer_when_auth_enabled(tmp_path, monkeypatch):
    app, artifact_id = _build(tmp_path, monkeypatch, auth_disabled=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/v1/artifacts/{artifact_id}")
    assert resp.status_code == 401  # no longer a public endpoint
