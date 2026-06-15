"""gRPC code-executor: real in-process server + GrpcSandbox client over the wire.

Uses a fake in-executor Sandbox (deterministic across platforms) so these tests exercise the
gRPC plumbing — proto round-trip, request/response mapping, and the shared-dir artifact write
— rather than the subprocess backend (covered elsewhere).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import grpc
import pytest
from hivemind.config import Settings
from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.core.tools.sandbox.grpc_sandbox import GrpcSandbox
from hivemind.executor.proto import executor_pb2_grpc
from hivemind.executor.server import CodeExecutorServicer, build_executor_sandbox


class _FakeSandbox:
    """Records the run() call and optionally writes a file into the shared artifact_dir."""

    def __init__(self, result: SandboxResult, *, write: tuple[str, str] | None = None) -> None:
        self._result = result
        self._write = write
        self.calls: list[dict] = []

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        self.calls.append({"code": code, "artifact_dir": artifact_dir, "timeout_s": timeout_s})
        if self._write is not None:
            name, content = self._write
            (artifact_dir / name).write_text(content)
        return self._result


@asynccontextmanager
async def _running_executor(sandbox):
    server = grpc.aio.server()
    executor_pb2_grpc.add_CodeExecutorServicer_to_server(CodeExecutorServicer(sandbox), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        await server.stop(grace=None)


async def test_grpc_execute_roundtrip_and_shared_artifact(tmp_path):
    fake = _FakeSandbox(SandboxResult(0, "done\n", "", False), write=("out.txt", "hello"))
    async with _running_executor(fake) as target:
        client = GrpcSandbox(target=target, deadline_margin_s=5)
        res = await client.run("print('done')", artifact_dir=tmp_path, timeout_s=12)

    # Response mapped back into a SandboxResult.
    assert res.exit_code == 0 and res.stdout == "done\n" and not res.timed_out
    # The executor wrote to the shared artifact_dir; the caller reads it from the same path.
    assert (tmp_path / "out.txt").read_text() == "hello"
    # Request fields crossed correctly.
    assert fake.calls[0]["code"] == "print('done')"
    assert fake.calls[0]["artifact_dir"] == tmp_path
    assert fake.calls[0]["timeout_s"] == 12


async def test_grpc_propagates_timed_out(tmp_path):
    fake = _FakeSandbox(SandboxResult(124, "", "timed out", True))
    async with _running_executor(fake) as target:
        client = GrpcSandbox(target=target, deadline_margin_s=5)
        res = await client.run("while True: pass", artifact_dir=tmp_path, timeout_s=1)
    assert res.timed_out and res.exit_code == 124


def test_executor_refuses_grpc_backend():
    # The executor must not point its internal sandbox back at itself.
    with pytest.raises(RuntimeError, match="must not use SANDBOX_BACKEND=grpc"):
        build_executor_sandbox(Settings(sandbox_backend="grpc"))
