"""gRPC-backed sandbox — delegates execution to a remote code-executor service.

Implements the same ``Sandbox`` protocol as the in-process backends, but forwards the run to
the executor over gRPC. Results (files) are written by the executor to ``artifact_dir`` on a
**shared mount**, so this side reads them back from the same path — no bytes cross the wire,
only the small ``SandboxResult`` (exit code, stdout/stderr, timed_out).

Insecure (plaintext) channel for cluster-internal use; mTLS-ready (swap ``insecure_channel``
for ``secure_channel`` with credentials).
"""

from __future__ import annotations

from pathlib import Path

import grpc

from hivemind.core.errors import SandboxError
from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.executor.proto import executor_pb2, executor_pb2_grpc
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.sandbox.grpc")


class GrpcSandbox:
    def __init__(self, *, target: str, deadline_margin_s: int = 10) -> None:
        self._target = target
        self._margin = deadline_margin_s

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        # Ensure the shared output dir exists before the executor writes into it.
        artifact_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        request = executor_pb2.ExecuteRequest(
            code=code, artifact_dir=str(artifact_dir), timeout_s=timeout_s
        )
        # Client deadline = the execution cap plus a margin for transport/boot, so the server's
        # own timeout (which yields a clean timed_out result) fires first in the normal case.
        deadline = timeout_s + self._margin
        try:
            async with grpc.aio.insecure_channel(self._target) as channel:
                stub = executor_pb2_grpc.CodeExecutorStub(channel)
                resp = await stub.Execute(request, timeout=deadline)
        except grpc.aio.AioRpcError as exc:
            if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                # The whole RPC outran the deadline (server didn't return its own timed_out).
                return SandboxResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)
            logger.error("sandbox.grpc_failed", code=exc.code().name, detail=exc.details())
            raise SandboxError(
                f"Code executor RPC failed ({exc.code().name}): {exc.details()}"
            ) from exc
        return SandboxResult(
            exit_code=resp.exit_code,
            stdout=resp.stdout,
            stderr=resp.stderr,
            timed_out=resp.timed_out,
        )
