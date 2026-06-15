"""gRPC code-execution service.

A thin server that exposes the in-process ``Sandbox`` contract over gRPC: each ``Execute``
RPC runs the submitted code via a locally-built sandbox and writes any generated files to the
request's ``artifact_dir`` — a path on a volume **shared** with the caller, so results travel
over the mount, not the wire. Runs as its own deployment (typically on a hardened/sandboxed
node pool); the HiveMind worker points at it via the ``grpc`` sandbox backend.

Transport is currently insecure (cluster-internal). It's structured so mTLS can be added by
swapping ``add_insecure_port`` for ``add_secure_port`` with channel credentials.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import grpc

from hivemind.config import Settings, get_settings
from hivemind.core.tools.sandbox import build_sandbox
from hivemind.core.tools.sandbox.base import Sandbox
from hivemind.executor.proto import executor_pb2, executor_pb2_grpc
from hivemind.observability.logging import configure_logging, get_logger

logger = get_logger("hivemind.executor")

_DEFAULT_TIMEOUT_S = 30


class CodeExecutorServicer(executor_pb2_grpc.CodeExecutorServicer):
    """Runs code through an injected ``Sandbox`` (so it's testable without a real server)."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def Execute(
        self, request: executor_pb2.ExecuteRequest, context: grpc.aio.ServicerContext
    ) -> executor_pb2.ExecuteResponse:
        timeout_s = request.timeout_s or _DEFAULT_TIMEOUT_S
        try:
            result = await self._sandbox.run(
                request.code,
                artifact_dir=Path(request.artifact_dir),
                timeout_s=timeout_s,
            )
            return executor_pb2.ExecuteResponse(
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
            )
        except Exception as exc:
            # Infra-level failure (sandbox couldn't run at all). Code-level errors come back as
            # a normal response with a non-zero exit_code, not an exception.
            logger.error("executor.run_failed", error=str(exc))
            await context.abort(grpc.StatusCode.INTERNAL, f"execution failed: {exc}")
            raise  # unreachable: abort() raises — present so the type-checker sees no fallthrough


def build_executor_sandbox(settings: Settings) -> Sandbox:
    """Build the sandbox the executor runs code in. Guards against pointing it back at itself."""
    if settings.sandbox_backend == "grpc":
        raise RuntimeError(
            "The executor service must not use SANDBOX_BACKEND=grpc (it would call itself). "
            "Set its SANDBOX_BACKEND to subprocess (default), docker, or microsandbox."
        )
    return build_sandbox(settings)


async def serve(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.environment != "local")
    sandbox = build_executor_sandbox(settings)

    server = grpc.aio.server()
    executor_pb2_grpc.add_CodeExecutorServicer_to_server(CodeExecutorServicer(sandbox), server)
    address = f"[::]:{settings.grpc_executor_port}"
    server.add_insecure_port(address)  # mTLS-ready: swap for add_secure_port(creds) later
    await server.start()
    logger.info("executor.started", address=address, sandbox_backend=settings.sandbox_backend)
    await server.wait_for_termination()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
