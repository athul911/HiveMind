"""Docker-backed sandbox — one ephemeral, locked-down container per execution.

Isolation properties:
  * ``network_mode=none``      — no network egress.
  * ``read_only=True``         — read-only root filesystem.
  * ``cap_drop=["ALL"]``       — no Linux capabilities.
  * ``pids_limit`` / ``mem_limit`` / ``nano_cpus`` — resource caps.
  * non-root user, ``no-new-privileges`` security opt.
  * tmpfs ``/tmp``, artifact dir bind-mounted read-write at ``/artifacts``.
  * hard wall-clock kill via timeout.

The container is created, run to completion (or killed on timeout), and removed.
Docker calls are blocking, so they run in a worker thread to keep the event loop free.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from hivemind.core.errors import SandboxError
from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.sandbox.docker")

_RUNNER = (
    "import runpy, sys\n"
    "sys.argv = ['user_code']\n"
    "runpy.run_path('/artifacts/_code.py', run_name='__main__')\n"
)


class DockerSandbox:
    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        memory: str = "256m",
        cpus: float = 1.0,
        pids_limit: int = 128,
    ) -> None:
        self._image = image
        self._memory = memory
        self._nano_cpus = int(cpus * 1_000_000_000)
        self._pids_limit = pids_limit

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        return await asyncio.to_thread(self._run_sync, code, artifact_dir, timeout_s)

    def _run_sync(self, code: str, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        import docker  # imported lazily so the package isn't required at import time
        from docker.errors import ContainerError, ImageNotFound

        client = docker.from_env()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "_code.py").write_text(code, encoding="utf-8")

        try:
            container = client.containers.run(
                self._image,
                command=["python", "-c", _RUNNER],
                detach=True,
                network_mode="none",
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                user="65534:65534",  # nobody
                mem_limit=self._memory,
                nano_cpus=self._nano_cpus,
                pids_limit=self._pids_limit,
                tmpfs={"/tmp": "size=64m,exec"},
                volumes={str(artifact_dir): {"bind": "/artifacts", "mode": "rw"}},
                working_dir="/artifacts",
                environment={"PYTHONUNBUFFERED": "1", "HOME": "/tmp"},
            )
        except ImageNotFound as exc:
            raise SandboxError(f"Sandbox image not available: {self._image}") from exc
        except ContainerError as exc:  # pragma: no cover - defensive
            raise SandboxError(f"Container failed to start: {exc}") from exc

        timed_out = False
        try:
            result = container.wait(timeout=timeout_s)
            exit_code = int(result.get("StatusCode", 1))
        except Exception:
            timed_out = True
            exit_code = 124
            with _suppress():
                container.kill()
        finally:
            stdout = _safe_logs(container, stdout=True)
            stderr = _safe_logs(container, stdout=False)
            with _suppress():
                container.remove(force=True)

        return SandboxResult(exit_code=exit_code, stdout=stdout, stderr=stderr, timed_out=timed_out)


def _safe_logs(container, *, stdout: bool) -> str:
    try:
        return container.logs(stdout=stdout, stderr=not stdout).decode("utf-8", "replace")
    except Exception:
        return ""


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
