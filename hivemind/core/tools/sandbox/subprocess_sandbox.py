"""Hardened-subprocess sandbox — the no-Docker fallback.

Weaker isolation than the Docker backend (no network namespace, shared kernel/FS view),
explicitly flagged in logs and gated by ``SANDBOX_BACKEND=subprocess``. It still applies
RLIMIT caps (CPU, address space, open files, processes), scrubs the environment, jails the
working directory to the artifact dir, and enforces a wall-clock timeout. Use only on
trusted single-tenant deployments (laptops, CI).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.sandbox.subprocess")


def _preexec(mem_bytes: int, cpu_seconds: int) -> None:  # pragma: no cover - child process
    # Each limit is best-effort: a preexec_fn that raises aborts the whole spawn, and some
    # limits aren't honored on every platform (notably RLIMIT_AS on macOS, which counts
    # mapped shared libraries and would kill the interpreter at startup). Apply what works;
    # the Docker backend is the hard-isolation path for production.
    import contextlib
    import resource

    limits = [
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_NOFILE, (256, 256)),
        (resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
        (getattr(resource, "RLIMIT_NPROC", None), (64, 64)),
    ]
    for res, value in limits:
        if res is not None:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(res, value)
    with contextlib.suppress(OSError):
        os.setsid()


class SubprocessSandbox:
    def __init__(self, *, memory_bytes: int = 256 * 1024 * 1024) -> None:
        self._memory_bytes = memory_bytes
        logger.warning(
            "sandbox.weak_isolation",
            detail="SubprocessSandbox provides weaker isolation than the Docker backend.",
        )

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        # Tiny, one-time local FS prep before spawning the child; sync Path is fine here.
        artifact_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        code_path = artifact_dir / "_code.py"
        code_path.write_text(code, encoding="utf-8")

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(artifact_dir),
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(artifact_dir),
        }
        preexec = (
            None if sys.platform == "win32" else (lambda: _preexec(self._memory_bytes, timeout_s))
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(code_path),
            cwd=str(artifact_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            with _suppress():
                proc.kill()
            return SandboxResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)

        return SandboxResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
        )


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
