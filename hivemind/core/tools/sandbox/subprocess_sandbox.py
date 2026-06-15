"""Hardened-subprocess sandbox — the no-Docker / no-VM backend.

Two isolation levels, chosen by ``SUBPROCESS_ISOLATION``:

* ``none`` (default): a plain child process with RLIMIT caps (CPU, address space, open files,
  processes), a scrubbed environment, the cwd set to the artifact dir, ``python -I``, its own
  session (``setsid``), and a wall-clock timeout. Shares the host kernel **and** filesystem
  view and has network access — weak isolation, for trusted single-tenant use (laptops, CI).

* ``namespaces``: wraps the same execution in **bubblewrap** (``bwrap``) — a new network
  namespace (no egress), a mount namespace with the filesystem **jailed to the artifact dir**
  over a read-only runtime, plus PID/IPC/UTS/user namespaces. This is container-grade
  (shared-kernel) isolation with no daemon and no ``/dev/kvm``. It requires ``bwrap`` and
  unprivileged user namespaces; when those aren't available (e.g. macOS, or a pod that blocks
  userns) it **falls back** to the plain path with a one-time warning, so it's safe to enable
  everywhere. RLIMITs are still applied (they're inherited across the ``bwrap`` exec).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

from hivemind.config import SubprocessIsolation
from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.sandbox.subprocess")

# Read-only host paths exposed inside the bwrap sandbox so the interpreter + stdlib + shared
# libs resolve. Tuned for python:3.11-slim (python lives under /usr/local, i.e. within /usr).
# Deliberately excludes /etc so mounted secrets aren't exposed. ``--ro-bind-try`` skips any
# that don't exist on a given base image.
_DEFAULT_RO_BINDS = ("/usr", "/lib", "/lib64", "/bin", "/sbin")
_GUEST_DIR = "/artifacts"


def _preexec(mem_bytes: int, cpu_seconds: int) -> None:  # pragma: no cover - child process
    # Each limit is best-effort: a preexec_fn that raises aborts the whole spawn, and some
    # limits aren't honored on every platform (notably RLIMIT_AS on macOS, which counts
    # mapped shared libraries and would kill the interpreter at startup). Apply what works.
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
    def __init__(
        self,
        *,
        memory_bytes: int = 256 * 1024 * 1024,
        isolation: SubprocessIsolation = "none",
        ro_binds: tuple[str, ...] = _DEFAULT_RO_BINDS,
    ) -> None:
        self._memory_bytes = memory_bytes
        self._isolation = isolation
        self._ro_binds = ro_binds
        self._bwrap_ok: bool | None = None  # cached capability probe (namespaces mode only)
        if isolation != "namespaces":
            logger.warning(
                "sandbox.weak_isolation",
                detail="SubprocessSandbox (isolation=none) is weaker than the Docker backend.",
            )

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        # Tiny, one-time local FS prep before spawning the child; sync Path is fine here.
        artifact_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        code_path = artifact_dir / "_code.py"
        code_path.write_text(code, encoding="utf-8")

        argv = await self._argv(code_path, artifact_dir)
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
            *argv,
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

    async def _argv(self, code_path: Path, artifact_dir: Path) -> list[str]:
        """The command to spawn: bubblewrap-wrapped when isolation is on and available."""
        plain = [sys.executable, "-I", str(code_path)]
        if self._isolation != "namespaces":
            return plain
        if not await self._bubblewrap_available():
            return plain  # graceful fallback (logged once in the probe)
        return self._bwrap_argv(artifact_dir)

    def _bwrap_argv(self, artifact_dir: Path) -> list[str]:
        """bubblewrap invocation: no network, FS jailed to the artifact dir, unprivileged."""
        cmd = [
            "bwrap",
            "--unshare-all",  # net + pid + ipc + uts + user + cgroup; NO --share-net => no network
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--setenv", "HOME", "/tmp",
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "PYTHONUNBUFFERED", "1",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
        ]
        for path in self._ro_binds:
            cmd += ["--ro-bind-try", path, path]
        cmd += [
            "--bind", str(artifact_dir), _GUEST_DIR,  # only this dir is writable
            "--chdir", _GUEST_DIR,
            "--",
            sys.executable, "-I", f"{_GUEST_DIR}/_code.py",
        ]
        return cmd

    async def _bubblewrap_available(self) -> bool:
        """Probe (once) whether bwrap + unprivileged userns actually work here; cache it."""
        if self._bwrap_ok is not None:
            return self._bwrap_ok
        ok = False
        if shutil.which("bwrap"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "bwrap", "--unshare-all", "--die-with-parent",
                    "--ro-bind", "/usr", "/usr", "--proc", "/proc", "--dev", "/dev",
                    sys.executable, "-I", "-c", "pass",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                ok = (await asyncio.wait_for(proc.wait(), timeout=10)) == 0
            except Exception:
                ok = False
        if not ok:
            logger.warning(
                "sandbox.bwrap_unavailable",
                detail="bubblewrap/userns not usable; falling back to plain subprocess "
                "(no namespace isolation).",
            )
        self._bwrap_ok = ok
        return ok


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
