"""Shared helpers for the start/stop/status commands.

Just process lifecycle plumbing -- pid files, port probes, daemon
spawning, kill-by-pattern. Kept separate from the command modules so the
control-flow per command stays readable. The actual platform-specific
process bits (POSIX signals vs Windows ``taskkill`` etc.) live in
:mod:`llmstack._platform` so this module stays portable.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from llmstack._platform import (
    describe_matching,
    detached_popen,
    find_pids,
    kill_matching,
    pid_alive,
    terminate_pid,
)


def is_running(pid_file: Path) -> bool:
    """``True`` iff ``pid_file`` exists and points at a live process."""
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False
    if pid <= 0:
        return False
    return pid_alive(pid)


def read_pid(pid_file: Path) -> int | None:
    if not pid_file.is_file():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def port_responds(url: str, *, timeout: float = 2.0) -> bool:
    """Probe ``url`` for any 2xx response. Used to detect external daemons."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def spawn_daemon(
    argv: list[str],
    *,
    log: Path,
    pid_file: Path,
    env: dict[str, str] | None = None,
) -> int:
    """Spawn ``argv`` detached, redirect stdio to ``log``, write the pid."""
    log.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    fp = log.open("ab")
    proc = detached_popen(argv, stdout=fp, stderr=fp, env=env)
    fp.close()
    pid_file.write_text(f"{proc.pid}\n")
    return proc.pid


def kill_pid(pid: int, *, grace: float = 5.0) -> None:
    """SIGTERM (or taskkill), wait up to ``grace`` seconds, then hard-kill."""
    terminate_pid(pid, grace=grace)


def pgrep(pattern: str) -> list[int]:
    """Return PIDs whose full command-line matches ``pattern``."""
    return find_pids(pattern)


def pkill(pattern: str, *, grace: float = 5.0) -> int:
    """Terminate every process matching ``pattern``."""
    return kill_matching(pattern, grace=grace)


def pgrep_describe(pattern: str) -> str:
    """``pgrep -af``-style multi-line summary (empty when nothing matches)."""
    return describe_matching(pattern)
