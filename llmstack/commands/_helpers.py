"""Shared helpers for the start/stop/status commands.

Just process lifecycle plumbing -- pid files, port probes, daemon
spawning, kill-by-pattern. Kept separate from the command modules so the
control-flow per command stays readable.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


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
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


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
    """Spawn ``argv`` in a detached session, redirect stdio to ``log``,
    write the pid to ``pid_file``, return the pid."""
    log.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    fp = log.open("ab")
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=fp,
        stderr=subprocess.STDOUT,
        env=env if env is not None else os.environ.copy(),
        start_new_session=True,
    )
    fp.close()
    pid_file.write_text(f"{proc.pid}\n")
    return proc.pid


def kill_pid(pid: int, *, grace: float = 5.0) -> None:
    """SIGTERM, wait up to ``grace`` seconds, then SIGKILL if still alive."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    waited = 0.0
    while waited < grace:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
        waited += 0.5
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def pgrep(pattern: str) -> list[int]:
    """Return PIDs whose full command-line matches ``pattern`` (regex).

    Returns ``[]`` when ``pgrep`` is unavailable or finds nothing.
    """
    if not shutil.which("pgrep"):
        return []
    try:
        proc = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode not in (0, 1):
        return []
    return [int(line) for line in proc.stdout.splitlines() if line.strip().isdigit()]


def pkill(pattern: str, *, grace: float = 5.0) -> int:
    """SIGTERM (then SIGKILL after ``grace``) every process matching ``pattern``."""
    killed = 0
    for pid in pgrep(pattern):
        kill_pid(pid, grace=grace)
        killed += 1
    return killed


def pgrep_describe(pattern: str) -> str:
    """``pgrep -af`` output as a single string (empty when nothing matches)."""
    if not shutil.which("pgrep"):
        return ""
    try:
        proc = subprocess.run(
            ["pgrep", "-af", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode in (0, 1) else ""
