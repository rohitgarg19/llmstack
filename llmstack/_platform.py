"""Tiny cross-OS shim for the bits of the stack that touch the kernel.

Everything platform-specific (process lifecycle, detached daemon spawn,
default shell, executable suffix) lives behind this single module so the
rest of the package can stay portable. Three supported families:

  * **macOS / Linux / FreeBSD** -- POSIX. ``os.kill`` for liveness +
    SIGTERM/SIGKILL, ``pgrep -f`` for pattern lookup, ``start_new_session``
    for detached spawn.
  * **Windows** -- ``OpenProcess`` for liveness, ``taskkill /T /F`` for
    pattern + tree kill, WMI/``tasklist`` for command-line probing,
    ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` flags for daemon spawn.
    ``signal.SIGKILL`` doesn't exist; SIGTERM == TerminateProcess.

The names exposed here mirror the POSIX ones so callers don't need to
care which branch runs.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"
IS_POSIX = not IS_WINDOWS

EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


# ---------------------------------------------------------------------------
# liveness + termination
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """True iff ``pid`` points at a live process the current user can see.

    POSIX: ``os.kill(pid, 0)`` raises ``ProcessLookupError`` for dead
    pids and ``PermissionError`` for someone else's process (still alive,
    just not ours -- treat as alive).

    Windows: ``os.kill(pid, 0)`` would dispatch a ``CTRL_C_EVENT``,
    which is wrong; we ``OpenProcess`` instead and ignore the handle.
    """
    if pid <= 0:
        return False
    if IS_WINDOWS:
        return _win_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # process exists, just not ours.
        return True
    except OSError:
        return False
    return True


def _win_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def terminate_pid(pid: int, *, grace: float = 5.0) -> None:
    """Best-effort terminate. SIGTERM (or taskkill), wait, SIGKILL on hold-out.

    Silent on already-dead / not-ours pids -- callers don't need to care.
    """
    if pid <= 0 or not pid_alive(pid):
        return
    if IS_WINDOWS:
        _win_terminate(pid, grace=grace)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    waited = 0.0
    while waited < grace:
        if not pid_alive(pid):
            return
        time.sleep(0.5)
        waited += 0.5
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _win_terminate(pid: int, *, grace: float) -> None:
    """``taskkill /T /F`` -- /T includes any children; /F is the hard kill.

    We send a graceful taskkill first (no /F) and only escalate after
    ``grace`` seconds, mirroring the POSIX SIGTERM-then-SIGKILL flow.
    """
    if not shutil.which("taskkill"):
        return
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    waited = 0.0
    while waited < grace:
        if not pid_alive(pid):
            return
        time.sleep(0.5)
        waited += 0.5
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# pattern-based process lookup
# ---------------------------------------------------------------------------

def find_pids(pattern: str) -> list[int]:
    """PIDs whose full command line matches ``pattern`` (POSIX regex).

    Returns ``[]`` when the underlying lookup tool isn't available; the
    caller is expected to treat that as "no matches" rather than an
    error -- this is best-effort housekeeping, not load-bearing.
    """
    if IS_WINDOWS:
        return [pid for pid, _ in _win_proc_list_matching(pattern)]
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


def find_processes(pattern: str) -> list[tuple[int, str]]:
    """``[(pid, cmdline)]`` for every process whose command line matches.

    POSIX: ``pgrep -af`` (full cmdline + pid). Windows: WMIC / PowerShell
    Get-CimInstance, falling back to ``tasklist`` (which only knows the
    image name, not the full cmdline).
    """
    if IS_WINDOWS:
        return _win_proc_list_matching(pattern)
    if not shutil.which("pgrep"):
        return []
    try:
        proc = subprocess.run(
            ["pgrep", "-af", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode not in (0, 1):
        return []
    out: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        line = line.rstrip()
        if not line:
            continue
        head, _, tail = line.partition(" ")
        if not head.isdigit():
            continue
        out.append((int(head), tail))
    return out


def kill_matching(pattern: str, *, grace: float = 5.0) -> int:
    """Terminate every process whose cmdline matches ``pattern``."""
    n = 0
    for pid in find_pids(pattern):
        terminate_pid(pid, grace=grace)
        n += 1
    return n


def describe_matching(pattern: str) -> str:
    """``pgrep -af``-style multi-line string (empty when nothing matches)."""
    return "\n".join(f"{pid} {cmd}" for pid, cmd in find_processes(pattern))


# Windows process listing: prefer PowerShell's Get-CimInstance (richer
# output, present since PS 3.0), fall back to tasklist /v which only
# carries window title + image name.

def _win_proc_list_matching(pattern: str) -> list[tuple[int, str]]:
    rx = re.compile(pattern)
    rows = _win_proc_list_powershell() or _win_proc_list_tasklist()
    return [(pid, cmd) for pid, cmd in rows if rx.search(cmd)]


def _win_proc_list_powershell() -> list[tuple[int, str]] | None:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        return None
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId, CommandLine | "
        "ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    )
    try:
        proc = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", cmd],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    rows: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        head, _, tail = line.partition("|")
        head = head.strip()
        if head.isdigit():
            rows.append((int(head), tail.strip()))
    return rows


def _win_proc_list_tasklist() -> list[tuple[int, str]]:
    if not shutil.which("tasklist"):
        return []
    try:
        proc = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    import csv
    rows: list[tuple[int, str]] = []
    for fields in csv.reader(proc.stdout.splitlines()):
        if len(fields) < 2:
            continue
        image, pid_str = fields[0], fields[1]
        if pid_str.isdigit():
            rows.append((int(pid_str), image))
    return rows


# ---------------------------------------------------------------------------
# detached background spawn
# ---------------------------------------------------------------------------

def detached_popen(
    argv: list[str],
    *,
    stdout,
    stderr,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.Popen:
    """``subprocess.Popen`` with the right "detach from controlling tty" flags.

    POSIX: ``start_new_session=True`` (calls ``setsid``).
    Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` so closing the
    parent console doesn't drag the daemon down with it.
    """
    kw: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": stdout,
        "stderr": stderr,
        "env": env if env is not None else os.environ.copy(),
        "cwd": cwd,
    }
    if IS_WINDOWS:
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kw["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kw["close_fds"] = False
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(argv, **kw)


# ---------------------------------------------------------------------------
# user shell discovery
# ---------------------------------------------------------------------------

def default_shell() -> tuple[str, str]:
    """Return ``(absolute_path, basename_lower)`` for the shell to spawn.

    Resolution: ``$LLMSTACK_SHELL`` → POSIX ``$SHELL`` / Windows
    ``$ComSpec`` → hard-coded fallback.
    """
    raw = os.environ.get("LLMSTACK_SHELL")
    if not raw:
        if IS_WINDOWS:
            raw = (
                shutil.which("pwsh")
                or shutil.which("powershell")
                or os.environ.get("ComSpec")
                or "cmd.exe"
            )
        else:
            raw = os.environ.get("SHELL") or "/bin/bash"
    base = os.path.basename(raw).lower()
    if base.endswith(".exe"):
        base = base[: -len(".exe")]
    return raw, base


def shell_family(name: str) -> str:
    """Coarse classification: ``"powershell"`` / ``"cmd"`` / ``"bash"`` /
    ``"zsh"`` / ``"posix"`` (anything else POSIX-shaped)."""
    n = name.lower()
    if n in ("pwsh", "powershell", "powershell_ise"):
        return "powershell"
    if n in ("cmd",):
        return "cmd"
    if n in ("bash",):
        return "bash"
    if n in ("zsh",):
        return "zsh"
    return "posix"


# ---------------------------------------------------------------------------
# data-directory roots
# ---------------------------------------------------------------------------

def user_data_root() -> Path:
    """Where persistent per-user data (binaries, caches) should live.

    POSIX: respects ``$XDG_DATA_HOME`` then falls back to
    ``~/.local/share``.
    Windows: prefers ``$LOCALAPPDATA``, falling back to
    ``%USERPROFILE%/AppData/Local``.
    """
    if IS_WINDOWS:
        raw = os.environ.get("LOCALAPPDATA")
        if raw:
            return Path(raw)
        return Path.home() / "AppData" / "Local"
    raw = os.environ.get("XDG_DATA_HOME") or ""
    return Path(raw) if raw else Path.home() / ".local" / "share"


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def make_executable(path: Path) -> None:
    """Mark ``path`` runnable. No-op on Windows (file extension decides)."""
    if IS_WINDOWS:
        return
    try:
        path.chmod(0o755)
    except OSError:
        pass


__all__ = [
    "EXE_SUFFIX",
    "IS_POSIX",
    "IS_WINDOWS",
    "default_shell",
    "describe_matching",
    "detached_popen",
    "find_pids",
    "find_processes",
    "kill_matching",
    "make_executable",
    "pid_alive",
    "shell_family",
    "terminate_pid",
    "user_data_root",
]


# best-effort import-time sanity: warn (not fail) if signal.SIGKILL is
# missing on a non-Windows host -- shouldn't happen, but the rest of
# the module assumes it.
if IS_POSIX and not hasattr(signal, "SIGKILL"):  # pragma: no cover
    print("[!] llmstack: signal.SIGKILL missing on this POSIX system", file=sys.stderr)
