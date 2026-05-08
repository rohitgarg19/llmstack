"""``llmstack activate <shell>`` -- install + source the auto-activation hook.

Writes the hook to ``~/.<shell>_llmstack_hook`` and prints the matching
``source`` line to **stdout** so a one-shot

    eval "$(llmstack activate zsh)"

both regenerates the file and turns on the hook in the current shell.
Pasting the same line into your shell rc keeps it on for every new
shell. All informational output goes to stderr so it doesn't get
captured by ``eval``.

Once the hook is installed, ``cd`` into any project with ``.llmstack/``
and the env (``OPENCODE_CONFIG``, ``LLMSTACK_WORK_DIR``,
``LLMSTACK_CHANNEL``) is set up automatically -- there is no separate
``llmstack shell`` action.
"""

from __future__ import annotations

import sys
from pathlib import Path

from llmstack.shell_env import activate_hook


def _print_help() -> None:
    print("usage: llmstack activate <zsh|bash|powershell>", file=sys.stderr)


def _is_powershell(shell: str) -> bool:
    return shell in ("powershell", "pwsh")


def _hook_path(shell: str) -> Path:
    """``~/.<shell>_llmstack_hook`` -- ``pwsh`` is normalised to ``powershell``
    so the user doesn't end up with two redundant files.

    PowerShell additionally needs a ``.ps1`` suffix or the host won't
    dot-source it -- without the extension Windows hands the file to
    the OS shell file-association (Notepad, etc.) instead of running
    it as a script.
    """
    if _is_powershell(shell):
        return Path.home() / ".powershell_llmstack_hook.ps1"
    return Path.home() / f".{shell}_llmstack_hook"


def _source_line(shell: str, path: Path) -> str:
    """Shell-specific incantation to load the hook file."""
    if _is_powershell(shell):
        return f". '{path}'"
    return f'source "{path}"'


def eval_line(shell: str) -> str:
    """The one-shot the user pastes / adds to their rc to install the hook.

    POSIX shells use ``eval "$(...)"``; PowerShell has no ``eval`` and
    needs ``Invoke-Expression`` over the captured stdout.
    """
    if _is_powershell(shell):
        return f"llmstack activate {shell} | Out-String | Invoke-Expression"
    return f'eval "$(llmstack activate {shell})"'


def write_hook(shell: str) -> tuple[Path, str]:
    """Render the hook for ``shell``, write it to disk, return ``(path, source_line)``.

    Shared by ``llmstack activate`` (CLI surface) and ``llmstack setup``
    (first-run walkthrough) so they install the hook the same way.
    """
    body = activate_hook(shell)  # raises SystemExit on unknown shell
    path = _hook_path(shell)
    path.write_text(body)
    return path, _source_line(shell, path)


def run(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0
    shell = args[0]

    path, src = write_hook(shell)

    line = eval_line(shell)
    print(f"[OK] hook written: {path}", file=sys.stderr)
    print( "     activate in this shell now (and for every new shell:", file=sys.stderr)
    print(f"     paste into your rc):  {line}", file=sys.stderr)
    if _is_powershell(shell):
        # PowerShell's default `Restricted` policy on Windows blocks
        # dot-sourcing any .ps1; surface the one-time fix so the
        # `Invoke-Expression` line above doesn't fail with "running
        # scripts is disabled on this system".
        print(
            "     PowerShell execution policy must allow local scripts; "
            "if dot-sourcing fails, run once:",
            file=sys.stderr,
        )
        print(
            "         Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned",
            file=sys.stderr,
        )

    print(src)
    return 0
