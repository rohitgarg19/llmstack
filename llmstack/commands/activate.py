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


def _hook_path(shell: str) -> Path:
    """``~/.<shell>_llmstack_hook`` -- ``pwsh`` is normalised to ``powershell``
    so the user doesn't end up with two redundant files."""
    name = "powershell" if shell in ("powershell", "pwsh") else shell
    return Path.home() / f".{name}_llmstack_hook"


def _source_line(shell: str, path: Path) -> str:
    """Shell-specific incantation to load the hook file."""
    if shell in ("powershell", "pwsh"):
        return f". '{path}'"
    return f'source "{path}"'


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

    eval_line = f'eval "$(llmstack activate {shell})"'
    print(f"[OK] hook written: {path}", file=sys.stderr)
    print( "     activate in this shell now (and for every new shell:", file=sys.stderr)
    print(f"     paste into your rc):  {eval_line}", file=sys.stderr)

    print(src)
    return 0
