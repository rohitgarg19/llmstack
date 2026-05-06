"""``llmstack reload`` -- refresh env + prompt of the current shell.

The activate hook normally re-evaluates env and ``PROMPT`` / ``PS1`` on
every chpwd, so cd-ing into a project (or switching between projects)
keeps things current. But mid-session events -- ``llmstack start --next``
inside an already-active shell, the channel marker getting rewritten by
another process, etc. -- don't trigger a chpwd, so the prompt would stay
stale until the next directory change.

Pipe this command's output through your shell's eval to apply the
current channel's env + prompt in-place::

    # zsh / bash
    eval "$(llmstack reload)"

    # powershell
    Invoke-Expression (& llmstack reload | Out-String)

We resolve the channel from ``.llmstack/active-channel`` (live, written
by ``start``), falling back to ``.llmstack/default-channel`` (intent,
written by ``install``), and finally to ``current``. All informational
output goes to stderr so stdout stays eval-safe.
"""

from __future__ import annotations

import sys

from llmstack.paths import read_marker, resolve
from llmstack.shell_env import emit_shell_refresh


def _print_help() -> None:
    print('usage: eval "$(llmstack reload)"', file=sys.stderr)


def run(args: list[str]) -> int:
    for a in args:
        if a in ("-h", "--help"):
            _print_help()
            return 0
        print(f"[!] unknown arg to reload: {a}", file=sys.stderr)
        return 2

    paths = resolve()
    if not paths.opencode_json.is_file():
        # No project -- emit nothing on stdout (eval no-op) and a hint
        # on stderr. We don't fail-hard because users may have their
        # rc wired to call reload defensively.
        print(
            f"[!] no .llmstack/opencode.json under {paths.work_dir} -- nothing to reload.",
            file=sys.stderr,
        )
        return 0

    mark = read_marker(paths.active_marker) or read_marker(paths.default_marker)
    channel = mark.channel if mark else "current"
    emit_shell_refresh(channel)
    return 0
