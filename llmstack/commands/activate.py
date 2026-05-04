"""``llmstack activate <zsh|bash>`` -- emit the auto-activation rc snippet."""

from __future__ import annotations

import sys

from llmstack.shell_env import activate_hook


def _print_help() -> None:
    print("usage: llmstack activate <zsh|bash>")


def run(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0
    shell = args[0]
    sys.stdout.write(activate_hook(shell))
    return 0
