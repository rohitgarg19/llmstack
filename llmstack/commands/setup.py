"""``llmstack setup`` -- first-time walkthrough.

Mirrors the shell ``cmd_setup``:

  1. Kick off GGUF downloads in the background (skip with ``--skip-download``).
  2. Wait for them to finish (skip with ``--skip-wait``).
  3. Install the ``llama-swap`` binary.
  4. Print the shell activation hook eval line + the auto-detected hook.
  5. Verify ``opencode`` is on PATH.

Does NOT run ``install`` or ``start`` -- those are separate steps once
downloads finish.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from llmstack.commands import activate as activate_cmd
from llmstack.download.binary import install_llama_swap
from llmstack.download.ggufs import download_all, wait_for_downloads
from llmstack.paths import is_remote, remote_url
from llmstack.shell_env import _user_shell


def _print_help() -> None:
    print("usage: llmstack setup [--skip-download] [--skip-wait]")


def run(args: list[str]) -> int:
    skip_download = False
    skip_wait = False
    for arg in args:
        if arg == "--skip-download":
            skip_download = True
        elif arg == "--skip-wait":
            skip_wait = True
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            print(f"[!] unknown arg to setup: {arg} (try --skip-download, --skip-wait, -h)")
            return 2

    if is_remote():
        print(f"[!] LLMSTACK_REMOTE_URL is set ({remote_url()}); setup is local-only.")
        print("    in client mode you only need:")
        print("      llmstack install      # generate .llmstack/opencode.json (points at remote)")
        print("      llmstack start        # verify remote + enter the client subshell")
        return 1

    if not skip_download:
        print("[1/3] downloading required GGUFs...")
        download_all()
        print()
    else:
        print("[1/3] (skipped) downloads")
        print()

    if not skip_download and not skip_wait:
        print("[2/3] waiting for downloads to finish...")
        wait_for_downloads()
        print()
    else:
        print("[2/3] (skipped) wait")
        print()

    _, shell_name = _user_shell()

    print("[3/3] installing llama-swap binary...")
    install_llama_swap()
    print()

    print("[4/4] wiring shell activation hook...")
    print()
    print(f"Add the following line to your ~/{shell_name}rc to auto-activate")
    print("llmstack whenever you cd into a project with .llmstack/:")
    print()
    print(f'    eval "$(llmstack activate {shell_name})"')
    print()
    activate_cmd.run([shell_name])

    print()
    print("[5/5] checking opencode...")
    if shutil.which("opencode"):
        path = shutil.which("opencode") or "opencode"
        print(f"[OK] opencode found: {path}")
        try:
            ver = subprocess.run(
                ["opencode", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            ver = "(unknown)"
        print(f"     version: {ver or '(unknown)'}")
    else:
        print("[!] opencode not found in PATH.")
        print()
        print("Install it with:")
        print("  curl -fsSL https://opencode.ai/install | sh")
        print("  # or via npm:")
        print("  npm install -g opencode-ai")
        print()
        print("After installing, run:")
        print("  llmstack install     # generate configs for this project")
        print("  llmstack start       # bring up the stack")
        return 0

    print()
    print("[OK] setup complete.")
    print()
    print("Next steps:")
    print(f"  1. Add the eval line above to your ~/{shell_name}rc (one time)")
    print("  2. llmstack install     # generate .llmstack/ configs for this project")
    print("  3. llmstack start       # bring up the stack + enter the shell")
    print()
    print("To check configured GGUFs + drift vs models.ini:")
    print("  llmstack check")
    return 0
