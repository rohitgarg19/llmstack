"""``llmstack install-llama-swap`` -- (re-)download the llama-swap binary only.

Exposes the previously-internal ``_install_llama_swap`` helper as a
first-class subcommand. Useful for ``--force`` upgrades without touching
the generated configs.
"""

from __future__ import annotations

from llmstack.download.binary import install_llama_swap
from llmstack.paths import is_remote, remote_url


def _print_help() -> None:
    print("usage: llmstack install-llama-swap [--force]")


def run(args: list[str]) -> int:
    force = False
    for arg in args:
        if arg in ("-f", "--force"):
            force = True
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            print(f"[!] unknown arg to install-llama-swap: {arg}")
            return 2

    if is_remote():
        print(f"[!] LLMSTACK_REMOTE_URL is set ({remote_url()}); the binary lives on the remote.")
        print("    `llmstack install-llama-swap` is a local-only command.")
        return 1

    install_llama_swap(force=force)
    return 0
