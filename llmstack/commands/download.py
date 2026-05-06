"""``llmstack download`` -- queue every GGUF in models.ini in the background."""

from __future__ import annotations

from llmstack.download.ggufs import download_all
from llmstack.paths import is_remote, remote_url


def _print_help() -> None:
    print("usage: llmstack download")


def run(args: list[str]) -> int:
    for arg in args:
        if arg in ("-h", "--help"):
            _print_help()
            return 0
        print(f"[!] unknown arg to download: {arg}")
        return 2

    if is_remote():
        print(f"[!] this project is wired as a thin client of {remote_url()} (channel: external);")
        print("    GGUFs live on the remote. `llmstack download` is a local-only command.")
        return 1

    download_all()
    return 0
