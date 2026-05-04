"""Asset downloaders.

Two distinct concerns live here:

  :mod:`llmstack.download.ggufs`
    Background download of every GGUF named in ``models.ini`` using
    ``llama-completion`` (or legacy ``llama-cli``) so the standard
    llama.cpp HF cache stays the single canonical store.

  :mod:`llmstack.download.binary`
    One-shot installer for the ``llama-swap`` Go binary, fetched from its
    GitHub release tag. Detects host OS/arch, optionally pinned via the
    ``LLAMA_SWAP_VERSION`` env var.
"""

from __future__ import annotations

from llmstack.download.binary import install_llama_swap
from llmstack.download.ggufs import download_all

__all__ = ["install_llama_swap", "download_all"]
