"""``python -m llmstack`` -> :func:`llmstack.cli.main`."""

from __future__ import annotations

import sys

from llmstack.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
