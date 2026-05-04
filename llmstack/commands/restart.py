"""``llmstack restart`` -- ``stop`` followed by ``start`` (passes flags through)."""

from __future__ import annotations

from llmstack.commands import start, stop


def run(args: list[str]) -> int:
    rc = stop.run([])
    if rc not in (0, None):
        return rc
    return start.run(args)
