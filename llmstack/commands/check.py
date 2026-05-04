"""``llmstack check`` -- snapshot configured GGUFs + flag drift.

Thin wrapper around :mod:`llmstack.check_models` so the action stays in
the standard commands/ tree rather than special-cased in the dispatcher.
"""

from __future__ import annotations

from llmstack import check_models


def run(args: list[str]) -> int:
    return check_models.main(args)
