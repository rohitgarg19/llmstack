"""``llmstack restart`` -- ``stop``, ``configure``, then ``start``.

Re-reads ``.llmstack/models.ini`` (and the channel pinned by ``init``)
on every restart, so edits to ``models.ini`` or agent prompts land
without a separate ``configure`` step. Flags are forwarded to ``start``
(``--detach``, ``--host``, ``--port``); ``configure`` takes no flags
(channel is already in the marker).
"""

from __future__ import annotations

from llmstack.commands import configure, start, stop


def run(args: list[str]) -> int:
    rc = stop.run([])
    if rc not in (0, None):
        return rc
    rc = configure.run([])
    if rc not in (0, None):
        return rc
    return start.run(args)
