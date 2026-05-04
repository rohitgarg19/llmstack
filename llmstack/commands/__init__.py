"""One module per CLI action.

Each module exports a single ``run(args: list[str]) -> int`` callable
that the dispatcher in :mod:`llmstack.cli` invokes after stripping the
action name. ``args`` is the rest of ``sys.argv``; modules do their own
argparse / manual parsing -- the commands are small enough that one
shared parser would be more friction than it's worth.
"""

from __future__ import annotations
