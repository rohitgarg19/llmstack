"""Config generators that render the runtime configs from ``models.ini``.

Every command that mutates state runs through ``render_to`` so the file is
written atomically (tmp file in the same directory, validated, then
renamed) -- mirrors the old shell ``_render_install`` helper.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def render_to(target: Path, render: Callable[[Path], None], validate: Callable[[Path], None]) -> None:
    """Render -> validate -> atomic ``mv`` into ``target``.

    ``render`` writes the candidate file, ``validate`` raises on a bad
    payload (e.g. by trying to ``yaml.safe_load`` it).  We unlink the
    tempfile if anything fails so we never leave a half-written config.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        render(tmp)
        validate(tmp)
        os.replace(tmp, target)
        target.chmod(0o644)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


__all__ = ["render_to"]
