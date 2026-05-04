"""Tier inventory: parse ``models.ini`` into Python objects.

This is the **data layer** for the stack -- the single source of truth for
"what tiers exist and where their GGUFs live". Used by:

  - :mod:`llmstack.check_models`         snapshot table + HF metadata lookup
  - :mod:`llmstack.download.ggufs`       drives the downloader

Stdlib only -- safe to import before any extra dependency is present.

CLI (kept for backwards-compatible scripting):

  python -m llmstack.tiers                 # human-readable summary
  python -m llmstack.tiers --downloads     # TSV: tag<TAB>repo<TAB>file<TAB>label
"""

from __future__ import annotations

import configparser
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from llmstack.paths import models_ini_path, require_models_ini

DIGITS = re.compile(r"\d+")


def _int(value: str, default: int = 0) -> int:
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


@dataclass(frozen=True)
class TierFile:
    """One downloadable GGUF for a tier (current or upgrade target)."""

    tier: str       # tier section name, e.g. "code-smart"
    role: str       # role from ini, e.g. "agent"
    label: str      # "current" or "next"
    repo: str       # HuggingFace repo (owner/name)
    file: str       # GGUF filename inside that repo

    @property
    def tag(self) -> str:
        """Stable slug used for log filenames: ``<tier>-<label>``."""
        return f"{self.tier}-{self.label}"


@dataclass(frozen=True)
class Tier:
    """A single tier in models.ini (one of code-fast / code-smart / plan / ...)."""

    name: str
    role: str
    description: str
    ctx_size: int
    repo: str
    file: str
    repo_next: str | None
    file_next: str | None

    def files(self) -> list[TierFile]:
        out = [TierFile(self.name, self.role, "current", self.repo, self.file)]
        if self.file_next:
            out.append(TierFile(
                self.name, self.role, "next",
                self.repo_next or self.repo, self.file_next,
            ))
        return out


def load_tiers(ini_path: Path | None = None) -> dict[str, Tier]:
    """Parse ``models.ini`` into a dict of tier-name -> Tier.

    Tiers without an ``hf_repo`` + ``hf_file`` pair are skipped silently
    (e.g. the ``[ROUTING]`` block).
    """
    path = ini_path or require_models_ini()

    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),
        interpolation=None,
    )
    cfg.read(path)

    tiers: dict[str, Tier] = {}
    for sec in cfg.sections():
        if sec == "ROUTING":
            continue
        s = cfg[sec]
        repo = (s.get("hf_repo") or "").strip()
        file = (s.get("hf_file") or "").strip()
        if not (repo and file):
            continue
        tiers[sec] = Tier(
            name=sec,
            role=(s.get("role") or "").strip(),
            description=(s.get("description") or sec).strip(),
            ctx_size=_int(s.get("ctx_size", "")),
            repo=repo,
            file=file,
            repo_next=(s.get("hf_repo_next") or "").strip() or None,
            file_next=(s.get("hf_file_next") or "").strip() or None,
        )
    return tiers


def iter_download_targets(ini_path: Path | None = None) -> Iterator[TierFile]:
    """Yield every :class:`TierFile` worth caching, across all tiers."""
    for tier in load_tiers(ini_path).values():
        yield from tier.files()


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--downloads":
        for tf in iter_download_targets():
            print(f"{tf.tag}\t{tf.repo}\t{tf.file}\t{tf.label}")
        return 0

    path = models_ini_path()
    print(f"# tiers loaded from {path}\n")
    for tier in load_tiers(path).values():
        print(f"[{tier.name}]  role={tier.role}  ctx={tier.ctx_size}")
        print(f"  current : {tier.repo} / {tier.file}")
        if tier.file_next:
            print(f"  next    : {tier.repo_next or tier.repo} / {tier.file_next}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
