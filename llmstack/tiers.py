"""Tier inventory: parse ``models.ini`` into Python objects.

This is the **data layer** for the stack -- the single source of truth for
"what tiers exist and where their weights live". A tier has a *backend*:

  ``gguf``     local llama-server (managed by llama-swap), driven by
               ``hf_repo`` + ``hf_file`` (and optional ``_next`` upgrade
               target). This is the only backend the original stack
               supported.
  ``bedrock``  hosted AWS Bedrock model, driven by ``aws_model_id``
               (and per-tier ``aws_region`` / ``aws_profile`` /
               ``aws_endpoint_url``). Credentials live in the standard
               AWS config (``~/.aws/config`` and ``~/.aws/credentials``),
               selected by ``aws_profile`` -- never in ``models.ini``,
               which is meant to be committable. Anything boto3 can do
               via a named profile (long-term keys, SSO, role chaining
               via ``role_arn`` + ``source_profile`` in
               ``~/.aws/config``, MFA, IMDS) is supported transparently.

Used by:

  - :mod:`llmstack.app`                   request dispatch (gguf -> proxy
                                          to llama-swap; bedrock -> AWS).
  - :mod:`llmstack.check_models`          snapshot table + HF metadata lookup
  - :mod:`llmstack.download.ggufs`        drives the GGUF downloader
  - :mod:`llmstack.generators.llama_swap` only emits gguf tiers
  - :mod:`llmstack.generators.opencode`   exposes every tier to opencode

Stdlib only -- safe to import before any extra dependency is present.

CLI (kept for backwards-compatible scripting):

  python -m llmstack.tiers                 # human-readable summary
  python -m llmstack.tiers --downloads     # TSV: tag<TAB>repo<TAB>file<TAB>label
"""

from __future__ import annotations

import configparser
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from llmstack.paths import models_ini_path, require_models_ini

DIGITS = re.compile(r"\d+")
SAMPLER_KV = re.compile(r"(\w+)\s*=\s*([0-9.]+)")

BACKEND_GGUF = "gguf"
BACKEND_BEDROCK = "bedrock"
KNOWN_BACKENDS = {BACKEND_GGUF, BACKEND_BEDROCK}


def _int(value: str, default: int = 0) -> int:
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


def parse_sampler(raw: str) -> dict[str, float]:
    """Parse a ``sampler = temp=0.5, top_p=0.85, top_k=20, ...`` line.

    Returns a dict keyed by the short name as it appears in models.ini
    (``temp``, ``top_p``, ``top_k``, ``min_p``, ``rep_pen``). The router
    is responsible for translating these into the OpenAI-compatible
    request-body field names that backends understand. An empty / missing
    line yields ``{}`` -- the canonical "no sampler tuning" signal that
    the router uses to pass requests through untouched (which is what
    Bedrock Claude Opus 4.7 et al. require).
    """
    return {k: float(v) for k, v in SAMPLER_KV.findall(raw or "")}


def _strip(value: str | None) -> str:
    return (value or "").strip()


def _opt(value: str | None) -> str | None:
    """Return a stripped non-empty string, else ``None``.

    Values can also reference an env var with ``$NAME`` or ``${NAME}`` so
    secrets stay out of ``models.ini`` if the operator prefers.
    """
    raw = _strip(value)
    if not raw:
        return None
    if raw.startswith("${") and raw.endswith("}"):
        return os.environ.get(raw[2:-1]) or None
    if raw.startswith("$"):
        return os.environ.get(raw[1:]) or None
    return raw


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
class BedrockConfig:
    """AWS Bedrock backend config for a single tier.

    Identity-only -- never holds credentials. The tier names a profile
    via :attr:`profile`; everything boto3 needs (long-term access keys,
    SSO, role chaining via ``role_arn`` + ``source_profile`` in
    ``~/.aws/config``, MFA, IMDS) is resolved by the standard AWS
    config files, not by ``models.ini``. When :attr:`profile` is
    ``None``, boto3's default credential chain applies (env vars,
    default profile, instance role, ...).

    Upgrade pre-staging (mirrors gguf ``hf_file_next``)
    ----------------------------------------------------
    ``model_id_next`` (and optional ``region_next``) is the queued
    upgrade target -- e.g. flip ``code-smart`` from Sonnet 4.5 to a
    newer Sonnet revision once it ships in your region. The router
    reads it only when ``--next`` is in effect (env var
    ``LLMSTACK_USE_NEXT=1``); the rest of the time the active
    ``model_id`` / ``region`` are used. Permanent promotion is the same
    as gguf: edit ``aws_model_id`` in models.ini and re-run
    ``llmstack install``.
    """

    model_id: str
    region: str | None = None
    profile: str | None = None
    endpoint_url: str | None = None
    model_id_next: str | None = None
    region_next: str | None = None

    @property
    def has_next(self) -> bool:
        return bool(self.model_id_next)

    def resolved(self, use_next: bool = False) -> BedrockConfig:
        """Return a copy with model_id/region swapped to the queued upgrade.

        No-op when ``use_next`` is false or the tier has no queued
        upgrade; this is what the dispatcher actually hands to boto3.
        """
        if not use_next or not self.model_id_next:
            return self
        from dataclasses import replace
        return replace(
            self,
            model_id=self.model_id_next,
            region=self.region_next or self.region,
        )


@dataclass(frozen=True)
class Tier:
    """A single tier in models.ini.

    ``backend`` discriminates between local GGUF tiers (the historical
    default) and hosted AWS Bedrock tiers. Only one set of fields is
    populated at a time:

    - ``backend == "gguf"``     -> ``repo`` + ``file`` (and optional
                                   ``repo_next`` + ``file_next``).
    - ``backend == "bedrock"``  -> ``bedrock`` is non-None.
    """

    name: str
    role: str
    backend: str
    description: str
    ctx_size: int
    repo: str = ""
    file: str = ""
    repo_next: str | None = None
    file_next: str | None = None
    bedrock: BedrockConfig | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # Per-tier sampling defaults (parsed from `sampler = ...` in models.ini).
    # The router injects these into outbound request bodies so that:
    #   1. opencode.json stays sampler-free (clients pick a model and let
    #      the stack decide how to sample it).
    #   2. Bedrock-hosted tiers whose backing model rejects sampler params
    #      (e.g. Claude Opus 4.7) can simply omit `sampler =` and the
    #      router will pass requests through untouched.
    # Keys are the short names as written in models.ini (`temp`, `top_p`,
    # `top_k`, `min_p`, `rep_pen`); the router maps them to OpenAI-compat
    # request fields.
    sampler: dict[str, float] = field(default_factory=dict)

    def files(self) -> list[TierFile]:
        """Return the GGUF download targets for this tier (empty for non-gguf)."""
        if self.backend != BACKEND_GGUF or not (self.repo and self.file):
            return []
        out = [TierFile(self.name, self.role, "current", self.repo, self.file)]
        if self.file_next:
            out.append(TierFile(
                self.name, self.role, "next",
                self.repo_next or self.repo, self.file_next,
            ))
        return out

    @property
    def is_gguf(self) -> bool:
        return self.backend == BACKEND_GGUF

    @property
    def is_bedrock(self) -> bool:
        return self.backend == BACKEND_BEDROCK

    @property
    def has_next(self) -> bool:
        """Does this tier declare a queued upgrade target?

        Backend-aware: gguf checks ``hf_file_next``, bedrock checks
        ``aws_model_id_next``. Used by ``start --next`` to decide
        whether the channel switch has anything to do, and by
        ``check`` to print an extra row.
        """
        if self.is_gguf:
            return bool(self.file_next)
        if self.is_bedrock:
            return bool(self.bedrock and self.bedrock.has_next)
        return False


def _detect_backend(section) -> str:
    """Pick the backend implied by which keys the section sets."""
    explicit = _strip(section.get("backend"))
    if explicit:
        if explicit not in KNOWN_BACKENDS:
            raise SystemExit(
                f"[!] models.ini [{section.name}] has unknown backend={explicit!r} "
                f"(supported: {', '.join(sorted(KNOWN_BACKENDS))})"
            )
        return explicit
    if _strip(section.get("aws_model_id")):
        return BACKEND_BEDROCK
    if _strip(section.get("hf_repo")) and _strip(section.get("hf_file")):
        return BACKEND_GGUF
    return ""


BANNED_BEDROCK_KEYS = {
    # Hard-secret material -- belongs in ~/.aws/credentials, never here.
    "aws_access_key_id":     "long-term access key",
    "aws_secret_access_key": "long-term secret key",
    "aws_session_token":     "STS session token",
    # Things boto3 already handles natively in ~/.aws/config under a
    # named profile -- pointing aws_profile at that profile is the
    # correct way to opt into them, not duplicating them here.
    "aws_role_arn":          "role to assume",
    "aws_role_session_name": "role-session name",
}


def _check_no_secrets(section) -> None:
    """Reject credentials/role-chaining keys in models.ini."""
    found = sorted(k for k in BANNED_BEDROCK_KEYS if section.get(k))
    if not found:
        return
    profile_hint = _strip(section.get("aws_profile")) or "<my-profile>"
    bullets = "\n".join(
        f"      - {k}  ({BANNED_BEDROCK_KEYS[k]})" for k in found
    )
    raise SystemExit(
        f"[!] models.ini [{section.name}] contains AWS credential keys -- "
        "these must NOT live in models.ini (it is meant to be committable):\n"
        f"{bullets}\n"
        "    Move them into a named profile in ~/.aws/credentials and/or\n"
        "    ~/.aws/config, then reference it from this section:\n\n"
        f"        aws_profile = {profile_hint}\n\n"
        "    boto3 picks up the profile's keys, role_arn + source_profile,\n"
        "    SSO, MFA, etc. transparently. See `aws configure --profile\n"
        f"    {profile_hint}` and the AWS shared-config docs."
    )


def _build_bedrock(section) -> BedrockConfig:
    _check_no_secrets(section)
    model_id = _strip(section.get("aws_model_id"))
    if not model_id:
        raise SystemExit(
            f"[!] models.ini [{section.name}] backend=bedrock but aws_model_id is missing"
        )
    return BedrockConfig(
        model_id=model_id,
        region=_opt(section.get("aws_region")),
        profile=_opt(section.get("aws_profile")),
        endpoint_url=_opt(section.get("aws_endpoint_url")),
        model_id_next=_opt(section.get("aws_model_id_next")),
        region_next=_opt(section.get("aws_region_next")),
    )


def _aliases(section) -> tuple[str, ...]:
    raw = _strip(section.get("aliases"))
    if not raw:
        return ()
    return tuple(a.strip() for a in raw.split(",") if a.strip())


def load_tiers(ini_path: Path | None = None) -> dict[str, Tier]:
    """Parse ``models.ini`` into a dict of tier-name -> Tier.

    Sections without a recognisable backend (no ``hf_repo``/``hf_file``
    pair *and* no ``aws_model_id``) are silently skipped -- this is how
    the ``[ROUTING]`` block stays out of the inventory.
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
        backend = _detect_backend(s)
        if not backend:
            continue

        common = {
            "name":        sec,
            "role":        _strip(s.get("role")),
            "backend":     backend,
            "description": _strip(s.get("description")) or sec,
            "ctx_size":    _int(s.get("ctx_size", "")),
            "aliases":     _aliases(s),
            "sampler":     parse_sampler(s.get("sampler", "")),
        }

        if backend == BACKEND_GGUF:
            tiers[sec] = Tier(
                **common,
                repo=_strip(s.get("hf_repo")),
                file=_strip(s.get("hf_file")),
                repo_next=_strip(s.get("hf_repo_next")) or None,
                file_next=_strip(s.get("hf_file_next")) or None,
            )
        elif backend == BACKEND_BEDROCK:
            tiers[sec] = Tier(**common, bedrock=_build_bedrock(s))
    return tiers


def iter_download_targets(ini_path: Path | None = None) -> Iterator[TierFile]:
    """Yield every :class:`TierFile` worth caching, across all tiers.

    Bedrock-backed tiers contribute nothing (no GGUFs to fetch).
    """
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
        print(f"[{tier.name}]  role={tier.role}  backend={tier.backend}  ctx={tier.ctx_size}")
        if tier.is_gguf:
            print(f"  current : {tier.repo} / {tier.file}")
            if tier.file_next:
                print(f"  next    : {tier.repo_next or tier.repo} / {tier.file_next}")
        elif tier.is_bedrock:
            b = tier.bedrock
            assert b is not None
            scope = b.region or "(default region)"
            print(f"  current : {b.model_id}  @  {scope}")
            if b.has_next:
                next_scope = b.region_next or scope
                print(f"  next    : {b.model_id_next}  @  {next_scope}")
            print(f"  profile : {b.profile or '(default chain)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
