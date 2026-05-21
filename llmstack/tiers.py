"""Tier inventory: parse ``models.ini`` into Python objects.

This is the **data layer** for the stack -- the single source of truth for
"what tiers exist and where their weights live". A tier has a *backend*:

  ``gguf``     local llama-server (managed by llama-swap), driven by
               ``hf_repo`` + ``hf_file`` (and optional ``_next`` upgrade
               target). This is the only backend the original stack
               supported.
  ``litellm``  remote LLM via litellm library, driven by ``model``
               (e.g. ``anthropic/claude-sonnet-4-20250514``). Credentials
               are resolved from environment variables (``ANTHROPIC_API_KEY``,
               ``OPENAI_API_KEY``, etc.) or litellm's config file. Never
               stored in ``models.ini``, which is meant to be committable.

Used by:

  - :mod:`llmstack.app`                   request dispatch (gguf -> proxy
                                          to llama-swap; litellm -> remote).
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
BACKEND_LITELLM = "litellm"
KNOWN_BACKENDS = {BACKEND_GGUF, BACKEND_LITELLM}

ROUTING_SECTION = "ROUTING"
ROUTING_DEFAULTS = {
    "high_fidelity_ceiling": 12000,
    "mid_fidelity_ceiling":  32000,
    "multi_turn":            10,
}


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
    some litellm-backed models may require).
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
class RoutingConfig:
    """Auto-router thresholds parsed from ``[ROUTING]`` in models.ini.

    These are the cost/speed/quality knobs that decide which tier
    handles a ``model="auto"`` request. Defaults match historical
    behaviour so an absent or partial ``[ROUTING]`` section is harmless.
    """

    high_fidelity_ceiling: int = ROUTING_DEFAULTS["high_fidelity_ceiling"]
    mid_fidelity_ceiling: int = ROUTING_DEFAULTS["mid_fidelity_ceiling"]
    multi_turn: int = ROUTING_DEFAULTS["multi_turn"]


@dataclass(frozen=True)
class RouterEndpoint:
    """``router_host`` + ``router_port`` parsed from ``[DEFAULT]`` in models.ini.

    The router (this process) binds to these and downstream consumers
    (opencode, the activate hook's ``OPENAI_BASE_URL``, ...) point at
    them. Defaults match :data:`llmstack.paths.ROUTER_HOST` /
    :data:`llmstack.paths.ROUTER_PORT` so an absent file still yields
    a working local config.
    """

    host: str = "127.0.0.1"
    router_port: int = 10101


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
class LiteLLMConfig:
    """LiteLLM backend config for a single tier.

    Identity-only -- never holds credentials. The model string (e.g.
    ``anthropic/claude-sonnet-4-20250514``) is passed directly to litellm,
    which resolves credentials from environment variables
    (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, etc.) or litellm's config
    file. Credentials are never stored in ``models.ini``, which is meant
    to be committable.

    Upgrade pre-staging (mirrors gguf ``hf_file_next``)
    ------------------------------------------------
    ``model_next`` is the queued upgrade target -- e.g. flip ``code-smart``
    from Sonnet 4.5 to a newer Sonnet revision. The router reads it only
    when ``--next`` is in effect (env var ``LLMSTACK_USE_NEXT=1``); the
    rest of the time the active ``model`` is used. Permanent promotion is
    the same as gguf: edit ``model`` in models.ini and re-run
    ``llmstack install``.
    """

    model: str
    model_next: str | None = None
    max_output_tokens: int | None = None

    @property
    def has_next(self) -> bool:
        return bool(self.model_next)

    def resolved(self, use_next: bool = False) -> LiteLLMConfig:
        """Return a copy with model swapped to the queued upgrade.

        No-op when ``use_next`` is false or the tier has no queued
        upgrade; this is what the dispatcher actually hands to litellm.
        """
        if not use_next or not self.model_next:
            return self
        from dataclasses import replace
        return replace(self, model=self.model_next)


@dataclass(frozen=True)
class Tier:
    """A single tier in models.ini.

    ``backend`` discriminates between local GGUF tiers (the historical
    default) and remote litellm-backed tiers. Only one set of fields is
    populated at a time:

    - ``backend == "gguf"``     -> ``repo`` + ``file`` (and optional
                                   ``repo_next`` + ``file_next``).
    - ``backend == "litellm"``  -> ``litellm`` is non-None.
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
    litellm: LiteLLMConfig | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    sampler: dict[str, float] = field(default_factory=dict)
    max_output_tokens: int | None = None

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
    def is_litellm(self) -> bool:
        return self.backend == BACKEND_LITELLM

    @property
    def has_next(self) -> bool:
        """Does this tier declare a queued upgrade target?

        Backend-aware: gguf checks ``hf_file_next``, litellm checks
        ``model_next``. Used by ``install --next`` to decide whether
        the channel switch has anything to do, and by ``check`` to
        print an extra row.
        """
        if self.is_gguf:
            return bool(self.file_next)
        if self.is_litellm:
            return bool(self.litellm and self.litellm.has_next)
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
    if _strip(section.get("model")):
        return BACKEND_LITELLM
    if _strip(section.get("hf_repo")) and _strip(section.get("hf_file")):
        return BACKEND_GGUF
    return ""


def _build_litellm(section) -> LiteLLMConfig:
    model = _strip(section.get("model"))
    if not model:
        raise SystemExit(
            f"[!] models.ini [{section.name}] backend=litellm but model is missing"
        )
    return LiteLLMConfig(
        model=model,
        model_next=_opt(section.get("model_next")),
        max_output_tokens=_int(section.get("max_output_tokens", "")) or None,
    )


def _aliases(section) -> tuple[str, ...]:
    raw = _strip(section.get("aliases"))
    if not raw:
        return ()
    return tuple(a.strip() for a in raw.split(",") if a.strip())


def load_tiers(ini_path: Path | None = None) -> dict[str, Tier]:
    """Parse ``models.ini`` into a dict of tier-name -> Tier.

    Sections without a recognisable backend (no ``hf_repo``/``hf_file``
    pair *and* no ``model``) are silently skipped -- this is how
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
                max_output_tokens=_int(s.get("max_output_tokens", "")) or None,
            )
        elif backend == BACKEND_LITELLM:
            litellm_cfg = _build_litellm(s)
            max_out = _int(s.get("max_output_tokens", "")) or litellm_cfg.max_output_tokens
            tiers[sec] = Tier(
                **common,
                litellm=litellm_cfg,
                max_output_tokens=max_out,
            )
    return tiers


def iter_download_targets(ini_path: Path | None = None) -> Iterator[TierFile]:
    """Yield every :class:`TierFile` worth caching, across all tiers.

    LiteLLM-backed tiers contribute nothing (no GGUFs to fetch).
    """
    for tier in load_tiers(ini_path).values():
        yield from tier.files()


def load_routing(ini_path: Path | None = None) -> RoutingConfig:
    """Parse ``[ROUTING]`` from models.ini into a :class:`RoutingConfig`.

    Missing file, missing section, or missing/blank/non-numeric keys all
    fall back to :data:`ROUTING_DEFAULTS`. This keeps the router functional
    when models.ini is absent (pure pass-through proxy mode) and makes
    partial ``[ROUTING]`` sections safe.
    """
    try:
        path = ini_path or require_models_ini()
    except SystemExit:
        return RoutingConfig()

    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),
        interpolation=None,
    )
    cfg.read(path)

    if not cfg.has_section(ROUTING_SECTION):
        return RoutingConfig()

    section = cfg[ROUTING_SECTION]
    values = {}
    for key, default in ROUTING_DEFAULTS.items():
        raw = _strip(section.get(key))
        values[key] = _int(raw, default) if raw else default
    return RoutingConfig(**values)


def load_router_endpoint(ini_path: Path | None = None) -> RouterEndpoint:
    """Parse ``router_host`` + ``router_port`` from ``[DEFAULT]`` in models.ini.

    Missing file or missing keys fall back to the
    :class:`RouterEndpoint` defaults so a fresh / partial ini still
    yields a working local endpoint.
    """
    try:
        path = ini_path or require_models_ini()
    except SystemExit:
        return RouterEndpoint()

    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),
        interpolation=None,
    )
    cfg.read(path)
    defaults = cfg["DEFAULT"]
    host = _strip(defaults.get("router_host")) or RouterEndpoint.host
    port_raw = _strip(defaults.get("router_port"))
    port = _int(port_raw, RouterEndpoint.router_port) if port_raw else RouterEndpoint.router_port
    return RouterEndpoint(host=host, router_port=port)


def tier_name_for_role(role: str, ini_path: Path | None = None) -> str | None:
    """Return the first tier whose ``role`` matches ``role``, or ``None``.

    Used by the auto-router to resolve symbolic role names (``fast``,
    ``agent``, ``ultra``) to concrete tier names (``code-fast``,
    ``code-smart``, ``code-ultra``) without baking either into env
    vars or source.
    """
    for tier in load_tiers(ini_path).values():
        if tier.role == role:
            return tier.name
    return None


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
        elif tier.is_litellm:
            b = tier.litellm
            assert b is not None
            print(f"  current : {b.model}")
            if b.has_next:
                print(f"  next    : {b.model_next}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
