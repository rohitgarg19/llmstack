"""Generate ``llama-swap.yaml`` from ``models.ini``.

Single source of truth: ``models.ini``. Top-level config (logging,
healthcheck, the ``llama_server`` binary path, the ``metal_defaults``
macro, the ``matrix`` and the ``on_startup.preload`` list) and per-tier
``cmd`` blocks are all DERIVED from the ini.

  - ``llama_server``    = ``[DEFAULT].llama_server_bin`` or the baked-in default
  - ``metal_defaults``  = built from ``[DEFAULT]`` (host, n_gpu_layers, ...) +
                          baked-in ``--no-warmup --no-mmap``.
  - ``matrix.vars``     = role -> single-letter from :data:`ROLE_LETTER`,
                          value = tier name
  - ``matrix.evict_costs`` = ``max(1, min(30, round(size_gb / 1.5)))``
  - ``matrix.sets``     = ``f & X`` per non-fast tier, plus an
                          ``all_chats_with_fast`` super-set when there
                          are 2+ chat tiers.
  - ``preload``         = every tier with ``role == "fast"``
  - ``aliases``         = names ``a, b, c``

Per-tier defaults (overridable in the ini per section):

  - ``ttl``     : :data:`ROLE_TTL`\\[role]       (override: ``ttl = 0``)

CLI (kept for scripting; the public entry point is ``llmstack install``):

  python -m llmstack.generators.llama_swap                 # YAML to stdout
  python -m llmstack.generators.llama_swap PATH            # write YAML to PATH
  python -m llmstack.generators.llama_swap --use-next ...  # swap hf_file_next
"""

from __future__ import annotations

import configparser
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

from llmstack._platform import EXE_SUFFIX, IS_WINDOWS
from llmstack.paths import models_ini_path, resolve
from llmstack.tiers import BACKEND_GGUF, _int, load_tiers

USE_NEXT_ENV = "LLMSTACK_USE_NEXT"
LITELLM_PROXY_PORT = 10103
LITELLM_PROXY_NAME = "litellm_proxy"
TIER_AGENT = "agent"
TIER_SUBAGENT = "subagent"


def _default_llama_server_bin() -> str:
    """Best-guess absolute path of the ``llama-server`` executable.

    Resolution order:

      1. ``$LLAMA_SERVER_BIN`` (escape hatch).
      2. ``shutil.which`` -- matches whatever the user actually has on PATH.
      3. Per-platform conventional install location, useful when the
         generated YAML will be loaded by llama-swap which doesn't share
         our PATH (e.g. launchd / systemd / scheduled tasks).

    Always returns a string, never raises -- if everything fails we hand
    back the bare ``llama-server`` name and let llama-swap surface the
    error at first request.
    """
    explicit = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if explicit:
        return explicit
    found = shutil.which(f"llama-server{EXE_SUFFIX}")
    if found:
        return found
    if IS_WINDOWS:
        for candidate in (
            r"C:\Program Files\llama.cpp\llama-server.exe",
            r"C:\tools\llama.cpp\llama-server.exe",
        ):
            if Path(candidate).is_file():
                return candidate
        return f"llama-server{EXE_SUFFIX}"
    for candidate in (
        "/opt/homebrew/bin/llama-server",   # mac (Apple Silicon Homebrew)
        "/usr/local/bin/llama-server",      # mac (Intel Homebrew) / generic
        "/usr/bin/llama-server",            # apt / dnf
    ):
        if Path(candidate).is_file():
            return candidate
    return "/opt/homebrew/bin/llama-server"


LLAMA_SERVER_BIN_DEFAULT = _default_llama_server_bin()
HEALTH_CHECK_TIMEOUT = 600
LOG_LEVEL = "info"
LOG_TO_STDOUT = "both"
START_PORT = 10001
GLOBAL_TTL = 0


def _default_litellm_bin() -> str:
    """Best-guess absolute path of the ``litellm`` proxy CLI.

    Same resolution strategy as ``llama-server``: explicit env var, then
    PATH, then conventional locations. Returns the bare name as a last
    resort so llama-swap surfaces the failure instead of us.
    """
    explicit = os.environ.get("LITELLM_BIN", "").strip()
    if explicit:
        return explicit
    found = shutil.which(f"litellm{EXE_SUFFIX}")
    if found:
        return found
    return f"litellm{EXE_SUFFIX}"


def build_litellm_proxy_cmd() -> str:
    """Build the ``cmd`` literal for the ``litellm_proxy`` model section.

    The litellm proxy must listen on a fixed well-known port
    (``LITELLM_PROXY_PORT``) so the router, status checks, and
    ``opencode.json`` all agree on where to find it without
    service-discovery.

    llama-swap captures stdout/stderr from all managed processes and
    writes them to its own log (``logToStdout: both``), so litellm's
    output lands in ``llama-swap.log`` alongside the proxy logs.

    Uses ``--config`` pointed at the per-project ``litellm_config.yaml``
    resolved through :mod:`llmstack.paths`, so the same CLI works for
    ``current`` and ``next`` channels (both share the file).
    """
    bin_path = _default_litellm_bin()
    yaml_path = resolve().litellm_config
    return (
        f"{bin_path}\n"
        f"--config {yaml_path}\n"
        f"--host 127.0.0.1\n"
        f"--port {LITELLM_PROXY_PORT}\n"
    )

ROLE_LETTER: dict[str, str] = {
    "build":         "b",
    "chat":          "c",
    "nofilter-chat": "nc",
    "publish":       "p",
}

ROLE_TTL: dict[str, int] = {
    "build":           3600,
    "publish":         1000,
    "plan":            1200,
    "plan-uncensored": 900,
}

ROPE_RE = re.compile(
    r"yarn\s*\(\s*scale\s*=\s*(\d+)\s*,\s*orig_ctx\s*=\s*(\d+)\s*,\s*type\s*=\s*(.+)\s*\)",
    re.IGNORECASE,
)
SIZE_RE = re.compile(r"[\d.]+")


def parse_rope(raw: str) -> tuple[int, int] | None:
    m = ROPE_RE.search(raw or "")
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def parse_size_gb(raw: str, default: float = 5.0) -> float:
    m = SIZE_RE.search(raw or "")
    return float(m.group()) if m else default


def evict_cost(size_gb: float) -> int:
    return max(1, min(30, int(round(size_gb / 1.5))))


def is_truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def build_metal_defaults(d) -> str:
    """The shared llama-server flags used by every model."""
    parts = [
        "--host 127.0.0.1",
        "--port ${PORT}",
        f"-ngl {(d.get('n_gpu_layers') or '99').strip()}",
        f"-fa {(d.get('flash_attn') or 'on').strip()}",
        f"--cache-type-k {(d.get('cache_type_k') or 'q8_0').strip()}",
        f"--cache-type-v {(d.get('cache_type_v') or 'q8_0').strip()}",
        f"--threads {(d.get('threads') or '-1').strip()}",
        "--no-warmup",
    ]
    return " ".join(parts)


def build_cmd(tier, section, *, use_next: bool = False) -> str:
    """The multi-line ``cmd`` literal block scalar for one tier.

    Sampling defaults (``--temp`` / ``--top-p`` / ``--top-k`` /
    ``--min-p`` / ``--repeat-penalty``) are baked into the llama-server
    startup command line for gguf tiers. They come from the tier's
    ``sampler = ...`` line in ``models.ini`` (already parsed into
    ``tier.sampler``). llama-server then applies them as its defaults
    for any request that does not override them in the body.
    """
    rope = parse_rope(section.get("rope_scaling", ""))
    sampler = tier.sampler

    has_queued = bool(tier.file_next)
    running_next = use_next and has_queued
    if running_next:
        active_repo = tier.repo_next or tier.repo
        active_file = tier.file_next
    else:
        active_repo = tier.repo
        active_file = tier.file

    lines: list[str] = ["${llama_server} ${metal_defaults}"]
    if running_next:
        lines += [
            f"# >>> RUNNING NEXT ({tier.name}): this YAML was generated with --use-next.",
            "# To revert, regenerate without --use-next (default for `llmstack start`).",
            "# Permanent promotion: edit hf_file in models.ini and re-run `llmstack install`.",
            "# Previous current file (still cached, still loadable):",
            f"#   -hff {tier.file}",
        ]
    else:
        lines += [
            f"# >>> UPGRADE-POINT ({tier.name}): swap the -hf/-hff pair below to change this tier.",
            "# See UPGRADING.md. To change permanently, edit models.ini and re-run `llmstack install`.",
        ]
        if has_queued:
            lines += [
                "# Queued upgrade target (already pre-fetched if `llmstack download` has run):",
                f"#   -hff {tier.file_next}",
                "# Try it without committing: llmstack install --next && llmstack restart",
            ]

    lines += [
        f"-hf {active_repo}",
        f"-hff {active_file}",
        f"--alias {tier.name}-{tier.role}",
        f"-c {tier.ctx_size}",
    ]
    if rope:
        scale, orig_ctx, model_type = rope
        lines += [
            "--rope-scaling yarn",
            f"--rope-scale {scale}",
            f"--yarn-orig-ctx {orig_ctx}",
            f"--override-kv {model_type}.context_length=int:{tier.ctx_size}",
        ]
    if tier.chat_template:
        lines.append(f"--chat-template {tier.chat_template}")
    else:
        lines.append("--jinja")
    if "temp" in sampler:
        lines.append(f"--temp {sampler['temp']}")
    if "top_p" in sampler:
        lines.append(f"--top-p {sampler['top_p']}")
    if "top_k" in sampler:
        lines.append(f"--top-k {int(sampler['top_k'])}")
    if "min_p" in sampler:
        lines.append(f"--min-p {sampler['min_p']}")
    if "rep_pen" in sampler:
        lines.append(f"--repeat-penalty {sampler['rep_pen']}")

    return "\n".join(lines) + "\n"

def aliases_for(tier, section) -> list[str]:
    explicit = (section.get("aliases") or "").strip()
    if explicit:
        return [a.strip() for a in explicit.split(",") if a.strip()]
    return None

def ttl_for(tier, section) -> int:
    explicit = (section.get("ttl") or "").strip()
    if explicit:
        return _int(explicit, ROLE_TTL.get(tier.role, 1200))
    return ROLE_TTL.get(tier.role, 1200)


def build_models_block(cfg, *, use_next: bool = False) -> dict:
    tiers = load_tiers()
    out: dict = {}
    has_litellm = False
    litellm_aliases: list[str] = []
    for name, tier in tiers.items():
        if tier.is_litellm:
            has_litellm = True
            base = name
            litellm_aliases.append(base)
            if tier.litellm and tier.litellm.has_next:
                litellm_aliases.append(f"{base}_next")
        if not tier.is_gguf:
            # Remote tiers (litellm, ...) are handled by the litellm
            # proxy entry below; llama-swap doesn't load them as
            # llama-server processes.
            continue
        section = cfg[name]
        running_next = use_next and bool(tier.file_next)
        size_key = "size_gb_next" if running_next else "size_gb"
        quant_key = "quant_next" if running_next else "quant"
        size_raw = section.get(size_key) or section.get("size_gb", "")
        quant_raw = section.get(quant_key) or section.get("quant", "")
        out[name] = {
            "name": tier.description,
            "description": tier.description,
            "cmd": build_cmd(tier, section, use_next=use_next),
            "ttl": ttl_for(tier, section),
            "aliases": aliases_for(tier, section),
            "metadata": {
                "tier": (section.get("tier") or "").strip() or tier.role,
                "role": tier.role,
                "ctx_size": tier.ctx_size,
                "size_gb": parse_size_gb(size_raw, default=0.0),
                "quant": (quant_raw or "").strip(),
                "channel": "next" if running_next else "current",
            },
        }

    if has_litellm:
        # Always-on, externally-managed-config model. Lives at a fixed
        # port (LITELLM_PROXY_PORT) so the router and any external
        # dashboard / MCP client always know where to find it. The
        # aliases list is the routing table: when a request comes in
        # for a model named ``<tier>_<role>``, llama-swap loads (or
        # keeps loaded) this entry and forwards the request to the
        # litellm proxy, which dispatches by ``model_name`` against
        # ``litellm_config.yaml``'s ``model_list``.
        #
        # llama-swap requires ${PORT} in cmd (validation rule), but the
        # litellm proxy must listen on a *fixed* well-known port so the
        # router, status checks, and opencode.json all agree on where to
        # find it. The ``proxy`` field overrides the default
        # ``http://localhost:${PORT}`` forwarding target, pointing
        # llama-swap at the fixed port while the cmd still satisfies the
        # ${PORT} validation requirement.
        out[LITELLM_PROXY_NAME] = {
            "name": "litellm proxy (remote model gateway + MCP + dashboard)",
            "description": "litellm proxy: hosts every backend=litellm tier, /ui dashboard, /mcp gateway",
            "cmd": build_litellm_proxy_cmd(),
            "proxy": f"http://127.0.0.1:{LITELLM_PROXY_PORT}",
            "ttl": 0,
            "aliases": litellm_aliases,
            "metadata": {
                "tier": "litellm-proxy",
                "role": "proxy",
                "channel": "current",
                "port": LITELLM_PROXY_PORT,
            },
        }
    return out


def build_matrix(cfg) -> dict | None:
    """Build the ``matrix`` block for llama-swap.

    Returns ``None`` when there are no gguf tiers -- llama-swap only
    validates the matrix when the key is present, and it requires at
    least one set when it is. Omitting the key entirely is the correct
    signal for an all-litellm (no local model) configuration.
    """
    tiers = load_tiers()
    vars_: dict[str, str] = {}
    evict: dict[str, int] = {}
    subagents = [(name, t) for name, t in tiers.items()
                    if t.tier == TIER_SUBAGENT and t.backend == BACKEND_GGUF]
    agents = [(name, t) for name, t in tiers.items()
                    if t.tier == TIER_AGENT and t.backend == BACKEND_GGUF]

    for name, tier in subagents:
        letter = ROLE_LETTER.get(tier.role)
        if not letter or letter in vars_:
            continue
        vars_[f"{letter}s"] = name
        size_gb = parse_size_gb(cfg[name].get("size_gb", ""), default=5.0)
        evict[f"{letter}s"] = evict_cost(size_gb)

    # No gguf subagent tiers at all -- omit the matrix key so llama-swap doesn't
    # attempt to validate an empty sets block.
    if not vars_:
        return None

    sets: dict[str, str] = {}
    for name, tier in agents:
        letter = ROLE_LETTER.get(tier.role)
        if not letter:
            continue
        slug = name.replace("-", "_")
        for small_key, small_name in vars_.items():
            small_slug = small_name.replace("-", "_")
            sets[f"{slug}_with_{small_slug}"] = f"{letter} & {small_key}"

    for name, tier in agents:
        letter = ROLE_LETTER.get(tier.role)
        if not letter or letter in vars_:
            continue
        vars_[f"{letter}"] = name
        size_gb = parse_size_gb(cfg[name].get("size_gb", ""), default=5.0)
        evict[f"{letter}"] = evict_cost(size_gb)
    return {"vars": vars_, "evict_costs": evict, "sets": sets}


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


HEADER_CURRENT = """\
# yaml-language-server: $schema=https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json
#
# AUTO-GENERATED by llmstack.generators.llama_swap from models.ini.
# Written by `llmstack install` (channel pinned at install time); hand
# edits will be overwritten next time the stack installs. To change
# behaviour, edit models.ini (per-tier or [DEFAULT]) and re-run
# `llmstack install` (and `llmstack restart` to reload daemons).
"""

HEADER_NEXT = """\
# yaml-language-server: $schema=https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json
#
# AUTO-GENERATED by llmstack.generators.llama_swap --use-next from models.ini.
# This is the "next" config produced by `llmstack install --next`. Tiers
# with hf_file_next defined are running their queued upgrade target; all
# other tiers are unchanged. To make any of these promotions permanent,
# flip hf_file/hf_file_next in models.ini and re-run `llmstack install
# --current` -- that regenerates the canonical yaml.
"""


def build_config(*, use_next: bool = False) -> dict:
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),
        interpolation=None,
    )
    cfg.read(models_ini_path())
    defaults = cfg["DEFAULT"]

    llama_bin = (defaults.get("llama_server_bin") or LLAMA_SERVER_BIN_DEFAULT).strip()
    metal_defaults = build_metal_defaults(defaults)

    tiers = load_tiers()
    preload = [name for name, t in tiers.items() if t.role == "fast" and t.is_gguf]
    if any(t.is_litellm for t in tiers.values()):
        preload.append(LITELLM_PROXY_NAME)

    config: dict = {
        "healthCheckTimeout": HEALTH_CHECK_TIMEOUT,
        "logLevel": LOG_LEVEL,
        "logToStdout": LOG_TO_STDOUT,
        "startPort": START_PORT,
        "sendLoadingState": True,
        "includeAliasesInList": True,
        "globalTTL": GLOBAL_TTL,
        "macros": {
            "llama_server": llama_bin,
            "metal_defaults": metal_defaults,
        },
        "models": build_models_block(cfg, use_next=use_next),
    }

    matrix = build_matrix(cfg)
    if matrix is not None:
        config["matrix"] = matrix

    config["hooks"] = {
        "on_startup": {"preload": preload},
    }
    return config


def render(*, use_next: bool = False) -> str:
    """Return the full YAML document (header + body) as a string."""
    yaml.add_representer(str, _str_presenter, Dumper=yaml.SafeDumper)
    body = yaml.safe_dump(
        build_config(use_next=use_next),
        sort_keys=False,
        default_flow_style=False,
        width=200,
    )
    header = HEADER_NEXT if use_next else HEADER_CURRENT
    return header + "\n" + body


def validate(path: Path) -> None:
    """Cheap structural sanity check: parses cleanly as YAML."""
    yaml.safe_load(path.read_text())


def _parse_argv(argv: list[str]) -> tuple[str, bool]:
    use_next = (
        os.getenv(USE_NEXT_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    )
    positional: list[str] = []
    for arg in argv[1:]:
        if arg == "--use-next":
            use_next = True
        elif arg in ("-h", "--help"):
            sys.stdout.write(__doc__ or "")
            sys.exit(0)
        else:
            positional.append(arg)
    target = positional[0] if positional else "-"
    return target, use_next


def main(argv: list[str]) -> int:
    target, use_next = _parse_argv(argv)
    text = render(use_next=use_next)
    if target == "-":
        sys.stdout.write(text)
    else:
        Path(target).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
