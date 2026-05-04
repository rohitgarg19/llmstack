"""Generate ``llama-swap.yaml`` from ``models.ini``.

Single source of truth: ``models.ini`` in the project root.

Top-level config (healthCheckTimeout, logLevel, ...), the ``llama_server``
binary path, the ``metal_defaults`` macro, the ``matrix`` (vars,
evict_costs, sets) and the ``on_startup.preload`` list are all DERIVED
from ``models.ini``:

  - llama_server   = LLAMA_SERVER_BIN (or `llama_server_bin =` in [DEFAULT])
  - metal_defaults = built from [DEFAULT] (host, n_gpu_layers, flash_attn,
                     jinja, threads, cache_type_k, cache_type_v) plus baked-in
                     `--no-warmup --no-mmap`.
  - matrix.vars    = role -> single-letter from ROLE_LETTER, value = tier name
  - matrix.evict_costs = max(1, min(30, round(size_gb / 1.5)))
  - matrix.sets    = `f & X` for each non-fast tier, plus `all_chats` =
                     `f & p & u` when 2+ chat tiers exist.
  - preload        = every tier with role == "fast"

Per-tier defaults (overridable in the ini per section):

  - aliases : ROLE_ALIASES[role]               (override: `aliases = a, b, c`)
  - ttl     : ROLE_TTL[role]                   (override: `ttl = 0`)

CLI:
  python src/gen_llama_swap_yaml.py             # print YAML to stdout
  python src/gen_llama_swap_yaml.py PATH        # write YAML to PATH
  python src/gen_llama_swap_yaml.py --use-next  # swap to hf_file_next per tier
                                                #   (or set LLMSTACK_USE_NEXT=1)
                                                # Tiers without hf_file_next stay
                                                # on their current file. Used by
                                                # `bash llmstack.sh start --next`.
"""

from __future__ import annotations

import configparser
import os
import re
import sys
from pathlib import Path

import yaml

from tiers import INI_PATH, _int, load_tiers

USE_NEXT_ENV = "LLMSTACK_USE_NEXT"

# --- Defaults baked into the generator -------------------------------------

LLAMA_SERVER_BIN_DEFAULT = "/opt/homebrew/bin/llama-server"
HEALTH_CHECK_TIMEOUT = 600
LOG_LEVEL = "info"
LOG_TO_STDOUT = "proxy"
START_PORT = 10001
GLOBAL_TTL = 0

# Role -> single-letter symbol used in matrix expressions.
ROLE_LETTER: dict[str, str] = {
    "fast":            "f",
    "agent":           "c",
    "plan":            "p",
    "plan-uncensored": "u",
}

# Role -> default aliases (overridable via ``aliases =`` in the ini section).
ROLE_ALIASES: dict[str, list[str]] = {
    "fast":            ["fast", "small", "autocomplete"],
    "agent":           ["agent", "smart", "code", "coder"],
    "plan":            ["plan", "planner", "chat"],
    "plan-uncensored": ["uncensored", "nofilter", "plan-nofilter", "heretic"],
}

# Role -> default TTL seconds (0 = never unload). Overridable via ``ttl =``.
ROLE_TTL: dict[str, int] = {
    "fast":            0,
    "agent":           1800,
    "plan":            1200,
    "plan-uncensored": 1200,
}

ROPE_RE = re.compile(
    r"yarn\s*\(\s*scale\s*=\s*(\d+)\s*,\s*orig_ctx\s*=\s*(\d+)\s*\)",
    re.IGNORECASE,
)
SAMPLER_RE = re.compile(r"(\w+)\s*=\s*([0-9.]+)")
SIZE_RE = re.compile(r"[\d.]+")


# --- Small parsers ---------------------------------------------------------

def parse_sampler(raw: str) -> dict[str, float]:
    return {k: float(v) for k, v in SAMPLER_RE.findall(raw or "")}


def parse_rope(raw: str) -> tuple[int, int] | None:
    m = ROPE_RE.search(raw or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_size_gb(raw: str, default: float = 5.0) -> float:
    m = SIZE_RE.search(raw or "")
    return float(m.group()) if m else default


def evict_cost(size_gb: float) -> int:
    return max(1, min(30, int(round(size_gb / 1.5))))


def is_truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Building blocks -------------------------------------------------------

def build_metal_defaults(d) -> str:
    """The shared llama-server flags used by every model."""
    parts = [
        f"--host {(d.get('host') or '127.0.0.1').strip()}",
        "--port ${PORT}",
        f"-ngl {(d.get('n_gpu_layers') or '999').strip()}",
        f"-fa {(d.get('flash_attn') or 'on').strip()}",
    ]
    if is_truthy(d.get("jinja"), default=True):
        parts.append("--jinja")
    parts += [
        f"--cache-type-k {(d.get('cache_type_k') or 'q8_0').strip()}",
        f"--cache-type-v {(d.get('cache_type_v') or 'q8_0').strip()}",
        f"--threads {(d.get('threads') or '-1').strip()}",
        "--no-warmup",
        "--no-mmap",
    ]
    return " ".join(parts)


def build_cmd(tier, section, *, use_next: bool = False) -> str:
    """The multi-line ``cmd`` literal block scalar for one tier.

    When ``use_next=True`` and the tier defines ``hf_file_next``, the active
    ``-hf``/``-hff`` pair is swapped to the queued upgrade target. Tiers
    without a queued upgrade are unchanged in either mode.
    """
    sampler = parse_sampler(section.get("sampler", ""))
    rope = parse_rope(section.get("rope_scaling", ""))

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
            "# To revert, regenerate without --use-next (default for `llmstack.sh start`).",
            "# Permanent promotion: edit hf_file in models.ini and re-run `llmstack.sh install`.",
            f"# Previous current file (still cached, still loadable):",
            f"#   -hff {tier.file}",
        ]
    else:
        lines += [
            f"# >>> UPGRADE-POINT ({tier.name}): swap the -hf/-hff pair below to change this tier.",
            "# See UPGRADING.md. To change permanently, edit models.ini and re-run `llmstack.sh install`.",
        ]
        if has_queued:
            lines += [
                "# Queued upgrade target (already pre-fetched if `llmstack.sh download` has run):",
                f"#   -hff {tier.file_next}",
                "# Try it without committing: bash llmstack.sh start --next",
            ]

    lines += [
        f"-hf {active_repo}",
        f"-hff {active_file}",
        f"--alias {tier.name}",
        f"-c {tier.ctx_size}",
    ]
    if rope:
        scale, orig_ctx = rope
        lines += [
            "--rope-scaling yarn",
            f"--rope-scale {scale}",
            f"--yarn-orig-ctx {orig_ctx}",
        ]
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
    return list(ROLE_ALIASES.get(tier.role, [tier.role]))


def ttl_for(tier, section) -> int:
    explicit = (section.get("ttl") or "").strip()
    if explicit:
        return _int(explicit, ROLE_TTL.get(tier.role, 1200))
    return ROLE_TTL.get(tier.role, 1200)


def build_models_block(cfg, *, use_next: bool = False) -> dict:
    tiers = load_tiers()
    out: dict = {}
    for name, tier in tiers.items():
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
    return out


def build_matrix(cfg) -> dict:
    tiers = load_tiers()
    vars_: dict[str, str] = {}
    evict: dict[str, int] = {}

    for name, tier in tiers.items():
        letter = ROLE_LETTER.get(tier.role)
        if not letter or letter in vars_:
            continue
        vars_[letter] = name
        size_gb = parse_size_gb(cfg[name].get("size_gb", ""), default=5.0)
        evict[letter] = evict_cost(size_gb)

    sets: dict[str, str] = {}
    fast = "f"
    if fast in vars_:
        for letter, name in vars_.items():
            if letter == fast:
                continue
            slug = (tiers[name].role or letter).replace("-", "_")
            sets[f"{slug}_with_fast"] = f"{fast} & {letter}"
        chat_letters = [l for l in vars_ if l != fast and l != "c"]
        if len(chat_letters) >= 2:
            sets["all_chats_with_fast"] = f"{fast} & " + " & ".join(chat_letters)

    return {"vars": vars_, "evict_costs": evict, "sets": sets}


# --- YAML emission ---------------------------------------------------------

def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


HEADER_CURRENT = """\
# yaml-language-server: $schema=https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json
#
# AUTO-GENERATED by src/gen_llama_swap_yaml.py from ../models.ini.
# Hand edits will be overwritten on the next `bash llmstack.sh install`.
# To change behaviour, edit models.ini (per-tier or [DEFAULT]) and re-run install.
"""

HEADER_NEXT = """\
# yaml-language-server: $schema=https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json
#
# AUTO-GENERATED by src/gen_llama_swap_yaml.py --use-next from ../models.ini.
# This is the EPHEMERAL "next" config used by `bash llmstack.sh start --next`.
# Tiers with hf_file_next defined are running their queued upgrade target;
# all other tiers are unchanged. Do not commit this file. To make any of
# these promotions permanent, edit hf_file in models.ini and re-run
# `bash llmstack.sh install` - that regenerates the canonical llama-swap.yaml.
"""


def build_config(*, use_next: bool = False) -> dict:
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),
        interpolation=None,
    )
    cfg.read(INI_PATH)
    defaults = cfg["DEFAULT"]

    llama_bin = (defaults.get("llama_server_bin") or LLAMA_SERVER_BIN_DEFAULT).strip()
    metal_defaults = build_metal_defaults(defaults)

    tiers = load_tiers()
    preload = [name for name, t in tiers.items() if t.role == "fast"]

    return {
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
        "matrix": build_matrix(cfg),
        "hooks": {
            "on_startup": {"preload": preload},
        },
    }


def _parse_argv(argv: list[str]) -> tuple[str, bool]:
    """Returns (output_path, use_next). Output ``-`` means stdout."""
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
    yaml.add_representer(str, _str_presenter, Dumper=yaml.SafeDumper)
    body = yaml.safe_dump(
        build_config(use_next=use_next),
        sort_keys=False,
        default_flow_style=False,
        width=200,
    )
    header = HEADER_NEXT if use_next else HEADER_CURRENT
    text = header + "\n" + body
    if target == "-":
        sys.stdout.write(text)
    else:
        Path(target).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
