"""Generate ``opencode.json`` from ``models.ini``.

Reads ``../../models.ini`` (relative to this file - i.e. the project root,
since this file lives in ``llmstack/src/``) and writes a fresh
opencode-compatible config to the path given as the first argument, or to
stdout if no argument / ``-`` is passed.

The mapping rules:

  provider.<key>            single provider, alias = "llama.cpp"
  provider.options.baseURL  http://<DEFAULT.host>:<DEFAULT.router_port>/v1
  models.auto               synthetic entry for the FastAPI auto-router
  models.<section>          one per non-DEFAULT, non-ROUTING section
    .limit.context          int(<section>.ctx_size)
    .limit.output           per-role output budget (see OUTPUT_BUDGET)
    .tool_call              True for every tier
    .reasoning              True only for role == "agent"
  small_model               model whose role == "fast"
  agent.<name>              one per tier whose role maps via ROLE_TO_AGENT
    .description            opencode-side role description (not the model's)
    .model                  llama.cpp/<section>
    .temperature, .top_p    parsed from <section>.sampler
"""

from __future__ import annotations

import configparser
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # llmstack/src/
LLMSTACK_DIR = HERE.parent                      # llmstack/
PROJECT_ROOT = LLMSTACK_DIR.parent              # ../
INI_PATH = PROJECT_ROOT / "models.ini"

PROVIDER_KEY = "llama.cpp"
PROVIDER_NAME = "llmstack (local llama-swap + auto router)"
API_KEY_PLACEHOLDER = "sk-no-key-required"

# role -> ("small_model" | "agent", agent_name_or_None)
ROLE_TO_AGENT: dict[str, tuple[str, str | None]] = {
    "fast":            ("small_model", None),
    "agent":           ("agent",       "build"),
    "plan":            ("agent",       "plan"),
    "plan-uncensored": ("agent",       "plan-nofilter"),
}

# Sensible per-role output caps. Clients can still override per request.
OUTPUT_BUDGET: dict[str, int] = {
    "fast":             8192,
    "agent":           32768,
    "plan":             8192,
    "plan-uncensored":  8192,
}

# opencode-side descriptions of each agent's *purpose* (distinct from the
# upstream model's description, which lives in models.ini).
AGENT_DESCRIPTIONS: dict[str, str] = {
    "build":         "Default builder/agent. Heavy coder for multi-file edits, tool use, refactors.",
    "plan":          "Read-only planning agent. Chat-tuned model for design and architecture.",
    "plan-nofilter": "Uncensored planning. Use when the topic requires no refusal behaviour.",
}

SAMPLER_KV = re.compile(r"(\w+)\s*=\s*([0-9.]+)")
DIGITS = re.compile(r"\d+")


def _int_field(value: str, default: int) -> int:
    """Pull the first integer out of an ini value (tolerates inline comments)."""
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


def parse_sampler(raw: str) -> dict[str, float]:
    """``"temp=0.5, top_p=0.85, ..."`` -> ``{"temp": 0.5, "top_p": 0.85, ...}``."""
    return {k: float(v) for k, v in SAMPLER_KV.findall(raw or "")}


def build_config(ini_path: Path = INI_PATH) -> dict:
    if not ini_path.exists():
        raise SystemExit(f"models.ini not found at {ini_path}")

    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),
        interpolation=None,
    )
    cfg.read(ini_path)

    defaults = cfg["DEFAULT"]
    host = (defaults.get("host") or "127.0.0.1").strip()
    port = (defaults.get("router_port") or "10101").strip()
    base_url = f"http://{host}:{port}/v1"

    tier_sections = [s for s in cfg.sections() if s != "ROUTING"]

    # Auto entry: capacity = max ctx across all tiers (router will downscale per pick).
    max_ctx = max(
        (_int_field(cfg[s].get("ctx_size", ""), 0) for s in tier_sections),
        default=131072,
    ) or 131072

    models: dict[str, dict] = {
        "auto": {
            "name": "Auto (router: fast/agent/plan/uncensored)",
            "limit": {"context": max_ctx, "output": 16384},
            "tool_call": True,
            "reasoning": False,
        }
    }

    small_model: str | None = None
    agent_block: dict[str, dict] = {}

    for sec in tier_sections:
        s = cfg[sec]
        role = (s.get("role") or "").strip()
        ctx = _int_field(s.get("ctx_size", ""), 8192)
        desc = (s.get("description") or sec).strip()
        sampler = parse_sampler(s.get("sampler", ""))

        entry: dict = {
            "name": desc,
            "limit": {
                "context": ctx,
                "output": OUTPUT_BUDGET.get(role, 8192),
            },
            "tool_call": True,
        }
        if role == "agent":
            entry["reasoning"] = True
        models[sec] = entry

        binding = ROLE_TO_AGENT.get(role)
        if not binding:
            continue
        kind, agent_name = binding
        model_ref = f"{PROVIDER_KEY}/{sec}"
        if kind == "small_model":
            small_model = model_ref
            continue

        assert agent_name is not None
        agent_entry: dict = {
            "description": AGENT_DESCRIPTIONS.get(agent_name, desc),
            "model": model_ref,
        }
        if "temp" in sampler:
            agent_entry["temperature"] = sampler["temp"]
        if "top_p" in sampler:
            agent_entry["top_p"] = sampler["top_p"]
        if agent_name == "plan-nofilter":
            agent_entry["mode"] = "primary"
            agent_entry["permission"] = {
                "edit": "deny",
                "write": "deny",
                "bash": "ask",
            }
            agent_entry["color"] = "warning"
        agent_block[agent_name] = agent_entry

    out: dict = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_KEY: {
                "npm": "@ai-sdk/openai-compatible",
                "name": PROVIDER_NAME,
                "options": {
                    "baseURL": base_url,
                    "apiKey": API_KEY_PLACEHOLDER,
                },
                "models": models,
            }
        },
        "model": f"{PROVIDER_KEY}/auto",
    }
    if small_model:
        out["small_model"] = small_model
    if agent_block:
        # Stable order: build, plan, plan-nofilter (only those that exist).
        ordered = {k: agent_block[k] for k in ("build", "plan", "plan-nofilter") if k in agent_block}
        out["agent"] = ordered

    return out


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "-"
    cfg = build_config()
    text = json.dumps(cfg, indent=2) + "\n"
    if target == "-":
        sys.stdout.write(text)
    else:
        Path(target).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
