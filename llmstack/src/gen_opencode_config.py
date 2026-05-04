"""Generate ``opencode.json`` from ``models.ini``.

Reads ``../../models.ini`` (relative to this file - i.e. the project root,
since this file lives in ``llmstack/src/``) and writes a fresh
opencode-compatible config to the path given as the first argument, or to
stdout if no argument / ``-`` is passed.

The mapping rules:

  provider.<key>            single provider, alias = "llama.cpp"
  provider.options.baseURL  http://<DEFAULT.host>:<DEFAULT.router_port>/v1
  models.auto               synthetic entry for the FastAPI auto-router;
                            its limit.context is the MIN across tier ctx
                            sizes (router can route here -> never overflow)
  models.<section>          one per non-DEFAULT, non-ROUTING section
    .limit.context          int(<section>.ctx_size)
    .limit.output           per-role output budget (see OUTPUT_BUDGET)
    .tool_call              True for every tier
    .reasoning              True for role == "agent" or "plan-uncensored"
    .cost                   {input: 0, output: 0}    (local = free)
  small_model               model whose role == "fast"
  agent.<name>              one per tier whose role maps via ROLE_TO_AGENT
                            plus a synthetic "chat" agent that pins fast
    .description            opencode-side role description (not the model's)
    .model                  llama.cpp/<section>
    .temperature, .top_p    parsed from <section>.sampler
    .permission             per-agent lockdown (see AGENT_PERMISSIONS)
  command.<name>            slash-commands wired to specific agents
                            (see COMMANDS) - shortcuts for common flows
  share / autoupdate /      top-level policy (see overridable env vars)
  disabled_providers /
  instructions / username

Top-level policy is overridable via environment variables, so the same
generator works for users with different homes / preferences:

  OPENCODE_SHARE              "manual" | "auto" | "disabled"
  OPENCODE_AUTOUPDATE         "true" | "false" | "notify"
  OPENCODE_USERNAME           free-form string (unset -> opencode uses $USER)
  OPENCODE_INSTRUCTIONS       colon-separated list of paths
  OPENCODE_DISABLED_PROVIDERS comma-separated provider keys
"""

from __future__ import annotations

import configparser
import json
import os
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
    "chat":          "Tiny coder for quick Q&A. Read-only, no tool use.",
}

# Per-agent permission lockdown. None -> use opencode defaults (allow with
# prompts on sensitive ops). Anything stricter goes here. Keys are accepted
# as PermissionRuleConfig by the schema (additionalProperties), so we keep
# the existing `write` key alongside `edit` for backwards compatibility.
AGENT_PERMISSIONS: dict[str, dict[str, str] | None] = {
    "build":         None,
    "plan":          {"edit": "deny", "write": "deny", "bash": "ask"},
    "plan-nofilter": {"edit": "deny", "write": "deny", "bash": "deny"},
    "chat":          {"edit": "deny", "write": "deny", "bash": "deny"},
}

# Slash-commands. The router rewrites `model: auto` based on content, but
# users still benefit from named entry points for the most common flows.
# Keys: command name (becomes /<name>); fields match the opencode schema.
COMMANDS: dict[str, dict[str, str]] = {
    "fast": {
        "template":    "{{input}}",
        "description": "Quick one-shot answer on the local 3B coder (no tools).",
        "agent":       "chat",
    },
    "review": {
        "template":    "Review the following for trade-offs, risks, and follow-ups. Be concrete.\n\n{{input}}",
        "description": "Architectural review on the planning model.",
        "agent":       "plan",
    },
    "nofilter": {
        "template":    "[nofilter] {{input}}",
        "description": "Route to the no-filter planning model.",
        "agent":       "plan-nofilter",
    },
}

# Top-level policy. All overridable via env at install time.
SHARE = os.getenv("OPENCODE_SHARE", "disabled")            # local-only by default
AUTOUPDATE_RAW = os.getenv("OPENCODE_AUTOUPDATE", "notify")
USERNAME = os.getenv("OPENCODE_USERNAME") or None          # None -> opencode falls back to $USER

# Providers opencode will auto-load if matching env vars are present. Disable
# them explicitly so the model picker stays scoped to the local stack even if
# you happen to have e.g. ANTHROPIC_API_KEY exported for unrelated work.
_DEFAULT_DISABLED_PROVIDERS = (
    "anthropic,openai,google,openrouter,xai,groq,deepseek,"
    "mistral,cerebras,azure,perplexity,vercel,morph,bedrock"
)
DISABLED_PROVIDERS = [
    p.strip() for p in
    os.getenv("OPENCODE_DISABLED_PROVIDERS", _DEFAULT_DISABLED_PROVIDERS).split(",")
    if p.strip()
]

# Instructions: extra context loaded into every session. Defaults to the
# project-tracked AGENTS.md (sibling to this generator's parent dir, i.e.
# llmstack/AGENTS.md) so the user's ~/.config/opencode/ stays untouched.
# Resolved to an absolute path so opencode doesn't have to do tilde-expansion.
_DEFAULT_INSTRUCTIONS = str(LLMSTACK_DIR / "AGENTS.md")
INSTRUCTIONS = [
    p for p in os.getenv("OPENCODE_INSTRUCTIONS", _DEFAULT_INSTRUCTIONS).split(":")
    if p
]

# Local models cost nothing; declare it so opencode's session footer doesn't
# guess (and so tokens-per-dollar plots stay sane).
ZERO_COST = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

SAMPLER_KV = re.compile(r"(\w+)\s*=\s*([0-9.]+)")
DIGITS = re.compile(r"\d+")


def _int_field(value: str, default: int) -> int:
    """Pull the first integer out of an ini value (tolerates inline comments)."""
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


def parse_sampler(raw: str) -> dict[str, float]:
    """``"temp=0.5, top_p=0.85, ..."`` -> ``{"temp": 0.5, "top_p": 0.85, ...}``."""
    return {k: float(v) for k, v in SAMPLER_KV.findall(raw or "")}


def _parse_autoupdate(raw: str) -> bool | str:
    s = (raw or "").strip().lower()
    if s == "notify":
        return "notify"
    if s in ("0", "false", "no", "off"):
        return False
    return True


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

    # Auto entry: capacity = MIN ctx across tiers it can route to. Using max
    # would mislead opencode into packing a context that overflows whichever
    # smaller tier the router actually picks. This is the single most
    # important correctness fix in this generator.
    tier_ctxs = [_int_field(cfg[s].get("ctx_size", ""), 0) for s in tier_sections]
    auto_ctx = min((c for c in tier_ctxs if c), default=8192)

    models: dict[str, dict] = {
        "auto": {
            "name": "Auto (router: fast/agent/plan/uncensored)",
            "limit": {"context": auto_ctx, "output": 16384},
            "tool_call": True,
            "reasoning": False,
            "cost": ZERO_COST,
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
            "cost": ZERO_COST,
        }
        # Render <think> tokens for any model with a reasoning mode. Today
        # that's the heavy coder (Qwen3-Coder-Next) and the uncensored plan
        # model (Mistral-Small 3.2 Heretic).
        if role in ("agent", "plan-uncensored"):
            entry["reasoning"] = True
        models[sec] = entry

        binding = ROLE_TO_AGENT.get(role)
        if not binding:
            continue
        kind, agent_name = binding
        model_ref = f"{PROVIDER_KEY}/{sec}"
        if kind == "small_model":
            small_model = model_ref
            # The fast tier doubles as a user-facing "chat" agent so users can
            # pin the small model directly without going through `auto`.
            agent_block["chat"] = _build_agent_entry(
                "chat", model_ref, sampler, desc,
            )
            continue

        assert agent_name is not None
        agent_block[agent_name] = _build_agent_entry(
            agent_name, model_ref, sampler, desc,
        )

    out: dict = {
        "$schema": "https://opencode.ai/config.json",
        "share": SHARE,
        "autoupdate": _parse_autoupdate(AUTOUPDATE_RAW),
        "snapshot": True,
    }
    if USERNAME:
        out["username"] = USERNAME
    if DISABLED_PROVIDERS:
        out["disabled_providers"] = DISABLED_PROVIDERS
    if INSTRUCTIONS:
        out["instructions"] = INSTRUCTIONS

    out["provider"] = {
        PROVIDER_KEY: {
            "npm": "@ai-sdk/openai-compatible",
            "name": PROVIDER_NAME,
            "options": {
                "baseURL": base_url,
                "apiKey": API_KEY_PLACEHOLDER,
            },
            "models": models,
        }
    }
    out["model"] = f"{PROVIDER_KEY}/auto"
    if small_model:
        out["small_model"] = small_model
    if agent_block:
        # Stable order: build, plan, plan-nofilter, chat (only those that exist).
        ordered = {
            k: agent_block[k]
            for k in ("build", "plan", "plan-nofilter", "chat")
            if k in agent_block
        }
        out["agent"] = ordered
    if COMMANDS:
        out["command"] = dict(COMMANDS)

    return out


def _build_agent_entry(
    agent_name: str,
    model_ref: str,
    sampler: dict[str, float],
    fallback_desc: str,
) -> dict:
    """Compose one agent block, including sampler + permission + special flags."""
    entry: dict = {
        "description": AGENT_DESCRIPTIONS.get(agent_name, fallback_desc),
        "model": model_ref,
    }
    if "temp" in sampler:
        entry["temperature"] = sampler["temp"]
    if "top_p" in sampler:
        entry["top_p"] = sampler["top_p"]

    if agent_name == "plan-nofilter":
        # Mark explicitly so it shows up in the agent picker; the warning
        # color makes it visually distinct from the safer agents.
        entry["mode"] = "primary"
        entry["color"] = "warning"

    perms = AGENT_PERMISSIONS.get(agent_name)
    if perms:
        entry["permission"] = dict(perms)

    return entry


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
