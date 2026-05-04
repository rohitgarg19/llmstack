"""Generate opencode.json from models.ini.

Reads ../../models.ini and writes an opencode config to the path given as
the first CLI argument, or stdout if omitted / '-'.

What gets wired:
  provider       single llama.cpp-compatible provider at router_port
  model          auto  (FastAPI router classifies and rewrites model names)
  small_model    tier with role=fast  (tab-complete, titles)
  agent.build    role=agent   — full permissions, tool use
  agent.plan     role=plan    — read-only, no bash
  agent.plan-nofilter  role=plan-uncensored  — read-only, no bash
  command./review, /nofilter  — shortcuts that can't be auto-classified
"""

from __future__ import annotations

import configparser
import json
import os
import re
import sys
from pathlib import Path

HERE         = Path(__file__).resolve().parent   # llmstack/src/
LLMSTACK_DIR = HERE.parent                       # llmstack/
PROJECT_ROOT = LLMSTACK_DIR.parent               # ../

_default_ini = PROJECT_ROOT / "models.ini"
INI_PATH = Path(os.environ["LLMSTACK_MODELS_INI"]) if "LLMSTACK_MODELS_INI" in os.environ else _default_ini

PROVIDER_KEY = "llama.cpp"
API_KEY      = "sk-no-key-required"

# role value in models.ini -> how it surfaces in opencode
ROLE_MAP: dict[str, tuple[str, str | None]] = {
    # role          kind          agent name
    "fast":            ("small_model", None),
    "agent":           ("agent",       "build"),
    "plan":            ("agent",       "plan"),
    "plan-uncensored": ("agent",       "plan-nofilter"),
}

# Agents that should be read-only (no file edits, no bash exec)
READ_ONLY_AGENTS = {"plan", "plan-nofilter"}

# Slash-commands that are genuinely useful shortcuts:
#   /review  — forces the planning model and prepends a review framing
#   /nofilter — injects the [nofilter] trigger that the router watches for
# (No /fast: `auto` already routes trivial queries to the fast tier.)
COMMANDS = {
    "review": {
        "template":    "Review the following for trade-offs, risks, and follow-ups. Be concrete.",
        "description": "Architectural review via the planning model.",
        "agent":       "plan",
    },
    "nofilter": {
        "template":    "[nofilter]",
        "description": "Route to the uncensored planning model.",
        "agent":       "plan-nofilter",
    },
}

# Top-level policy (env-overridable at install time)
SHARE        = os.getenv("OPENCODE_SHARE", "disabled")
USERNAME     = os.getenv("OPENCODE_USERNAME") or None

# Keep model picker scoped to the local stack even if hosted-API env vars leak in.
_REMOTE_PROVIDERS = (
    "anthropic,openai,google,openrouter,xai,groq,deepseek,"
    "mistral,cerebras,azure,perplexity,vercel,morph,bedrock"
)
DISABLED_PROVIDERS = [
    p.strip() for p in
    os.getenv("OPENCODE_DISABLED_PROVIDERS", _REMOTE_PROVIDERS).split(",")
    if p.strip()
]

# Instructions file: copied into .llmstack/ by `install`; path injected via
# OPENCODE_INSTRUCTIONS env var so each project gets its own copy.
_DEFAULT_INSTRUCTIONS = str(LLMSTACK_DIR / "AGENTS.md")
INSTRUCTIONS = [
    p for p in os.getenv("OPENCODE_INSTRUCTIONS", _DEFAULT_INSTRUCTIONS).split(":")
    if p
]

ZERO_COST  = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
SAMPLER_KV = re.compile(r"(\w+)\s*=\s*([0-9.]+)")
DIGITS     = re.compile(r"\d+")


def _int(value: str, default: int) -> int:
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


def _sampler(raw: str) -> dict[str, float]:
    return {k: float(v) for k, v in SAMPLER_KV.findall(raw or "")}


def build_config(ini_path: Path = INI_PATH) -> dict:
    if not ini_path.exists():
        raise SystemExit(f"models.ini not found at {ini_path}")

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
    cfg.read(ini_path)

    defaults = cfg["DEFAULT"]
    host     = (defaults.get("host") or "127.0.0.1").strip()
    port     = (defaults.get("router_port") or "10101").strip()
    base_url = f"http://{host}:{port}/v1"

    tier_sections = [s for s in cfg.sections() if s != "ROUTING"]

    # `auto` context = MIN across all tiers so opencode never packs a prompt
    # that overflows the tier the router actually picks.
    auto_ctx = min(
        (_int(cfg[s].get("ctx_size", ""), 0) for s in tier_sections),
        default=8192,
    ) or 8192

    models: dict[str, dict] = {
        "auto": {
            "name":      "Auto (router selects: fast / agent / plan / uncensored)",
            "limit":     {"context": auto_ctx, "output": 16384},
            "tool_call": True,
            "cost":      ZERO_COST,
        }
    }

    small_model: str | None = None
    agents: dict[str, dict] = {}

    for sec in tier_sections:
        s    = cfg[sec]
        role = (s.get("role") or "").strip()
        ctx  = _int(s.get("ctx_size", ""), 8192)
        desc = (s.get("description") or sec).strip()
        sp   = _sampler(s.get("sampler", ""))

        model_entry: dict = {
            "name":      desc,
            "limit":     {"context": ctx, "output": 32768 if role == "agent" else 8192},
            "tool_call": True,
            "cost":      ZERO_COST,
        }
        if role in ("agent", "plan-uncensored"):
            model_entry["reasoning"] = True
        models[sec] = model_entry

        kind, agent_name = ROLE_MAP.get(role, (None, None))
        if kind is None:
            continue

        model_ref = f"{PROVIDER_KEY}/{sec}"
        if kind == "small_model":
            small_model = model_ref
            continue

        agent: dict = {"model": model_ref}
        if "temp"  in sp: agent["temperature"] = sp["temp"]
        if "top_p" in sp: agent["top_p"]       = sp["top_p"]
        if agent_name in READ_ONLY_AGENTS:
            agent["permission"] = {"edit": "deny", "write": "deny", "bash": "deny"}
        agents[agent_name] = agent  # type: ignore[index]

    out: dict = {
        "$schema":  "https://opencode.ai/config.json",
        "share":    SHARE,
        "autoupdate": "notify",
    }
    if USERNAME:
        out["username"] = USERNAME
    if DISABLED_PROVIDERS:
        out["disabled_providers"] = DISABLED_PROVIDERS
    if INSTRUCTIONS:
        out["instructions"] = INSTRUCTIONS

    out["provider"] = {
        PROVIDER_KEY: {
            "npm":  "@ai-sdk/openai-compatible",
            "name": "llmstack (local llama-swap + auto router)",
            "options": {"baseURL": base_url, "apiKey": API_KEY},
            "models": models,
        }
    }
    out["model"] = f"{PROVIDER_KEY}/auto"
    if small_model:
        out["small_model"] = small_model
    if agents:
        # Stable order in the picker
        out["agent"] = {k: agents[k] for k in ("build", "plan", "plan-nofilter") if k in agents}
    out["command"] = COMMANDS
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
