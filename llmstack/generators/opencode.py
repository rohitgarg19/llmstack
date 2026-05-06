"""Generate ``opencode.json`` from ``models.ini``.

Reads the models.ini located by :func:`llmstack.paths.models_ini_path` and
writes an opencode config to the path given as the first CLI argument
(or stdout if omitted).

What gets wired:

  provider              single llama.cpp-compatible provider at router_port
  model                 ``auto``  (FastAPI router classifies and rewrites
                                  model names)
  small_model           tier with ``role=fast``  (tab-complete, titles)
  agent.build           ``auto``                 -- always routed
  agent.plan            ``role=plan``            -- read-only, no bash
  agent.plan-nofilter   ``role=plan-uncensored`` -- read-only, no bash
  command./review,
  command./nofilter     shortcuts that the router can't auto-classify

What is **deliberately NOT wired**:

  Sampler params (``temperature``, ``top_p``, etc.) are NEVER emitted on
  any agent or model in opencode.json. Sampling is the *backend's*
  responsibility, with ``models.ini`` as the single source of truth:

  * gguf tiers -- :mod:`llmstack.generators.llama_swap` bakes the
    tier's ``sampler = ...`` into the llama-server startup command line
    (``--temp``/``--top-p``/...). llama-server applies them as defaults
    for every request.
  * Bedrock tiers -- :func:`llmstack.app._inject_sampler` adds the
    tier's sampler keys to each outbound request body (Bedrock has no
    server-side defaults mechanism). Bedrock models that reject sampler
    params (e.g. Claude Opus 4.7) declare an empty ``sampler =`` and
    the router passes requests through untouched.

  Either way, opencode.json never carries sampler params -- one place to
  edit (``models.ini``), no risk of opencode drifting from the actual
  tier config, and any other router client (curl, a different IDE) gets
  the same per-tier behaviour for free.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import sys
from pathlib import Path

from llmstack.paths import AGENTS_TEMPLATE, models_ini_path, remote_url

PROVIDER_KEY = "llama.cpp"
API_KEY      = "sk-no-key-required"

ROLE_MAP: dict[str, tuple[str, str | None]] = {
    "fast":            ("small_model", None),
    "agent":           ("agent",       "build"),
    "plan":            ("agent",       "plan"),
    "plan-uncensored": ("agent",       "plan-nofilter"),
}

READ_ONLY_AGENTS = {"plan", "plan-nofilter"}

# Slash-command shortcuts (no `/fast`: `auto` already routes trivial chat).
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


def _instructions_paths() -> list[str]:
    """Resolve the ``instructions`` array in opencode.json.

    Honours ``OPENCODE_INSTRUCTIONS`` (colon-separated) for the per-project
    install path; falls back to the bundled template inside the package.
    """
    raw = os.getenv("OPENCODE_INSTRUCTIONS", str(AGENTS_TEMPLATE))
    return [p for p in raw.split(":") if p]


ZERO_COST = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
DIGITS    = re.compile(r"\d+")


def _int(value: str, default: int) -> int:
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


def build_config(ini_path: Path | None = None) -> dict:
    path = ini_path or models_ini_path()
    if not path.exists():
        raise SystemExit(f"models.ini not found at {path}")

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
    cfg.read(path)

    defaults = cfg["DEFAULT"]
    rurl = remote_url()
    if rurl:
        # Client mode: send all traffic to the remote router. Keep the
        # tier / agent wiring derived from the local models.ini -- the
        # remote stack is expected to expose the same tier names; tier
        # ``ctx_size`` is a useful client-side hint (used by opencode
        # for prompt-packing) regardless of where the actual model
        # lives. Sampling is the *router's* responsibility (it injects
        # per-tier defaults from its own models.ini), so it never
        # appears in opencode.json.
        base_url = f"{rurl}/v1"
    else:
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

        # `build` is always wired to the auto router so escalation to
        # code-ultra (or fallback to code-fast) happens transparently.
        if agent_name == "build":
            agent_model_ref = f"{PROVIDER_KEY}/auto"
        else:
            agent_model_ref = model_ref

        # Sampler params are intentionally absent here -- the router
        # injects per-tier defaults from models.ini at request time
        # (see :func:`llmstack.app._inject_sampler`). See the module
        # docstring for the rationale.
        agent: dict = {"model": agent_model_ref}
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

    instructions = _instructions_paths()
    if instructions:
        out["instructions"] = instructions

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
        out["agent"] = {k: agents[k] for k in ("build", "plan", "plan-nofilter") if k in agents}
    out["command"] = COMMANDS
    return out


def render() -> str:
    """Return the full opencode.json text (with trailing newline)."""
    return json.dumps(build_config(), indent=2) + "\n"


def validate(path: Path) -> None:
    """Cheap structural sanity check: parses cleanly as JSON."""
    json.loads(path.read_text())


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "-"
    text = render()
    if target == "-":
        sys.stdout.write(text)
    else:
        Path(target).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
