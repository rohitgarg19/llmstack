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
  * litellm tiers -- :func:`llmstack.app._inject_sampler` adds the
    tier's sampler keys to each outbound request body (litellm has no
    server-side defaults mechanism). litellm models that reject sampler
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

from llmstack.paths import (
    AGENTS_TEMPLATE,
    AGENTS_TEMPLATE_PLAN,
    AGENTS_TEMPLATE_NOFILTER,
    AGENTS_TEMPLATE_BUILD,
    AGENTS_TEMPLATE_DEPLOY,
    agent_prompt_path,
    models_ini_path,
    remote_url,
    resolve
)

PROVIDER_KEY = "llama-swap"
API_KEY      = "sk-no-key-required"
LITELLM_PROXY_PORT = 10103

ROLE_MAP: dict[str, tuple[str, str | None, Path | None]] = {
    "fast":            ("small_model", None,            None),
    "agent":           ("agent",       "build",         AGENTS_TEMPLATE_BUILD),
    "plan":            ("agent",       "plan",          AGENTS_TEMPLATE_PLAN),
    "plan-uncensored": ("agent",       "plan-nofilter", AGENTS_TEMPLATE_NOFILTER),
    "deploy":          ("agent",       "deploy",        AGENTS_TEMPLATE_DEPLOY),
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
        "template":    "",
        "description": "Route to the uncensored planning model.",
        "agent":       "plan-nofilter",
    },
}

SHARE        = os.getenv("OPENCODE_SHARE", "disabled")
USERNAME     = os.getenv("OPENCODE_USERNAME") or None

# Keep model picker scoped to the local stack even if hosted-API env vars leak in.
_REMOTE_PROVIDERS = (
    "anthropic,openai,google,openrouter,xai,groq,deepseek,"
    "mistral,cerebras,azure,perplexity,vercel,morph"
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


def _has_litellm_tier(cfg: configparser.ConfigParser) -> bool:
    """True when any non-ROUTING section declares a litellm backend.

    Mirrors :func:`llmstack.tiers._detect_backend` without instantiating
    the full Tier dataclass: a section is litellm-backed when
    ``backend = litellm`` is set explicitly, or when ``model = ...`` is
    set (the implicit-litellm convention). Used to decide whether the
    litellm proxy will actually be running on
    ``LITELLM_PROXY_PORT`` (llama-swap only spawns the proxy entry when
    there's at least one litellm tier, see
    :func:`llmstack.generators.llama_swap.build_models_block`).
    """
    for section in cfg.sections():
        if section == "ROUTING":
            continue
        s = cfg[section]
        backend = (s.get("backend") or "").strip()
        if backend == "litellm":
            return True
        if not backend and (s.get("model") or "").strip():
            return True
    return False


def _mcp_block(
    remote: str | None,
    *,
    has_litellm: bool = False,
) -> dict[str, dict]:
    """Mirror ``mcp_servers`` from ``litellm_config.yaml`` into opencode.

    Reads the per-project litellm yaml, extracts each entry under
    ``mcp_servers``, and converts it to an opencode ``mcp.*`` entry
    pointed at the litellm proxy's per-server endpoint
    ``http://<host>:<port>/mcp/<name>``. The proxy fan-outs to the
    actual MCP server and applies its own auth / permission layer --
    so opencode never holds the raw MCP credentials, just a route
    through the gateway.

    External (thin-client) installs point at ``<remote>/mcp/<name>``
    so the remote llmstack's litellm proxy answers. Local installs
    use ``http://127.0.0.1:LITELLM_PROXY_PORT``.

    Additionally, when ``has_litellm`` is True (i.e. ``models.ini``
    declares at least one ``backend = litellm`` tier, which is what
    causes llama-swap to spawn the proxy), an entry named ``litellm``
    is emitted pointing at the proxy's ``/mcp`` gateway root. That
    single connection exposes every fan-out tool the proxy aggregates
    (whether or not the user has populated ``mcp_servers`` in
    ``litellm_config.yaml``), giving opencode tool access to the
    proxy's MCP gateway as soon as the litellm server is up.

    Empty / missing yaml, missing ``mcp_servers`` key, or PyYAML
    not importable still yield the ``litellm`` gateway entry when
    ``has_litellm`` is True; otherwise the result is an empty dict
    (no ``mcp`` key in opencode.json).
    """
    if remote:
        base = f"{remote.rstrip('/')}/mcp"
    else:
        base = f"http://127.0.0.1:{LITELLM_PROXY_PORT}/mcp"

    out: dict[str, dict] = {}

    if has_litellm:
        out["litellm"] = {
            "type": "remote",
            "url": base,
            "enabled": True,
        }

    try:
        import yaml
    except ImportError:
        return out
    try:
        path = resolve().litellm_config
    except Exception:
        return out
    if not path.is_file():
        return out
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return out
    servers = data.get("mcp_servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or not servers:
        return out

    for name, entry in servers.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        if name == "litellm":
            # Don't let a user-defined mcp_servers entry shadow the
            # gateway entry we just added (and vice-versa).
            continue
        # Honour stdio MCP servers transparently: opencode supports
        # type=local for those, but the litellm proxy is already
        # hosting the stdio server -- so we still expose a remote
        # tunnel into the proxy. Users who want opencode to spawn the
        # process directly can override the entry in their personal
        # opencode.json (post-install) without affecting the gateway.
        out[name] = {
            "type": "remote",
            "url": f"{base}/{name}",
            "enabled": True,
        }
    return out


ZERO_COST = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
DIGITS    = re.compile(r"\d+")
_FLOAT_RE = re.compile(r"[0-9]*\.?[0-9]+")


def _int(value: str, default: int) -> int:
    m = DIGITS.search(value or "")
    return int(m.group()) if m else default


def _parse_float(value: str | None) -> float | None:
    m = _FLOAT_RE.search(value or "")
    return float(m.group()) if m else None


def _is_litellm_section(s) -> bool:
    """True when a models.ini section is litellm-backed."""
    backend = (s.get("backend") or "").strip()
    if backend == "litellm":
        return True
    return not backend and bool((s.get("model") or "").strip())


def _tier_cost(s) -> dict:
    """Return an opencode cost dict for a models.ini section.

    For litellm-backed tiers the operator can declare pricing in
    ``models.ini`` using USD-per-1M-token keys:

        price_input_per_1m       = 3.0
        price_output_per_1m      = 15.0
        price_cache_read_per_1m  = 0.30   ; optional
        price_cache_write_per_1m = 3.75   ; optional

    opencode's cost schema uses USD-per-1M-tokens, so the values are
    passed through as-is to ``opencode.json``. Missing keys default to
    0 so the dict is always fully populated.

    For gguf (local) tiers all fields are 0 -- local inference has no
    per-token dollar cost.
    """
    if not _is_litellm_section(s):
        return ZERO_COST

    def _price(key: str) -> float:
        v = _parse_float(s.get(key))
        return v if v is not None else 0.0

    return {
        "input":        _price("price_input_per_1m"),
        "output":       _price("price_output_per_1m"),
        "cache_read":   _price("price_cache_read_per_1m"),
        "cache_write":  _price("price_cache_write_per_1m"),
    }


def build_config(
    ini_path: Path | None = None,
    *,
    ini_text: str | None = None,
    remote: str | None = None,
) -> dict:
    """Build the opencode.json dict from ``models.ini``.

    Source of the INI is one of (mutually exclusive):

      * ``ini_text`` -- raw INI content as a string. Used by
        ``llmstack install --external``: it fetches ``models.ini``
        straight from the router (``GET /models.ini``) and renders the
        opencode config without writing the file to disk. Thin clients
        don't keep a local copy of ``models.ini`` -- they re-fetch it
        on each ``install``.
      * ``ini_path`` -- explicit path. Used by callers that have a
        ``Path`` in hand (``check_models``, tests).
      * neither -- read from :func:`models_ini_path` (canonical
        per-project location, the local-mode default).

    ``remote`` overrides the router base URL: when given, opencode is
    pointed at ``{remote}/v1`` (thin-client / external mode). When
    ``None``, fall back to :func:`llmstack.paths.remote_url` -- which
    reads the persisted channel marker first, env var second -- and
    finally to the local router host/port from ``models.ini``.

    Passing ``remote`` explicitly is what ``llmstack install --external
    [URL]`` does: it has just *decided* the URL from flags + env and
    needs the renderer to honour that decision rather than looking
    again at a possibly-stale marker.
    """
    if ini_text is not None and ini_path is not None:
        raise ValueError("build_config: pass ini_text OR ini_path, not both")

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
    if ini_text is not None:
        cfg.read_string(ini_text)
    else:
        path = ini_path or models_ini_path()
        if not path.exists():
            raise SystemExit(f"models.ini not found at {path}")
        cfg.read(path)

    defaults = cfg["DEFAULT"]
    rurl = remote if remote is not None else remote_url()
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
        host     = (defaults.get("router_host") or "127.0.0.1").strip()
        port     = (defaults.get("router_port") or "10101").strip()
        base_url = f"http://{host}:{port}/v1"

    tier_sections = [s for s in cfg.sections() if s != "ROUTING"]

    # `auto` context = the fast tier's ctx_size. The router runs a
    # step-DOWN ladder (ultra -> smart -> fast as context grows), so
    # the largest window in the ladder is fast's, and that's the
    # effective ceiling for `model = auto` -- anything bigger has no
    # tier to land on. Using `min(...)` here would clip opencode to
    # the smallest tier's window even though the router would never
    # actually send a long prompt to that tier.
    fast_ctx = next(
        (
            _int(cfg[s].get("ctx_size", ""), 0)
            for s in tier_sections
            if (cfg[s].get("role") or "").strip() == "fast"
        ),
        0,
    )
    auto_ctx = fast_ctx or max(
        (_int(cfg[s].get("ctx_size", ""), 0) for s in tier_sections),
        default=8192,
    ) or 8192

    fast_output = next(
        (
            _int(cfg[s].get("max_output_tokens", ""), 0)
            for s in tier_sections
            if (cfg[s].get("role") or "").strip() == "fast"
        ),
        0,
    )
    auto_output = fast_output or 16384

    models: dict[str, dict] = {
        "auto": {
            "name":      "Auto (router selects: fast / agent / ultra)",
            "limit":     {"context": auto_ctx, "output": auto_output},
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

        _default_output = 32768 if role in ("agent", "plan-uncensored") else 8192
        output = _int(s.get("max_output_tokens", ""), 0) or _default_output

        model_entry: dict = {
            "name":      desc,
            "limit":     {"context": ctx, "output": output},
            "tool_call": True,
            "cost":      _tier_cost(s),
        }
        if role in ("agent", "plan-uncensored"):
            model_entry["reasoning"] = True
        models[sec] = model_entry

        kind, agent_name, agent_prompt = ROLE_MAP.get(role, (None, None, None))
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
        if agent_prompt:
            agent["prompt"] = agent_prompt_path(agent_prompt).read_text(encoding="utf-8")
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
        out["agent"] = {k: agents[k] for k in ("build", "plan", "plan-nofilter", "deploy") if k in agents}
    out["command"] = {
        name: spec
        for name, spec in COMMANDS.items()
        if not spec.get("agent") or spec.get("agent") in agents
    }
    mcp = _mcp_block(rurl, has_litellm=_has_litellm_tier(cfg))
    if mcp:
        out["mcp"] = mcp
    return out


def render(*, ini_text: str | None = None, remote: str | None = None) -> str:
    """Return the full opencode.json text (with trailing newline).

    ``ini_text`` and ``remote`` are forwarded to :func:`build_config`;
    see there for the resolution order.
    """
    return json.dumps(build_config(ini_text=ini_text, remote=remote), indent=2) + "\n"


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
