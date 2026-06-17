"""
FastAPI auto-router proxy in front of llama-swap (and litellm).

Public endpoint: ``http://127.0.0.1:10101``
Upstream:        ``http://127.0.0.1:10102`` (llama-swap)

Behaviour:

* ``GET /v1/models``                       -> proxied verbatim, plus an
                                              ``auto`` entry and any
                                              remote (e.g. litellm) tiers
                                              declared in ``models.ini``.
* ``GET /models.ini``                      -> raw text of the router's
                                              ``models.ini``. Thin
                                              clients (``llmstack
                                              install --external``)
                                              fetch this on every
                                              install and use it to
                                              regenerate
                                              ``opencode.json`` without
                                              keeping a local copy of
                                              the file. Returning a
                                              200 + valid INI doubles
                                              as the canonical health
                                              check for external
                                              clients -- there is no
                                              separate ``/health``
                                              route on the router (the
                                              catch-all proxies any
                                              such request through to
                                              llama-swap's own
                                              ``/health`` for
                                              backwards-compat curl
                                              users).
* ``POST /v1/chat/completions``,
  ``POST /v1/completions``
    - if request body ``model == "auto"`` (or unset), classify the request
      and rewrite ``model`` -> one of: ``code-fast``, ``code-smart``,
      ``code-ultra`` (when wired).
    - otherwise pass through unchanged.
    - tiers with ``backend = litellm`` in ``models.ini`` are dispatched
      to LiteLLM via :mod:`llmstack.backends.litellm` instead of
      proxied to llama-swap.
* Streaming (SSE) responses are forwarded chunk-by-chunk.
* Anything else is reverse-proxied.

Routing philosophy: **start at the top of the fidelity ladder and
step DOWN as context grows**. This inverts the classic
"escalate-on-size" pattern, and it's deliberate:

  * Top-tier hosted models (Claude Opus/Sonnet on litellm) are
    fastest *and* most accurate on short prompts, but their
    per-request latency and $cost scale with input tokens, and
    long-context performance degrades faster than headline
    benchmarks suggest.
  * The local heavy coder (``code-smart``, Qwen3-Coder 80B-A3B) has
    a 64k window -- it does its best work in the middle of that
    range, and saturates near the top.
  * The always-resident fast coder (``code-fast``, Qwen2.5-Coder 3B
    with YaRN x4) has a **128k** window, costs nothing, and benefits
    from more context: small models lean on retrieval / explicit
    examples to disambiguate, where bigger models would just guess
    from priors.

So as the conversation accumulates context, we step *down*: ultra
-> smart -> fast.

Routing decision tree (first match wins):

  1. Explicit "ultra" trigger (``[ultra]``, ``[opus]``,
     ``ultra:``, ``opus:``) AND ultra tier configured -> code-ultra
  2. Estimated input tokens <= HIGH_FIDELITY_CEILING
     ("reasonable context still being built")         -> code-ultra
                                                         (else code-smart)
  3. Estimated input tokens <= MID_FIDELITY_CEILING   -> code-smart
  4. Otherwise (long context, top-tier becomes
     expensive/slow, fast tier's 128k window is the
     best fit and it's free)                          -> code-fast

Plan and uncensored tiers are accessible via their dedicated agent
modes (``agent.plan``, ``agent.plan-nofilter``) and slash commands;
they are not auto-routed through ``model = auto``.

The auto router's effective max context window is
``[code-fast].ctx_size`` -- fast is the bottom of the step-down
ladder, so any context that would overflow the tiers above lands on
fast. Inputs longer than fast's window have no safe home and should
be considered out of scope for ``model = auto``.

Ultra-tier routing is gated on availability: rule (2) and the
"high-fidelity" rung of (4) first check that the tier is loaded
from ``models.ini`` (i.e. present in :data:`TIER_BY_ALIAS`). When
it isn't, the router silently falls back to ``code-smart`` --
otherwise rewriting ``model`` to a tier name that isn't wired up
surfaces as a 404 from llama-swap or a tier-not-found error from
the litellm dispatcher, which is just a confusing way to fail.

Run with::

    python -m llmstack.app
    # or
    uvicorn llmstack.app:app --host 127.0.0.1 --port 10101
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from llmstack.paths import SWAP_PORT, models_ini_path
from llmstack.tiers import (
    Tier,
    load_router_endpoint,
    load_routing,
    load_tiers,
    tier_name_for_role,
)

# Router endpoint comes from models.ini ``[DEFAULT]`` (router_host /
# router_port). Upstream llama-swap always binds to 127.0.0.1.
_ENDPOINT = load_router_endpoint()

UPSTREAM = f"http://127.0.0.1:{SWAP_PORT}"

USE_NEXT_ENV = "LLMSTACK_USE_NEXT"


def _use_next() -> bool:
    """``--next`` channel flag, honoured by litellm tier dispatch."""
    return os.environ.get(USE_NEXT_ENV, "").strip().lower() in ("1", "true", "yes", "on")

# Symbolic auto-router rungs. Names are resolved from models.ini by
# matching on tier ``role`` so renaming a tier (e.g. swapping
# ``code-fast`` for ``code-haiku``) needs no code change. ``None`` if
# no tier with that role is loaded -- :func:`_ultra_available` and
# :func:`classify` handle missing rungs gracefully.
FAST_MODEL = tier_name_for_role("fast") or "code-fast"
AGENT_MODEL = tier_name_for_role("agent") or "code-smart"
ULTRA_MODEL = tier_name_for_role("ultra") or "code-ultra"

# Step-DOWN ladder (see module docstring). Both ceilings are *upper
# bounds* of a tier's sweet-spot range, expressed in estimated input
# tokens (chars/4):
#
#   est <= HIGH_FIDELITY_CEILING  -> top tier (ultra, else smart)
#   est <= MID_FIDELITY_CEILING   -> code-smart
#   est >  MID_FIDELITY_CEILING   -> code-fast (or smart with tools/loop)
#
# Each ceiling is half of the corresponding tier's ``ctx_size`` in
# models.ini -- the ceiling marks where the tier still has comfortable
# headroom, and double the ceiling is where the router has already
# stepped down to the next tier (so the upper tier never has to handle
# inputs at its own limit).
#
# Defaults:
#   HIGH 12000 - "reasonable context built": a couple of files loaded,
#                instructions clear, top-tier still cheap+fast here.
#                Pairs with a 24k ctx_size on code-ultra.
#   MID  32000 - half of code-smart's 64k window; past this, hosted
#                top-tier latency/$cost balloons and code-smart starts
#                getting cramped, while code-fast's 128k YaRN window
#                still has comfortable headroom.
#
# Source of truth: models.ini ``[ROUTING]`` (high_fidelity_ceiling,
# ini and re-run ``llmstack install`` to change these.
_ROUTING = load_routing()
HIGH_FIDELITY_CEILING = _ROUTING.high_fidelity_ceiling
MID_FIDELITY_CEILING = _ROUTING.mid_fidelity_ceiling
AUTO_ALIASES = {"auto", "", None}

ULTRA_TRIGGERS = re.compile(
    r"(\[(ultra|opus)\]|^[ \t]*(ultra|opus)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s router %(message)s",
)
log = logging.getLogger("router")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global client
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    client = httpx.AsyncClient(base_url=UPSTREAM, timeout=timeout)
    litellm_tiers = sorted(t.name for t in TIERS.values() if t.is_litellm)
    log.info(
        "router up upstream=%s ladder=[ultra<=%d -> agent<=%d -> fast] "
        "fast=%s agent=%s ultra=%s litellm=%s",
        UPSTREAM, HIGH_FIDELITY_CEILING, MID_FIDELITY_CEILING,
        FAST_MODEL, AGENT_MODEL,
        f"{ULTRA_MODEL} (active)" if _ultra_available()
            else f"{ULTRA_MODEL} (unwired -- high-fidelity rung falls back to {AGENT_MODEL})",
        ",".join(litellm_tiers) or "(none)",
    )
    yield
    if client:
        await client.aclose()


app = FastAPI(title="llmstack-auto-router", version="3.0", lifespan=_lifespan)
client: httpx.AsyncClient | None = None
TIERS: dict[str, Tier] = {}
TIER_BY_ALIAS: dict[str, Tier] = {}


def _index_tiers() -> None:
    """Load ``models.ini`` and index by name + alias for fast lookup."""
    global TIERS, TIER_BY_ALIAS
    try:
        TIERS = load_tiers()
    except SystemExit as exc:
        # No models.ini -- run as a pure pass-through proxy and let
        # downstream errors describe the problem.
        log.warning("models.ini not loaded (%s); litellm dispatch disabled", exc)
        TIERS = {}
    TIER_BY_ALIAS = {}
    for tier in TIERS.values():
        TIER_BY_ALIAS[tier.name] = tier
        for alias in tier.aliases:
            TIER_BY_ALIAS.setdefault(alias, tier)


_index_tiers()


# ----------------------------- routing logic -------------------------------

def _iter_message_text(messages: list[dict[str, Any]] | None):
    if not messages:
        return
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        yield t


def _last_user_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
    return ""


def _estimate_tokens(messages: list[dict[str, Any]] | None, prompt: str | None) -> int:
    chars = len(prompt) if prompt else 0
    for t in _iter_message_text(messages):
        chars += len(t)
    return chars // 4


def _ultra_available() -> bool:
    """True iff the ultra tier is loaded from ``models.ini``.

    Every auto-route to :data:`ULTRA_MODEL` is gated on this. Without
    the guard, an explicit ``[ultra]`` trigger or the high-fidelity
    rung of the step-down ladder on a vanilla install (no
    ``code-ultra`` section) would rewrite ``model`` to a tier that
    doesn't exist downstream -- llama-swap returns 404, the litellm
    dispatcher raises -- so the request would fail even though
    falling back to ``code-smart`` would have served it just fine.
    The check is a cheap dict lookup so we run it on every classify
    invocation; that also means re-indexing tiers at runtime (e.g.
    SIGHUP -> ``_index_tiers()``) flips routing behaviour live
    without restarting the router.
    """
    return ULTRA_MODEL in TIER_BY_ALIAS


def classify(body: dict[str, Any]) -> tuple[str, str]:
    """Return (chosen_model, reason).

    Step-DOWN ladder: top fidelity for short context, fall to mid for
    medium, drop to fast for long. See module docstring for rationale.

    Only the fast / agent / ultra rungs are implemented here. Plan and
    uncensored tiers are accessible via their dedicated agent modes
    (``agent.plan``, ``agent.plan-nofilter``) and slash commands; they
    are not auto-routed from the build agent.
    """
    messages = body.get("messages") if isinstance(body.get("messages"), list) else None
    prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else None

    last_user = _last_user_text(messages)
    sys_prompts = [
        m.get("content", "")
        for m in (messages or [])
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]

    if any(ULTRA_TRIGGERS.search(s) for s in (last_user, *sys_prompts) if s):
        if _ultra_available():
            return ULTRA_MODEL, "ultra-trigger"
        log.warning("ultra-trigger ignored: %s not in models.ini; falling back to %s",
                    ULTRA_MODEL, AGENT_MODEL)
        return AGENT_MODEL, f"ultra-trigger->agent ({ULTRA_MODEL} unavailable)"

    n_turns = sum(1 for m in (messages or []) if m.get("role") == "user")
    est = _estimate_tokens(messages, prompt)

    # Rung 1: short context -- start at the top.
    if est <= HIGH_FIDELITY_CEILING:
        if _ultra_available():
            return ULTRA_MODEL, f"high-fidelity tokens~{est}<={HIGH_FIDELITY_CEILING}"
        return AGENT_MODEL, (
            f"high-fidelity tokens~{est}<={HIGH_FIDELITY_CEILING} "
            f"({ULTRA_MODEL} unavailable)"
        )

    # Rung 2: mid context -- local heavy coder is at its sweet spot.
    if est <= MID_FIDELITY_CEILING:
        return AGENT_MODEL, f"mid-fidelity tokens~{est}<={MID_FIDELITY_CEILING}"

    # Rung 3: long context -- step down to fast.
    return FAST_MODEL, f"long-context tokens~{est}>{MID_FIDELITY_CEILING}"


# ----------------------------- proxy plumbing ------------------------------

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _filter_request_headers(req: Request) -> dict[str, str]:
    return {k: v for k, v in req.headers.items() if k.lower() not in HOP_BY_HOP}


def _filter_response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}


async def _stream_proxy(method: str, path: str, body: bytes, headers: dict[str, str]) -> StreamingResponse:
    if client is None:
        raise RuntimeError("HTTP client not initialised — lifespan not started")
    upstream_req = client.build_request(method, path, content=body, headers=headers)
    upstream = await client.send(upstream_req, stream=True)

    async def gen():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        gen(),
        status_code=upstream.status_code,
        headers=_filter_response_headers(upstream),
        media_type=upstream.headers.get("content-type"),
    )


# --------------------------------- routes ----------------------------------

@app.get("/models.ini")
async def serve_models_ini() -> Response:
    """Return the router's live ``models.ini`` as text.

    Read fresh on every request rather than from the cached
    :data:`TIERS` snapshot -- a thin client running
    ``llmstack install --external`` against this router should see
    whatever the operator has most recently written to disk, even if
    the router hasn't been restarted to pick up a re-parse. (Stale
    ``TIERS`` only affects in-flight routing decisions; the file on
    disk is the source of truth for downstream config generation.)

    Returning the file is also how external clients health-check the
    router: a 200 with a non-empty INI body proves both that the
    router process is up and that the operator has a usable config
    here -- which is exactly what the client needs to render its
    own ``opencode.json``. There is no separate ``/health`` route.
    """
    path = models_ini_path()
    if not path.is_file():
        # Router is up but the operator hasn't pointed it at a
        # models.ini yet (or the file went missing). Fail loud so the
        # thin-client install surfaces a real error message instead of
        # rendering an empty opencode.json.
        return PlainTextResponse(
            f"models.ini not found at {path} on the router host.\n"
            "Set $LLMSTACK_MODELS_INI on the router or run "
            "`llmstack install` there to seed the default.\n",
            status_code=404,
            media_type="text/plain",
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("failed to read %s for /models.ini: %s", path, e)
        return PlainTextResponse(
            f"failed to read {path}: {e}\n",
            status_code=500,
            media_type="text/plain",
        )
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    if client is None:
        raise RuntimeError("HTTP client not initialised — lifespan not started")
    try:
        r = await client.get("/v1/models")
        data = r.json()
        status = r.status_code
    except Exception as exc:
        log.warning("upstream /v1/models failed: %s", exc)
        data = {"object": "list", "data": []}
        status = 200

    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        data = {"object": "list", "data": []}

    # Hosted (litellm) tiers aren't known to llama-swap; fold them in.
    seen = {entry.get("id") for entry in data["data"] if isinstance(entry, dict)}
    from llmstack.backends import litellm_backend
    for tier in TIERS.values():
        if not tier.is_litellm:
            continue
        if tier.name in seen:
            continue
        data["data"].append(litellm_backend.model_descriptor(tier))
        seen.add(tier.name)
        for alias in tier.aliases:
            if alias not in seen:
                desc = litellm_backend.model_descriptor(tier)
                desc["id"] = alias
                desc["name"] = f"{tier.description} (alias of {tier.name})"
                data["data"].append(desc)
                seen.add(alias)

    if _ultra_available():
        top_blurb = (
            f"Step-down ladder (top->bottom as context grows): "
            f"'{ULTRA_MODEL}' up to ~{HIGH_FIDELITY_CEILING} tokens, "
            f"'{AGENT_MODEL}' up to ~{MID_FIDELITY_CEILING}, "
            f"'{FAST_MODEL}' beyond that."
        )
        name = "Auto (step-down router: ultra/agent/fast)"
    else:
        top_blurb = (
            f"Step-down ladder (top->bottom as context grows): "
            f"'{AGENT_MODEL}' up to ~{MID_FIDELITY_CEILING} tokens, "
            f"'{FAST_MODEL}' beyond that."
        )
        name = "Auto (step-down router: agent/fast)"
    data["data"].insert(0, {
        "id": "auto",
        "object": "model",
        "created": 0,
        "owned_by": "router",
        "name": name,
        "description": (
            f"{top_blurb} "
            f"'[ultra]'/'[opus]' triggers force '{ULTRA_MODEL}' regardless of size."
        ),
        "tier": "auto",
    })
    return JSONResponse(content=data, status_code=status)


def _resolve_tier(name: str | None) -> Tier | None:
    if not name:
        return None
    return TIER_BY_ALIAS.get(name)


def _inject_name_json(raw: bytes, tier_name: str) -> bytes:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    try:
        msg = data["choices"][0]["message"]
        if msg.get("content"):
            msg["name"] = tier_name
    except (KeyError, IndexError, TypeError):
        pass
    return json.dumps(data).encode()


def _inject_name_sse(chunk: bytes, tier_name: str, injected: list[bool]) -> bytes:
    if injected[0]:
        return chunk
    line = chunk.decode(errors="replace")
    if not line.startswith("data: "):
        return chunk
    payload_str = line[len("data: "):].strip()
    if payload_str in ("[DONE]", ""):
        return chunk
    try:
        payload = json.loads(payload_str)
        delta = payload["choices"][0]["delta"]
        if "role" in delta:
            delta["name"] = tier_name
            injected[0] = True
            return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        pass
    return chunk


async def _handle_completion(req: Request, path: str) -> Response:
    raw = await req.body()
    headers = _filter_request_headers(req)

    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return await _stream_proxy(req.method, path, raw, headers)

    mutated = False
    est_tokens: int | None = None
    requested = body.get("model")
    if requested in AUTO_ALIASES or requested == "auto":
        chosen, reason = classify(body)
        est_tokens = _estimate_tokens(
            body.get("messages") if isinstance(body.get("messages"), list) else None,
            body.get("prompt") if isinstance(body.get("prompt"), str) else None,
        )
        body["model"] = chosen
        log.info("auto -> %s (%s) [path=%s]", chosen, reason, path)
        mutated = True

    chosen_name = body.get("model")
    tier = _resolve_tier(chosen_name)

    # litellm tiers ride the same llama-swap dispatch path as gguf
    # tiers: llama-swap registers each litellm tier as an alias of the
    # ``litellm_proxy`` model entry, so a request for ``<tier>`` is
    # forwarded to the litellm proxy (which dispatches by ``model_name``
    # against ``model_list`` in litellm_config.yaml).
    # The dashboard / MCP gateway stay reachable directly at
    # http://127.0.0.1:10103 because llama-swap pins the proxy port.
    if tier is not None and tier.is_litellm:
        proxy_name = tier.name
        if _use_next() and tier.litellm and tier.litellm.has_next:
            proxy_name = f"{proxy_name}_next"
        body["model"] = proxy_name
        mutated = True

    if mutated:
        raw = json.dumps(body).encode()

    if tier is not None and body.get("stream"):
        proxy = await _stream_proxy(req.method, path, raw, headers)
        injected: list[bool] = [False]
        tier_name = tier.name
        original_gen = proxy.body_iterator

        async def _named_gen():
            async for chunk in original_gen:
                yield _inject_name_sse(chunk, tier_name, injected)

        proxy.body_iterator = _named_gen()
        resp = proxy
    elif tier is not None:
        proxy = await _stream_proxy(req.method, path, raw, headers)
        raw_resp = b"".join([chunk async for chunk in proxy.body_iterator])
        patched = _inject_name_json(raw_resp, tier.name)
        resp = Response(
            content=patched,
            status_code=proxy.status_code,
            headers=dict(proxy.headers),
            media_type=proxy.media_type,
        )
    else:
        resp = await _stream_proxy(req.method, path, raw, headers)

    if est_tokens is not None:
        resp.headers["X-LLMStack-Tokens"] = str(est_tokens)
    return resp


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Response:
    return await _handle_completion(req, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(req: Request) -> Response:
    return await _handle_completion(req, "/v1/completions")


# --------------------------- catch-all reverse proxy -----------------------

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def passthrough(path: str, req: Request) -> Response:
    raw = await req.body()
    headers = _filter_request_headers(req)
    return await _stream_proxy(req.method, "/" + path, raw, headers)


def main() -> None:
    """Run the router with uvicorn. Used by ``python -m llmstack.app``."""
    import asyncio
    import sys as _sys

    import uvicorn

    host = _ENDPOINT.host
    port = _ENDPOINT.router_port
    argv = _sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
        elif argv[i].startswith("--host="):
            host = argv[i].split("=", 1)[1]
            i += 1
        elif argv[i] == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
        elif argv[i].startswith("--port="):
            port = int(argv[i].split("=", 1)[1])
            i += 1
        else:
            i += 1

    log_level = os.getenv("LOG_LEVEL", "info").lower()
    cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
    asyncio.run(uvicorn.Server(cfg).serve())


if __name__ == "__main__":
    main()
