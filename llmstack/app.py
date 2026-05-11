"""
FastAPI auto-router proxy in front of llama-swap (and AWS Bedrock).

Public endpoint: ``http://127.0.0.1:10101``
Upstream:        ``http://127.0.0.1:10102`` (llama-swap)

Behaviour:

* ``GET /v1/models``                       -> proxied verbatim, plus an
                                              ``auto`` entry and any
                                              hosted (e.g. bedrock) tiers
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
      ``code-ultra`` (when wired), ``plan``, ``plan-uncensored``.
    - otherwise pass through unchanged.
    - tiers with ``backend = bedrock`` in ``models.ini`` are dispatched
      to AWS Bedrock via :mod:`llmstack.backends.bedrock` instead of
      proxied to llama-swap.
* Streaming (SSE) responses are forwarded chunk-by-chunk.
* Anything else is reverse-proxied.

Routing philosophy: **start at the top of the fidelity ladder and
step DOWN as context grows**. This inverts the classic
"escalate-on-size" pattern, and it's deliberate:

  * Top-tier hosted models (Claude Opus/Sonnet on Bedrock) are
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
-> smart -> fast. Triggers and the plan track sit alongside this
ladder.

Routing decision tree (first match wins):

  1. Explicit "uncensored" trigger in the last user message
     (``[nofilter]``, ``[uncensored]``, ``[heretic]``, or a line
     starting with ``uncensored:`` / ``nofilter:``) -> plan-uncensored
  2. Explicit "ultra" trigger (``[ultra]``, ``[opus]``,
     ``ultra:``, ``opus:``) AND ultra tier configured -> code-ultra
  3. PLAN signal words AND no code-block / agent verbs / tools
     AND estimated tokens <= ``[plan]`` tier's ctx_size
     (pure design discussion that fits the planner's
     window)                                          -> plan
                                                         (if the planner's
                                                          ctx_size is breached
                                                          we fall through to
                                                          the coding ladder
                                                          rather than send a
                                                          request that won't
                                                          fit -- the coding
                                                          tiers cover larger
                                                          windows by design)
  4. Estimated input tokens <= HIGH_FIDELITY_CEILING
     ("reasonable context still being built")         -> code-ultra
                                                         (else code-smart)
  5. Estimated input tokens <= MID_FIDELITY_CEILING   -> code-smart
   6. Otherwise (long context, top-tier becomes
      expensive/slow, fast tier's 128k window is the
      best fit and it's free)                          -> code-fast
                                                         (floored at
                                                          code-smart when
                                                          n_turns >=
                                                          MULTI_TURN_THRESHOLD)

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
the bedrock dispatcher, which is just a confusing way to fail.

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

from llmstack.paths import models_ini_path
from llmstack.tiers import Tier, load_tiers

UPSTREAM = os.getenv("LLAMA_SWAP_URL", "http://127.0.0.1:10102").rstrip("/")

FAST_MODEL = os.getenv("ROUTER_FAST_MODEL", "code-fast")
AGENT_MODEL = os.getenv("ROUTER_AGENT_MODEL", "code-smart")
ULTRA_MODEL = os.getenv("ROUTER_ULTRA_MODEL", "code-ultra")
PLAN_MODEL = os.getenv("ROUTER_PLAN_MODEL", "plan")
UNCENSORED_MODEL = os.getenv("ROUTER_UNCENSORED_MODEL", "plan-uncensored")

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
HIGH_FIDELITY_CEILING = int(os.getenv("ROUTER_HIGH_FIDELITY_CEILING", "12000"))
MID_FIDELITY_CEILING = int(os.getenv("ROUTER_MID_FIDELITY_CEILING", "32000"))
# Floor the long-context rung at code-smart whenever a tool-call
# protocol is in play -- 3B models tool-call unreliably regardless of
# how big their context window is.
MULTI_TURN_THRESHOLD = int(os.getenv("ROUTER_MULTI_TURN", "10"))
AUTO_ALIASES = {"auto", "", None}

UNCENSORED_TRIGGERS = re.compile(
    r"(\[(uncensored|nofilter|no-?filter|heretic)\]"
    r"|^[ \t]*(uncensored|nofilter|no-?filter)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)

ULTRA_TRIGGERS = re.compile(
    r"(\[(ultra|opus)\]|^[ \t]*(ultra|opus)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)

PLAN_SIGNALS = re.compile(
    r"\b(plan|design|architect(ure)?|approach|trade-?off|"
    r"should\s+we|how\s+would\s+(you|we)|what\s+would\s+you|"
    r"explain\s+why|reason\s+about|think\s+(through|step|hard|carefully)|"
    r"compare\s+(options|approaches)|review\s+(the|this|my)\s+"
    r"(architecture|design|approach|plan)|brainstorm|outline|"
    r"summari[sz]e|root\s*cause|migrate|port\s+to)\b",
    re.IGNORECASE,
)

AGENT_SIGNALS = re.compile(
    r"\b(implement|fix\s+(this|the|a|my)?\s*(bug|issue|error|test)|"
    r"write\s+(a|the|some)?\s*(function|class|test|script|module|method)|"
    r"add\s+(a|the)?\s*(function|class|method|test|file|endpoint)|"
    r"create\s+(a|the)?\s*(function|class|file|component|endpoint)|"
    r"refactor|edit|patch|generate\s+code|debug|trace|"
    r"run\s+tests?|build\s+(it|this)|compile)\b",
    re.IGNORECASE,
)

CODE_BLOCK = re.compile(r"```|`[^`\n]{30,}`")

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
    bedrock_tiers = sorted(t.name for t in TIERS.values() if t.is_bedrock)
    log.info(
        "router up upstream=%s ladder=[ultra<=%d -> agent<=%d -> fast] "
        "fast=%s agent=%s ultra=%s plan=%s uncensored=%s bedrock=%s",
        UPSTREAM, HIGH_FIDELITY_CEILING, MID_FIDELITY_CEILING,
        FAST_MODEL, AGENT_MODEL,
        f"{ULTRA_MODEL} (active)" if _ultra_available()
            else f"{ULTRA_MODEL} (unwired -- high-fidelity rung falls back to {AGENT_MODEL})",
        PLAN_MODEL, UNCENSORED_MODEL,
        ",".join(bedrock_tiers) or "(none)",
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
        log.warning("models.ini not loaded (%s); bedrock dispatch disabled", exc)
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


def _matches(pattern: re.Pattern[str], messages: list[dict[str, Any]] | None, prompt: str | None) -> bool:
    if prompt and pattern.search(prompt):
        return True
    return any(pattern.search(t) for t in _iter_message_text(messages))


def _ultra_available() -> bool:
    """True iff the ultra tier is loaded from ``models.ini``.

    Every auto-route to :data:`ULTRA_MODEL` is gated on this. Without
    the guard, an explicit ``[ultra]`` trigger or the high-fidelity
    rung of the step-down ladder on a vanilla install (no
    ``code-ultra`` section) would rewrite ``model`` to a tier that
    doesn't exist downstream -- llama-swap returns 404, the bedrock
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
    """
    messages = body.get("messages") if isinstance(body.get("messages"), list) else None
    prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else None

    last_user = _last_user_text(messages)
    sys_prompts = [
        m.get("content", "")
        for m in (messages or [])
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    if any(UNCENSORED_TRIGGERS.search(s) for s in (last_user, *sys_prompts) if s):
        return UNCENSORED_MODEL, "uncensored-trigger"

    if any(ULTRA_TRIGGERS.search(s) for s in (last_user, *sys_prompts) if s):
        if _ultra_available():
            return ULTRA_MODEL, "ultra-trigger"
        # Explicit user opt-in but the tier isn't wired up. Don't 404 --
        # serve the request from the heaviest tier we *do* have and let
        # the user notice in logs that their trigger was a no-op.
        log.warning("ultra-trigger ignored: %s not in models.ini; falling back to %s",
                    ULTRA_MODEL, AGENT_MODEL)
        return AGENT_MODEL, f"ultra-trigger->agent ({ULTRA_MODEL} unavailable)"

    n_turns = sum(1 for m in (messages or []) if m.get("role") == "user")
    _last_msgs = [{"role": "user", "content": last_user}] if last_user else None
    has_code_signal = (
        _matches(CODE_BLOCK, _last_msgs, prompt)
        or _matches(AGENT_SIGNALS, _last_msgs, prompt)
    )

    est = _estimate_tokens(messages, prompt)

    # Plan track is orthogonal to the code fidelity ladder: ``plan`` is a
    # chat-tuned model meant for design / "should we" discussions. Only
    # take it when nothing about the request says "I'm about to write
    # code" (no triple-backticks, no agent verbs). Tools are stripped
    # from the request body before dispatch (see ``_handle_completion``),
    # so their presence here does not block plan routing.
    # Only route to plan if the input fits in the planner's ctx_size --
    # past that we fall through to the coding ladder which has tiers
    # (smart, fast) explicitly sized for larger contexts.
    if (
        not has_code_signal
        and _matches(PLAN_SIGNALS, messages, prompt)
    ):
        plan_tier = TIER_BY_ALIAS.get(PLAN_MODEL)
        plan_ctx = plan_tier.ctx_size if plan_tier else 0
        if not plan_ctx or est <= plan_ctx:
            return PLAN_MODEL, "plan-signal"
        log.info(
            "plan-signal but tokens~%d > %s.ctx_size %d; "
            "falling through to coding ladder",
            est, PLAN_MODEL, plan_ctx,
        )

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

    # Rung 3: long context -- step down to fast. Floor at smart only
    # when the multi-turn threshold is hit; tools alone no longer
    # prevent the step-down (plan tiers strip tools before dispatch,
    # and code-fast is a hosted model that tool-calls reliably).
    if n_turns >= MULTI_TURN_THRESHOLD:
        return AGENT_MODEL, f"long-context tokens~{est}>{MID_FIDELITY_CEILING} (user-turns={n_turns}>={MULTI_TURN_THRESHOLD} floor)"
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

    # Hosted (bedrock) tiers aren't known to llama-swap; fold them in.
    seen = {entry.get("id") for entry in data["data"] if isinstance(entry, dict)}
    from llmstack.backends import bedrock as bedrock_backend
    for tier in TIERS.values():
        if not tier.is_bedrock:
            continue
        if tier.name in seen:
            continue
        data["data"].append(bedrock_backend.model_descriptor(tier))
        seen.add(tier.name)
        for alias in tier.aliases:
            if alias not in seen:
                desc = bedrock_backend.model_descriptor(tier)
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
        name = "Auto (step-down router: ultra/agent/fast + plan/uncensored)"
    else:
        top_blurb = (
            f"Step-down ladder (top->bottom as context grows): "
            f"'{AGENT_MODEL}' up to ~{MID_FIDELITY_CEILING} tokens, "
            f"'{FAST_MODEL}' beyond that."
        )
        name = "Auto (step-down router: agent/fast + plan/uncensored)"
    data["data"].insert(0, {
        "id": "auto",
        "object": "model",
        "created": 0,
        "owned_by": "router",
        "name": name,
        "description": (
            f"{top_blurb} "
            f"'{PLAN_MODEL}' for design/planning (orthogonal to ladder); "
            f"'{UNCENSORED_MODEL}' for explicit [nofilter] triggers; "
            f"'[ultra]'/'[opus]' triggers force '{ULTRA_MODEL}' regardless of size."
        ),
        "tier": "auto",
    })
    return JSONResponse(content=data, status_code=status)


def _resolve_tier(name: str | None) -> Tier | None:
    if not name:
        return None
    return TIER_BY_ALIAS.get(name)


# Map the short sampler keys used in models.ini to the OpenAI-compatible
# request-body fields that downstream backends understand. llama.cpp
# accepts `top_k`, `min_p`, and `repetition_penalty` as extensions; the
# Bedrock backend ignores fields it can't translate to Converse.
_SAMPLER_BODY_FIELD = {
    "temp":    "temperature",
    "top_p":   "top_p",
    "top_k":   "top_k",
    "min_p":   "min_p",
    "rep_pen": "repetition_penalty",
}


def _inject_sampler(body: dict[str, Any], tier: Tier) -> bool:
    """Layer this tier's `sampler = ...` defaults onto the request body.

    **Bedrock-only.** For gguf tiers, sampling defaults are baked into
    the llama-server startup command line by
    :mod:`llmstack.generators.llama_swap`, so llama-server already
    applies them for any request whose body lacks an explicit value.
    Bedrock has no equivalent server-side mechanism -- the only place to
    apply per-tier sampling for hosted models is the outbound request
    body, which is what this function does.

    Caller-supplied values always win -- if the client already set
    `temperature`, the tier default does not overwrite it. This makes
    models.ini the source of truth for "what sampler does each tier
    use", while still letting power users override per call.

    Returns ``True`` iff anything was added (the caller re-encodes the
    raw body bytes only when the dict actually changed).

    A Bedrock tier with an empty sampler dict (no `sampler =` line, or
    all keys stripped) is a no-op -- the canonical pattern for Bedrock
    families like Claude Opus 4.7 that reject every sampler param.
    """
    if not tier.is_bedrock or not tier.sampler:
        return False
    mutated = False
    for src, dst in _SAMPLER_BODY_FIELD.items():
        if src in tier.sampler and dst not in body:
            body[dst] = tier.sampler[src]
            mutated = True
    return mutated


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
    if chosen_name in {PLAN_MODEL, UNCENSORED_MODEL} and body.get("tools"):
        log.info("plan tier %s: stripping tools from request", chosen_name)
        body.pop("tools")
        body.pop("tool_choice", None)
        mutated = True
    tier = _resolve_tier(chosen_name)
    if tier is not None and _inject_sampler(body, tier):
        mutated = True

    if mutated:
        raw = json.dumps(body).encode()

    if tier is not None and tier.is_bedrock:
        from llmstack.backends import bedrock as bedrock_backend
        resp = await bedrock_backend.dispatch(req, tier, body)
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

    import uvicorn

    log_level = os.getenv("LOG_LEVEL", "info").lower()
    host = os.getenv("ROUTER_HOST", "127.0.0.1")
    port = int(os.getenv("ROUTER_PORT", "10101"))

    cfg = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    asyncio.run(uvicorn.Server(cfg).serve())


if __name__ == "__main__":
    main()
