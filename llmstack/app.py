"""
FastAPI auto-router proxy in front of llama-swap (and AWS Bedrock).

Public endpoint: ``http://127.0.0.1:10101``
Upstream:        ``http://127.0.0.1:10102`` (llama-swap)

Behaviour:

* ``GET /v1/models``                       -> proxied verbatim, plus an
                                              ``auto`` entry and any
                                              hosted (e.g. bedrock) tiers
                                              declared in ``models.ini``.
* ``POST /v1/chat/completions``,
  ``POST /v1/completions``
    - if request body ``model == "auto"`` (or unset), classify the request
      and rewrite ``model`` -> one of: ``code-fast``, ``code-smart``,
      ``plan``, ``plan-uncensored``.
    - otherwise pass through unchanged.
    - tiers with ``backend = bedrock`` in ``models.ini`` are dispatched
      to AWS Bedrock via :mod:`llmstack.backends.bedrock` instead of
      proxied to llama-swap.
* Streaming (SSE) responses are forwarded chunk-by-chunk.
* Anything else is reverse-proxied.

Routing decision tree (first match wins):

  1. Explicit "uncensored" trigger in the last user message
     (e.g. starts with ``[nofilter]``, ``uncensored:``, or contains
     ``[uncensored]``)                                   -> plan-uncensored
  2. Tools array non-empty (agent / function-calling)    -> code-smart
  3. >= MULTI_TURN_THRESHOLD turns (agent loop)          -> code-smart
  4. Estimated input tokens > FAST_TOKEN_BUDGET          -> code-smart
  5. Code blocks (triple-backticks) or AGENT signal words -> code-smart
     (``implement``, ``fix bug``, ``write a function``,
     ``refactor``, ``debug``, ...)
  6. PLAN signal words                                   -> plan
     (``design``, ``architect``, ``approach``,
     ``trade-off``, ``should we``, ...)
  7. default                                             -> code-fast

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
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llmstack.tiers import Tier, load_tiers

UPSTREAM = os.getenv("LLAMA_SWAP_URL", "http://127.0.0.1:10102").rstrip("/")

FAST_MODEL = os.getenv("ROUTER_FAST_MODEL", "code-fast")
AGENT_MODEL = os.getenv("ROUTER_AGENT_MODEL", "code-smart")
PLAN_MODEL = os.getenv("ROUTER_PLAN_MODEL", "plan")
UNCENSORED_MODEL = os.getenv("ROUTER_UNCENSORED_MODEL", "plan-uncensored")

FAST_TOKEN_BUDGET = int(os.getenv("ROUTER_FAST_TOKEN_BUDGET", "4000"))
MULTI_TURN_THRESHOLD = int(os.getenv("ROUTER_MULTI_TURN", "6"))
AUTO_ALIASES = {"auto", "", None}

UNCENSORED_TRIGGERS = re.compile(
    r"(\[(uncensored|nofilter|no-?filter|heretic)\]"
    r"|^[ \t]*(uncensored|nofilter|no-?filter)\s*:)",
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

app = FastAPI(title="llmstack-auto-router", version="2.1")
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


@app.on_event("startup")
async def _startup() -> None:
    global client
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    client = httpx.AsyncClient(base_url=UPSTREAM, timeout=timeout)
    bedrock_tiers = sorted(t.name for t in TIERS.values() if t.is_bedrock)
    log.info(
        "router up upstream=%s fast=%s agent=%s plan=%s uncensored=%s bedrock=%s",
        UPSTREAM, FAST_MODEL, AGENT_MODEL, PLAN_MODEL, UNCENSORED_MODEL,
        ",".join(bedrock_tiers) or "(none)",
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    if client:
        await client.aclose()


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


def classify(body: dict[str, Any]) -> tuple[str, str]:
    """Return (chosen_model, reason)."""
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

    tools = body.get("tools") or []
    if tools:
        return AGENT_MODEL, f"tools={len(tools)}"

    n_turns = len(messages) if messages else 0
    if n_turns >= MULTI_TURN_THRESHOLD:
        return AGENT_MODEL, f"turns={n_turns}"

    est = _estimate_tokens(messages, prompt)
    if est > FAST_TOKEN_BUDGET:
        return AGENT_MODEL, f"tokens~{est}"

    if _matches(CODE_BLOCK, messages, prompt):
        return AGENT_MODEL, "code-block"
    if _matches(AGENT_SIGNALS, messages, prompt):
        return AGENT_MODEL, "agent-signal"

    if _matches(PLAN_SIGNALS, messages, prompt):
        return PLAN_MODEL, "plan-signal"

    return FAST_MODEL, "default"


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
    assert client is not None
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

@app.get("/health")
async def health() -> dict[str, Any]:
    assert client is not None
    try:
        r = await client.get("/health", timeout=5.0)
        upstream_ok = r.status_code == 200
    except Exception as e:  # pragma: no cover
        upstream_ok = False
        log.warning("upstream health failed: %s", e)
    return {
        "router": "ok",
        "upstream_ok": upstream_ok,
        "upstream": UPSTREAM,
        "tiers": {
            "fast": FAST_MODEL,
            "agent": AGENT_MODEL,
            "plan": PLAN_MODEL,
            "uncensored": UNCENSORED_MODEL,
        },
        "bedrock_tiers": [t.name for t in TIERS.values() if t.is_bedrock],
    }


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    assert client is not None
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

    data["data"].insert(0, {
        "id": "auto",
        "object": "model",
        "created": 0,
        "owned_by": "router",
        "name": "Auto (router: fast/agent/plan/uncensored)",
        "description": (
            f"Routes to '{FAST_MODEL}' for trivial chat, "
            f"'{AGENT_MODEL}' for code/agent work, "
            f"'{PLAN_MODEL}' for design/planning, "
            f"'{UNCENSORED_MODEL}' for explicit [nofilter] triggers."
        ),
        "tier": "auto",
    })
    return JSONResponse(content=data, status_code=status)


def _resolve_tier(name: str | None) -> Tier | None:
    if not name:
        return None
    return TIER_BY_ALIAS.get(name)


async def _handle_completion(req: Request, path: str) -> Response:
    raw = await req.body()
    headers = _filter_request_headers(req)

    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return await _stream_proxy(req.method, path, raw, headers)

    requested = body.get("model")
    if requested in AUTO_ALIASES or requested == "auto":
        chosen, reason = classify(body)
        body["model"] = chosen
        log.info("auto -> %s (%s) [path=%s]", chosen, reason, path)
        raw = json.dumps(body).encode()

    chosen_name = body.get("model")
    tier = _resolve_tier(chosen_name)
    if tier is not None and tier.is_bedrock:
        from llmstack.backends import bedrock as bedrock_backend
        return await bedrock_backend.dispatch(req, tier, body)

    return await _stream_proxy(req.method, path, raw, headers)


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
