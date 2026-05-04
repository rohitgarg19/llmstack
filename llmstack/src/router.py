"""
Auto-router proxy in front of llama-swap.

Public endpoint: http://127.0.0.1:10101
Upstream:        http://127.0.0.1:10102 (llama-swap)

Behaviour:
  * /v1/models                       -> proxied verbatim, plus an "auto" entry.
  * /v1/chat/completions, /v1/completions
        - if request body model == "auto" (or unset), classify the request and
          rewrite model -> one of: code-fast, code-smart, plan, plan-uncensored.
        - otherwise pass through unchanged.
  * Streaming (SSE) responses are forwarded chunk-by-chunk.
  * Anything else is reverse-proxied.

Routing decision tree (first match wins):

  1. Explicit "uncensored" trigger in the last user message
     (e.g. starts with "[nofilter]", "uncensored:", or contains "[uncensored]")
                                                       -> plan-uncensored
  2. Tools array non-empty (agent / function-calling)  -> code-smart
  3. >= MULTI_TURN_THRESHOLD turns (agent loop)        -> code-smart
  4. Estimated input tokens > FAST_TOKEN_BUDGET        -> code-smart
  5. Code blocks (```) or AGENT signal words           -> code-smart
     ("implement", "fix bug", "write a function", "refactor", "debug", ...)
  6. PLAN signal words                                 -> plan
     ("design", "architect", "approach", "trade-off", "should we", ...)
  7. default                                           -> code-fast
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

# ---------------------------------------------------------------------------
UPSTREAM = os.getenv("LLAMA_SWAP_URL", "http://127.0.0.1:10102").rstrip("/")

FAST_MODEL = os.getenv("ROUTER_FAST_MODEL", "code-fast")
AGENT_MODEL = os.getenv("ROUTER_AGENT_MODEL", "code-smart")
PLAN_MODEL = os.getenv("ROUTER_PLAN_MODEL", "plan")
UNCENSORED_MODEL = os.getenv("ROUTER_UNCENSORED_MODEL", "plan-uncensored")

FAST_TOKEN_BUDGET = int(os.getenv("ROUTER_FAST_TOKEN_BUDGET", "4000"))
MULTI_TURN_THRESHOLD = int(os.getenv("ROUTER_MULTI_TURN", "6"))
AUTO_ALIASES = {"auto", "", None}

# Explicit uncensored opt-in. Matches:
#   "[nofilter]" / "[uncensored]" / "[heretic]" anywhere
#   "uncensored:" / "nofilter:" / "no-filter:" at the start of a line
UNCENSORED_TRIGGERS = re.compile(
    r"(\[(uncensored|nofilter|no-?filter|heretic)\]"
    r"|^[ \t]*(uncensored|nofilter|no-?filter)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)

# Plan-mode signals: design/discussion/architecture
PLAN_SIGNALS = re.compile(
    r"\b(plan|design|architect(ure)?|approach|trade-?off|"
    r"should\s+we|how\s+would\s+(you|we)|what\s+would\s+you|"
    r"explain\s+why|reason\s+about|think\s+(through|step|hard|carefully)|"
    r"compare\s+(options|approaches)|review\s+(the|this|my)\s+"
    r"(architecture|design|approach|plan)|brainstorm|outline|"
    r"summari[sz]e|root\s*cause|migrate|port\s+to)\b",
    re.IGNORECASE,
)

# Agent/build signals: doing work
AGENT_SIGNALS = re.compile(
    r"\b(implement|fix\s+(this|the|a|my)?\s*(bug|issue|error|test)|"
    r"write\s+(a|the|some)?\s*(function|class|test|script|module|method)|"
    r"add\s+(a|the)?\s*(function|class|method|test|file|endpoint)|"
    r"create\s+(a|the)?\s*(function|class|file|component|endpoint)|"
    r"refactor|edit|patch|generate\s+code|debug|trace|"
    r"run\s+tests?|build\s+(it|this)|compile)\b",
    re.IGNORECASE,
)

# Code blocks or backtick-fenced code
CODE_BLOCK = re.compile(r"```|`[^`\n]{30,}`")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s router %(message)s",
)
log = logging.getLogger("router")
# ---------------------------------------------------------------------------

app = FastAPI(title="llmstack-auto-router", version="2.0")
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global client
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    client = httpx.AsyncClient(base_url=UPSTREAM, timeout=timeout)
    log.info(
        "router up upstream=%s fast=%s agent=%s plan=%s uncensored=%s",
        UPSTREAM, FAST_MODEL, AGENT_MODEL, PLAN_MODEL, UNCENSORED_MODEL,
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

    # 1. explicit uncensored opt-in (only checked on the most recent user turn
    #    + the system prompt to avoid stale triggers polluting later turns)
    last_user = _last_user_text(messages)
    sys_prompts = [m.get("content", "") for m in (messages or []) if m.get("role") == "system" and isinstance(m.get("content"), str)]
    if any(UNCENSORED_TRIGGERS.search(s) for s in (last_user, *sys_prompts) if s):
        return UNCENSORED_MODEL, "uncensored-trigger"

    # 2. tools / function calling -> agent
    tools = body.get("tools") or []
    if tools:
        return AGENT_MODEL, f"tools={len(tools)}"

    # 3. multi-turn conversation -> agent loop
    n_turns = len(messages) if messages else 0
    if n_turns >= MULTI_TURN_THRESHOLD:
        return AGENT_MODEL, f"turns={n_turns}"

    # 4. heavy context -> agent
    est = _estimate_tokens(messages, prompt)
    if est > FAST_TOKEN_BUDGET:
        return AGENT_MODEL, f"tokens~{est}"

    # 5. code blocks or agent verbs -> agent
    if _matches(CODE_BLOCK, messages, prompt):
        return AGENT_MODEL, "code-block"
    if _matches(AGENT_SIGNALS, messages, prompt):
        return AGENT_MODEL, "agent-signal"

    # 6. plan signals -> plan
    if _matches(PLAN_SIGNALS, messages, prompt):
        return PLAN_MODEL, "plan-signal"

    # 7. trivial chat -> fast coder
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
    }


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    assert client is not None
    r = await client.get("/v1/models")
    data = r.json()
    if isinstance(data, dict) and isinstance(data.get("data"), list):
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
    return JSONResponse(content=data, status_code=r.status_code)


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

    return await _stream_proxy(req.method, path, raw, headers)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Response:
    return await _handle_completion(req, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(req: Request) -> Response:
    return await _handle_completion(req, "/v1/completions")


# --------------------------- catch-all reverse proxy -----------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def passthrough(path: str, req: Request) -> Response:
    raw = await req.body()
    headers = _filter_request_headers(req)
    return await _stream_proxy(req.method, "/" + path, raw, headers)


if __name__ == "__main__":
    # Two bindings:
    #   - TCP at ROUTER_HOST:ROUTER_PORT  -> for opencode (HTTP client over fetch)
    #   - UDS at ROUTER_UDS               -> for power-user tooling, e.g.:
    #         curl --unix-socket .llmstack/router.sock http://x/v1/models
    #     opencode can't dial Unix sockets, so TCP is still the primary
    #     interface; UDS is a side door that's per-project-isolated by path.
    #
    # uvicorn's CLI flags are mutually exclusive (one process == one bind),
    # so we run two Server instances over the same FastAPI app on a single
    # asyncio loop. If ROUTER_UDS is unset/empty, only TCP is bound.
    import asyncio

    import uvicorn

    log_level = os.getenv("LOG_LEVEL", "info").lower()
    host = os.getenv("ROUTER_HOST", "127.0.0.1")
    port = int(os.getenv("ROUTER_PORT", "10101"))
    uds_path = os.getenv("ROUTER_UDS", "").strip()

    async def _serve() -> None:
        servers = []

        tcp_cfg = uvicorn.Config("router:app", host=host, port=port, log_level=log_level)
        servers.append(uvicorn.Server(tcp_cfg))

        if uds_path:
            # remove stale socket file from a crashed previous run
            try:
                if os.path.exists(uds_path):
                    os.unlink(uds_path)
            except OSError as e:
                log.warning("could not remove stale UDS at %s: %s", uds_path, e)
            uds_cfg = uvicorn.Config("router:app", uds=uds_path, log_level=log_level)
            servers.append(uvicorn.Server(uds_cfg))

        await asyncio.gather(*(s.serve() for s in servers))

    asyncio.run(_serve())
