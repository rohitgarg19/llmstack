"""AWS Bedrock backend for the auto-router.

What this does
==============

A tier in ``models.ini`` declared with ``backend = bedrock`` (or just
``aws_model_id = ...``) is *not* loaded by llama-swap. Instead, when the
router selects that tier, this module:

  1. Builds a per-tier ``boto3`` ``bedrock-runtime`` client using the
     credentials from that tier's :class:`~llmstack.tiers.BedrockConfig`
     (region/profile/explicit creds/assume-role -- whichever the
     operator declared, falling back to boto3's default chain).
  2. Translates the inbound OpenAI-style chat/completions body to
     Bedrock's `Converse`_ shape (``system``, ``messages``,
     ``inferenceConfig``, ``toolConfig``).
  3. Calls :py:meth:`bedrock-runtime.converse_stream` (streaming) or
     :py:meth:`bedrock-runtime.converse` (non-streaming).
  4. Translates the response back to OpenAI's chat-completion or SSE
     ``chat.completion.chunk`` format so existing clients (opencode,
     curl, anything pointed at the router) don't have to know Bedrock
     exists.

Each tier gets its own client + session because each tier may live in
a different region / AWS account / role -- credentials are scoped to
the tier, not globalised in ``[DEFAULT]``.

.. _Converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

Limitations
===========

Text in / text out + tool calling. Multimodal (image) parts are passed
through as text where possible and dropped silently otherwise -- the
local stack is text-first and that's the 95 % case for the agent loop
opencode drives.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from threading import Lock
from typing import Any, AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from llmstack.tiers import BedrockConfig, Tier

log = logging.getLogger("router.bedrock")

USE_NEXT_ENV = "LLMSTACK_USE_NEXT"

# Lazy boto3 import: don't make every llmstack action depend on the AWS
# SDK -- only the router needs it, and only when a bedrock tier is hit.
_boto3 = None  # type: ignore[var-annotated]
_botocore = None  # type: ignore[var-annotated]
_clients: dict[str, Any] = {}
_clients_lock = Lock()


def _use_next() -> bool:
    """Read the ``--next`` channel flag from the router's environment.

    ``llmstack start --next`` exports ``LLMSTACK_USE_NEXT=1`` to the
    router subprocess so that bedrock tiers swap to ``aws_model_id_next``
    in lock-step with gguf tiers swapping to ``hf_file_next``.
    """
    return os.environ.get(USE_NEXT_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class BedrockUnavailableError(RuntimeError):
    """Raised when boto3 is not installed but a bedrock tier was hit."""


def _require_boto3() -> tuple[Any, Any]:
    global _boto3, _botocore
    if _boto3 is not None and _botocore is not None:
        return _boto3, _botocore
    try:
        import boto3 as _b  # type: ignore[import-not-found]
        import botocore  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - import-time only
        raise BedrockUnavailableError(
            "boto3 is required for bedrock-backed tiers; install with "
            "`pip install 'llmstack[bedrock]'`"
        ) from exc
    _boto3, _botocore = _b, botocore
    return _b, botocore


def _client_cache_key(cfg: BedrockConfig) -> str:
    """One client per distinct (profile, region, endpoint) tuple.

    Two tiers that point at the same profile + region collapse onto a
    single boto3 client; switching channel (current/next) builds a new
    one because ``model_id``-bound region differs.
    """
    return "|".join([
        cfg.profile or "",
        cfg.region or "",
        cfg.endpoint_url or "",
    ])


def _build_client(cfg: BedrockConfig):
    """Construct a ``bedrock-runtime`` client for a tier.

    All credential resolution (long-term keys, SSO, role chaining via
    ``role_arn`` + ``source_profile`` in ``~/.aws/config``, MFA, IMDS)
    is delegated to boto3 by passing ``profile_name``. We never touch
    raw secrets here.
    """
    boto3, botocore = _require_boto3()

    session_kwargs: dict[str, Any] = {}
    if cfg.profile:
        session_kwargs["profile_name"] = cfg.profile
    if cfg.region:
        session_kwargs["region_name"] = cfg.region
    session = boto3.session.Session(**session_kwargs)

    client_kwargs: dict[str, Any] = {}
    if cfg.endpoint_url:
        client_kwargs["endpoint_url"] = cfg.endpoint_url
    # Bedrock InvokeModelWithResponseStream / Converse can hold the
    # connection open for a while on slow models -- give it a generous
    # read timeout while keeping connect tight.
    client_kwargs["config"] = botocore.config.Config(
        connect_timeout=10,
        read_timeout=600,
        retries={"max_attempts": 2, "mode": "standard"},
    )
    return session.client("bedrock-runtime", **client_kwargs)


def get_client(cfg: BedrockConfig):
    """Return a process-wide cached client for the given tier config."""
    key = _client_cache_key(cfg)
    with _clients_lock:
        c = _clients.get(key)
        if c is not None:
            return c
        c = _build_client(cfg)
        _clients[key] = c
        return c


# ---------------------------------------------------------------------------
# OpenAI -> Bedrock Converse translation
# ---------------------------------------------------------------------------

def _coerce_text(content: Any) -> str:
    """Turn an OpenAI message ``content`` into a plain string.

    OpenAI accepts either a string or an array of typed parts. We keep
    only the ``text`` parts -- multimodal blobs (``image_url`` etc.) are
    dropped since the Bedrock text models we target won't accept them
    anyway and translating multimodal end-to-end is out of scope.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: list[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    bits.append(t)
        return "\n".join(bits)
    return str(content)


def _system_blocks(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") != "system":
            continue
        text = _coerce_text(m.get("content"))
        if text:
            out.append({"text": text})
    return out


def _converse_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the non-system slice of OpenAI messages to Converse shape.

    Tool calls (assistant) and tool results (role=tool) round-trip via
    ``toolUse`` / ``toolResult`` content blocks.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue

        if role == "tool":
            tool_call_id = m.get("tool_call_id") or m.get("id") or ""
            text = _coerce_text(m.get("content"))
            out.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_call_id,
                        "content": [{"text": text}],
                    }
                }],
            })
            continue

        blocks: list[dict[str, Any]] = []
        text = _coerce_text(m.get("content"))
        if text:
            blocks.append({"text": text})

        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed = {"_raw": raw_args}
                blocks.append({
                    "toolUse": {
                        "toolUseId": tc.get("id") or f"tool_{uuid.uuid4().hex[:12]}",
                        "name": name,
                        "input": parsed if isinstance(parsed, dict) else {"value": parsed},
                    }
                })

        if not blocks:
            continue
        out.append({
            "role": "assistant" if role == "assistant" else "user",
            "content": blocks,
        })
    return out


def _tool_config(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not tools:
        return None
    specs: list[dict[str, Any]] = []
    for t in tools:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        specs.append({
            "toolSpec": {
                "name": name,
                "description": fn.get("description") or "",
                "inputSchema": {"json": fn.get("parameters") or {"type": "object"}},
            }
        })
    if not specs:
        return None
    return {"tools": specs}


def _inference_config(body: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if "temperature" in body:
        try:
            cfg["temperature"] = float(body["temperature"])
        except (TypeError, ValueError):
            pass
    if "top_p" in body:
        try:
            cfg["topP"] = float(body["top_p"])
        except (TypeError, ValueError):
            pass
    if "max_tokens" in body or "max_completion_tokens" in body:
        try:
            cfg["maxTokens"] = int(body.get("max_tokens") or body.get("max_completion_tokens"))
        except (TypeError, ValueError):
            pass
    stop = body.get("stop")
    if isinstance(stop, str):
        cfg["stopSequences"] = [stop]
    elif isinstance(stop, list):
        cfg["stopSequences"] = [s for s in stop if isinstance(s, str)]
    return cfg


def _build_converse_kwargs(tier: Tier, body: dict[str, Any], cfg: BedrockConfig) -> dict[str, Any]:
    """OpenAI-style request body -> kwargs for ``converse[_stream]``.

    ``cfg`` is the channel-resolved :class:`BedrockConfig` (current vs.
    next), passed in so the caller controls the channel and we don't
    re-read the env mid-call.
    """
    assert tier.bedrock is not None
    messages = body.get("messages")
    if not isinstance(messages, list):
        # /v1/completions style: synthesise a single user message
        prompt = body.get("prompt") or ""
        messages = [{"role": "user", "content": prompt}]

    converse_kwargs: dict[str, Any] = {
        "modelId": cfg.model_id,
        "messages": _converse_messages(messages),
    }
    sys_blocks = _system_blocks(messages)
    if sys_blocks:
        converse_kwargs["system"] = sys_blocks

    inference = _inference_config(body)
    if inference:
        converse_kwargs["inferenceConfig"] = inference

    tools = _tool_config(body.get("tools"))
    if tools:
        converse_kwargs["toolConfig"] = tools
    return converse_kwargs


# ---------------------------------------------------------------------------
# Bedrock -> OpenAI translation
# ---------------------------------------------------------------------------

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
}


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now_unix() -> int:
    return int(time.time())


def _openai_message_from_converse(resp: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Pull text + tool calls out of a non-streaming Converse response."""
    msg = (resp.get("output") or {}).get("message") or {}
    blocks = msg.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for b in blocks:
        if "text" in b and b["text"]:
            text_parts.append(b["text"])
        elif "toolUse" in b:
            tu = b["toolUse"] or {}
            tool_calls.append({
                "id": tu.get("toolUseId") or f"tool_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tu.get("name") or "",
                    "arguments": json.dumps(tu.get("input") or {}),
                },
            })
    out: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        out["tool_calls"] = tool_calls
    finish = _STOP_REASON_MAP.get(resp.get("stopReason") or "", "stop")
    return out, finish


# ---------------------------------------------------------------------------
# Dispatch entry points (called by app.py)
# ---------------------------------------------------------------------------

async def dispatch(req: Request, tier: Tier, body: dict[str, Any]) -> StreamingResponse | JSONResponse:
    """Top-level entry: turn an OpenAI-style request into a Bedrock call.

    Streams when the request asked for ``stream: true``; otherwise
    returns a single chat-completion JSON object.
    """
    if tier.bedrock is None:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"tier {tier.name!r} has backend=bedrock but no aws_model_id"}},
        )

    streaming = bool(body.get("stream"))
    use_next = _use_next()
    cfg = tier.bedrock.resolved(use_next=use_next)
    channel = "next" if (use_next and tier.bedrock.has_next) else "current"
    converse_kwargs = _build_converse_kwargs(tier, body, cfg)
    log.info(
        "bedrock dispatch tier=%s model=%s region=%s channel=%s stream=%s",
        tier.name, cfg.model_id, cfg.region or "(default)", channel, streaming,
    )

    try:
        client = get_client(cfg)
    except BedrockUnavailableError as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc)}})

    if streaming:
        return await _stream_response(client, tier, converse_kwargs)
    return await _complete_response(client, tier, converse_kwargs)


async def _complete_response(client: Any, tier: Tier, converse_kwargs: dict[str, Any]) -> JSONResponse:
    try:
        resp = await asyncio.to_thread(client.converse, **converse_kwargs)
    except Exception as exc:  # noqa: BLE001 - surface upstream error verbatim
        log.warning("bedrock converse failed: %s", exc)
        return JSONResponse(status_code=502, content={"error": _error_payload(exc)})

    message, finish = _openai_message_from_converse(resp)
    usage_in = (resp.get("usage") or {})
    payload = {
        "id":      _completion_id(),
        "object":  "chat.completion",
        "created": _now_unix(),
        "model":   tier.name,
        "choices": [{
            "index":         0,
            "message":       message,
            "finish_reason": finish or "stop",
        }],
        "usage": {
            "prompt_tokens":     int(usage_in.get("inputTokens") or 0),
            "completion_tokens": int(usage_in.get("outputTokens") or 0),
            "total_tokens":      int(usage_in.get("totalTokens") or 0),
        },
    }
    return JSONResponse(content=payload)


def _error_payload(exc: Exception) -> dict[str, Any]:
    out: dict[str, Any] = {"message": str(exc), "type": exc.__class__.__name__}
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error") or {}
        if err.get("Code"):
            out["code"] = err["Code"]
    return out


async def _stream_response(client: Any, tier: Tier, converse_kwargs: dict[str, Any]) -> StreamingResponse:
    completion_id = _completion_id()
    created = _now_unix()
    model_label = tier.name

    def _sse(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()

    def _frame(delta: dict[str, Any], *, finish: str | None = None) -> dict[str, Any]:
        choice: dict[str, Any] = {
            "index":         0,
            "delta":         delta,
            "finish_reason": finish,
        }
        return {
            "id":      completion_id,
            "object":  "chat.completion.chunk",
            "created": created,
            "model":   model_label,
            "choices": [choice],
        }

    async def gen() -> AsyncIterator[bytes]:
        # Open the converse stream in a worker thread; the EventStream
        # iterator is sync, so we read it off the loop and bridge to an
        # asyncio queue.
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()
        sentinel = object()

        def _pump() -> None:
            try:
                resp = client.converse_stream(**converse_kwargs)
                stream = resp.get("stream")
                if stream is None:
                    raise RuntimeError("converse_stream returned no stream")
                for event in stream:
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(queue.put(("__error__", exc)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        pump_task = asyncio.create_task(asyncio.to_thread(_pump))

        # First chunk: announce the assistant role so OpenAI clients can
        # initialise their accumulator.
        yield _sse(_frame({"role": "assistant"}))

        # Per-content-block state: index -> "text" | "tool_use"
        block_kinds: dict[int, str] = {}
        # tool_use blocks need the OpenAI tool_calls index to map to.
        tool_call_index: dict[int, int] = {}
        next_tool_call_index = 0
        finish_reason: str | None = None

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                    err = item[1]
                    log.warning("bedrock stream failed: %s", err)
                    yield _sse(_frame({}, finish="error"))
                    yield b"data: " + json.dumps({"error": _error_payload(err)}).encode() + b"\n\n"
                    return

                event = item
                if "messageStart" in event:
                    continue
                if "contentBlockStart" in event:
                    cbs = event["contentBlockStart"]
                    idx = cbs.get("contentBlockIndex", 0)
                    start = cbs.get("start") or {}
                    if "toolUse" in start:
                        block_kinds[idx] = "tool_use"
                        oai_idx = next_tool_call_index
                        tool_call_index[idx] = oai_idx
                        next_tool_call_index += 1
                        tu = start["toolUse"]
                        yield _sse(_frame({
                            "tool_calls": [{
                                "index": oai_idx,
                                "id":    tu.get("toolUseId") or f"tool_{uuid.uuid4().hex[:12]}",
                                "type":  "function",
                                "function": {"name": tu.get("name") or "", "arguments": ""},
                            }]
                        }))
                    else:
                        block_kinds[idx] = "text"
                    continue

                if "contentBlockDelta" in event:
                    cbd = event["contentBlockDelta"]
                    idx = cbd.get("contentBlockIndex", 0)
                    delta = cbd.get("delta") or {}
                    kind = block_kinds.get(idx, "text")
                    if kind == "text":
                        text = delta.get("text")
                        if text:
                            yield _sse(_frame({"content": text}))
                    elif kind == "tool_use":
                        # toolUse deltas carry partial JSON in `input`.
                        tu = delta.get("toolUse") or {}
                        partial = tu.get("input")
                        if partial is None:
                            partial = ""
                        if not isinstance(partial, str):
                            partial = json.dumps(partial)
                        if partial:
                            yield _sse(_frame({
                                "tool_calls": [{
                                    "index":    tool_call_index.get(idx, 0),
                                    "function": {"arguments": partial},
                                }]
                            }))
                    continue

                if "contentBlockStop" in event:
                    continue

                if "messageStop" in event:
                    finish_reason = _STOP_REASON_MAP.get(
                        event["messageStop"].get("stopReason") or "", "stop",
                    )
                    continue

                if "metadata" in event:
                    # Could attach token counts here via an
                    # `x-llmstack-usage` event, but OpenAI's chunk schema
                    # has no usage field on intermediate chunks; skip.
                    continue
        finally:
            await pump_task

        yield _sse(_frame({}, finish=finish_reason or "stop"))
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# /v1/models metadata
# ---------------------------------------------------------------------------

def model_descriptor(tier: Tier) -> dict[str, Any]:
    """Return an OpenAI-style ``/v1/models`` entry for a bedrock tier."""
    assert tier.bedrock is not None
    use_next = _use_next()
    active = tier.bedrock.resolved(use_next=use_next)
    channel = "next" if (use_next and tier.bedrock.has_next) else "current"
    metadata: dict[str, Any] = {
        "model_id": active.model_id,
        "region":   active.region or os.environ.get("AWS_REGION") or "",
        "ctx_size": tier.ctx_size,
        "channel":  channel,
    }
    if tier.bedrock.has_next:
        metadata["model_id_next"] = tier.bedrock.model_id_next
        if tier.bedrock.region_next:
            metadata["region_next"] = tier.bedrock.region_next
    return {
        "id":          tier.name,
        "object":      "model",
        "created":     0,
        "owned_by":    "aws-bedrock",
        "name":        tier.description,
        "description": tier.description,
        "tier":        tier.role,
        "backend":     "bedrock",
        "metadata":    metadata,
    }
