"""The course gateway: an OpenAI- *and* Anthropic-dialect endpoint you can run.

    uvicorn app.main:app --port 8080          # or: docker compose up gateway

It exists so the whole course runs offline and every drill is reproducible on
demand — see ``brain.py`` for the honest account of what is real here and what is
simulated, and read that before quoting any number it produced.

Endpoints
---------
``POST /v1/chat/completions``  OpenAI dialect, streaming and not, tools, structured
                               outputs, ``usage`` with ``prompt_tokens_details``.
``POST /v1/messages``          Anthropic Messages dialect: top-level ``system``,
                               required ``max_tokens``, typed content blocks,
                               ``stop_reason``, ``cache_read_input_tokens``.
``GET  /v1/models``            Model listing.
``POST /admin/fault``          Fault injection: the 429 storm and the outage drill,
                               on a timer, targeted at one model or all of them.
``GET  /admin/stats``          Requests, tokens and cache hits since the last reset.
``POST /admin/reset``          Clears stats, faults, and the prompt cache.
``GET  /healthz``              Liveness.

Prompt caching is modelled the way providers actually do it: a byte-stable prefix
(the system message plus the tool schemas) above a minimum length, with a TTL. One
dynamic byte in the prefix and the hit disappears — which is the point of Lab 6
task 2 and cannot be taught with a mock that always says "cached".
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any

import tiktoken
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import brain

app = FastAPI(title="SDA-AIE-213 course gateway (simulator)", version="1.0.0")

ENC = tiktoken.get_encoding("o200k_base")

#: Wall-clock compression. 1.0 = the simulated latencies as written (roughly what a
#: mid-sized hosted model feels like); the default keeps a 50-minute lab moving.
SPEED = float(os.environ.get("MOCKGW_SPEED", "0.2"))
FAST = os.environ.get("MOCKGW_FAST") == "1"  # tests and CI: no sleeping at all

#: Providers only cache prefixes past a minimum length. So does this.
MIN_CACHEABLE_TOKENS = int(os.environ.get("MOCKGW_MIN_CACHEABLE_TOKENS", "1024"))
CACHE_TTL_S = float(os.environ.get("MOCKGW_CACHE_TTL_S", "300"))

KNOWN_MODELS = ["course-flagship", "course-small", "course-anthropic", "retail_support-onprem"]

_prefix_cache: dict[str, tuple[float, int]] = {}
_fault: dict[str, Any] = {"mode": "off"}
_stats: dict[str, Any] = {
    "requests": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cached_input_tokens": 0,
    "cache_hits": 0,
    "faults_served": 0,
    "by_model": {},
}


def ntokens(text: str) -> int:
    return len(ENC.encode(text or ""))


def _sleep(seconds: float) -> None:
    if FAST or seconds <= 0:
        return
    time.sleep(seconds * SPEED)


# --------------------------------------------------------------------------
# Fault injection — the drills, on a timer
# --------------------------------------------------------------------------


def _fault_response(model: str) -> JSONResponse | None:
    mode = _fault.get("mode", "off")
    if mode == "off":
        return None
    if _fault.get("until") and time.time() > _fault["until"]:
        _fault["mode"] = "off"
        return None
    target = _fault.get("model")
    if target and not model.startswith(target):
        return None
    _stats["faults_served"] += 1
    if mode == "rate_limit":
        retry_after = _fault.get("retry_after", 2)
        return JSONResponse(
            status_code=429,
            headers={"retry-after": str(retry_after)},
            content={
                "error": {
                    "message": "Rate limit reached for this key. Slow down.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )
    if mode == "overload":
        return JSONResponse(
            status_code=529,
            content={"error": {"message": "Overloaded", "type": "overloaded_error"}},
        )
    if mode == "timeout":
        time.sleep(float(_fault.get("seconds_per_request", 45)))
        return JSONResponse(status_code=504, content={"error": {"message": "Gateway timeout"}})
    if mode == "server_error":
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Upstream unavailable", "type": "api_error"}},
        )
    return None


@app.post("/admin/fault")
async def set_fault(payload: dict) -> dict:
    """``{"mode": "rate_limit"|"overload"|"timeout"|"server_error"|"off",
          "seconds": 180, "model": "course-flagship", "retry_after": 2}``"""
    mode = payload.get("mode", "off")
    _fault.clear()
    _fault["mode"] = mode
    if mode != "off":
        _fault["until"] = time.time() + float(payload.get("seconds", 120))
        if payload.get("model"):
            _fault["model"] = payload["model"]
        if payload.get("retry_after") is not None:
            _fault["retry_after"] = payload["retry_after"]
        if payload.get("seconds_per_request") is not None:
            _fault["seconds_per_request"] = payload["seconds_per_request"]
    return {"fault": dict(_fault)}


@app.get("/admin/stats")
async def stats() -> dict:
    return {**_stats, "fault": dict(_fault), "prefixes_cached": len(_prefix_cache)}


@app.post("/admin/reset")
async def reset() -> dict:
    _prefix_cache.clear()
    _fault.clear()
    _fault["mode"] = "off"
    _stats.update(
        requests=0,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        cache_hits=0,
        faults_served=0,
        by_model={},
    )
    return {"reset": True}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "models": KNOWN_MODELS, "speed": SPEED, "fast": FAST}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1750000000, "owned_by": "sda-aie-213"}
            for m in KNOWN_MODELS
        ],
    }


# --------------------------------------------------------------------------
# Prompt cache
# --------------------------------------------------------------------------


def prefix_hit(model: str, prefix_text: str) -> int:
    """Cached-token count for this prefix, or 0. Byte-stable or nothing."""
    prefix_tokens = ntokens(prefix_text)
    if prefix_tokens < MIN_CACHEABLE_TOKENS:
        return 0
    key = model + ":" + hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()
    now = time.time()
    entry = _prefix_cache.get(key)
    if entry and now - entry[0] < CACHE_TTL_S:
        _prefix_cache[key] = (now, prefix_tokens)  # TTL refreshed on hit, as providers do
        _stats["cache_hits"] += 1
        return prefix_tokens
    _prefix_cache[key] = (now, prefix_tokens)
    return 0


# --------------------------------------------------------------------------
# The dispatcher: which task is this request?
# --------------------------------------------------------------------------


def schema_name(response_format: dict | None) -> str:
    if not response_format:
        return ""
    if response_format.get("type") == "json_schema":
        return (response_format.get("json_schema") or {}).get("name", "")
    return response_format.get("type", "")


def run_brain(
    *,
    system: str,
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    response_format: dict | None,
) -> brain.BrainResult:
    user_texts = [
        m["content"]
        for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]
    user_text = user_texts[-1] if user_texts else ""
    name = schema_name(response_format)
    repairing = "failed validation" in user_text.lower()
    # On a repair turn the LAST user message is the validation feedback, not the
    # customer. Extracting from it produces a ticket about the error message —
    # complete with a city lifted out of the enum listed in the error. Repair
    # loops re-read the ORIGINAL input; this line is the whole difference.
    source_text = user_texts[0] if repairing and user_texts else user_text

    if name == "service_ticket":
        return brain.extract_ticket(system, source_text, model, repairing)
    if name == "guard_verdict":
        return brain.classify_guard(user_text, model)
    if name == "route_verdict":
        return brain.classify_route(user_text, model)
    if name == "judge_verdict":
        return brain.judge(system, user_text, model)

    if tools:
        result = brain.decide_tools(messages, tools, model)
        if result.text is not None or result.tool_calls:
            return result

    return brain.answer_faq(system, user_text, model)


def account(model: str, system: str, messages: list[dict], tools: list[dict] | None, output: str):
    prefix_text = system + json.dumps(tools or [], sort_keys=True, ensure_ascii=False)
    cached = prefix_hit(model, prefix_text)
    body_text = "".join(
        m.get("content") or ""
        for m in messages
        if isinstance(m.get("content"), str) and m.get("role") != "system"
    )
    input_tokens = ntokens(system) + ntokens(body_text) + 4 * len(messages)
    output_tokens = max(ntokens(output), 1)
    cached = min(cached, input_tokens)

    _stats["requests"] += 1
    _stats["input_tokens"] += input_tokens
    _stats["output_tokens"] += output_tokens
    _stats["cached_input_tokens"] += cached
    _stats["by_model"][model] = _stats["by_model"].get(model, 0) + 1
    return input_tokens, output_tokens, cached


def simulate_latency(model: str, input_tokens: int, cached: int, output_tokens: int):
    """Returns (ttft_seconds, per_token_seconds).

    TTFT is dominated by prompt processing, which is why a cache hit halves it —
    the cached prefix does not get reprocessed. Generation time is linear in the
    output. Module 6 §2, made observable.
    """
    tier = brain.tier_for(model)
    fresh = max(input_tokens - cached, 0)
    ttft = tier.latency_base_ms + (fresh / 1000.0) * tier.prompt_ms_per_1k
    return ttft / 1000.0, tier.latency_per_token_ms / 1000.0


# --------------------------------------------------------------------------
# OpenAI dialect
# --------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model = payload.get("model", "course-flagship")
    if (fault := _fault_response(model)) is not None:
        return fault

    messages = payload.get("messages") or []
    tools = payload.get("tools")
    response_format = payload.get("response_format")
    max_tokens = int(payload.get("max_tokens") or 1024)
    system = "\n\n".join(
        m.get("content") or "" for m in messages if m.get("role") == "system"
    )

    result = run_brain(
        system=system,
        messages=messages,
        model=model,
        tools=tools,
        response_format=response_format,
    )
    text = result.text or ""
    tool_calls = [
        {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
            },
        }
        for c in result.tool_calls
    ]
    payload_for_tokens = text or json.dumps([c["function"] for c in tool_calls], ensure_ascii=False)
    input_tokens, output_tokens, cached = account(model, system, messages, tools, payload_for_tokens)

    finish = result.finish_reason
    if finish == "stop" and output_tokens > max_tokens:
        # Truncation is silent unless the adapter reads finish_reason. Make it real.
        keep = ENC.decode(ENC.encode(text)[:max_tokens])
        text, output_tokens, finish = keep, max_tokens, "length"

    ttft_s, per_token_s = simulate_latency(model, input_tokens, cached, output_tokens)

    if payload.get("stream"):
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(
            _sse_openai(
                model=model,
                text=text,
                tool_calls=tool_calls,
                finish=finish,
                ttft_s=ttft_s,
                per_token_s=per_token_s,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
        )

    _sleep(ttft_s + per_token_s * output_tokens)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def _sse_openai(
    *, model, text, tool_calls, finish, ttft_s, per_token_s, usage, include_usage
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

    def frame(delta: dict, finish_reason=None) -> str:
        return "data: " + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            },
            ensure_ascii=False,
        ) + "\n\n"

    _sleep(ttft_s)
    yield frame({"role": "assistant"})
    if tool_calls:
        yield frame({"tool_calls": tool_calls})
    else:
        words = text.split(" ")
        for i, word in enumerate(words):
            _sleep(per_token_s * 1.4)
            yield frame({"content": word + (" " if i < len(words) - 1 else "")})
    yield frame({}, finish_reason=finish)
    if include_usage:
        yield "data: " + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [],
                "usage": usage,
            },
            ensure_ascii=False,
        ) + "\n\n"
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------
# Anthropic Messages dialect
# --------------------------------------------------------------------------


def _flatten_anthropic(payload: dict) -> tuple[str, list[dict], list[dict] | None]:
    """Anthropic's typed blocks -> the flat shape the brain reasons over."""
    system_raw = payload.get("system")
    if isinstance(system_raw, list):
        system = "\n\n".join(b.get("text", "") for b in system_raw if b.get("type") == "text")
    else:
        system = system_raw or ""

    messages: list[dict] = []
    for m in payload.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        text_parts, is_tool_result = [], False
        for block in content or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                is_tool_result = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": block.get("content")
                        if isinstance(block.get("content"), str)
                        else json.dumps(block.get("content"), ensure_ascii=False),
                    }
                )
        if text_parts and not is_tool_result:
            messages.append({"role": m["role"], "content": "\n".join(text_parts)})

    tools = None
    if payload.get("tools"):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in payload["tools"]
        ]
    return system, messages, tools


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    payload = await request.json()
    model = payload.get("model", "course-anthropic")
    if (fault := _fault_response(model)) is not None:
        return fault
    if payload.get("max_tokens") is None:
        # max_tokens is REQUIRED in this dialect. An adapter that forwards None
        # deserves to find that out here rather than in production.
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "max_tokens: field required"},
            },
        )

    system, messages, tools = _flatten_anthropic(payload)
    # Structured outputs arrive on output_config in this dialect, and carry no
    # name field — the contract's name rides inside the schema's title instead.
    response_format = None
    schema = ((payload.get("output_config") or {}).get("format") or {}).get("schema")
    if schema:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.get("title", ""), "schema": schema},
        }
    result = run_brain(
        system=system,
        messages=messages,
        model=model,
        tools=tools,
        response_format=response_format,
    )

    blocks: list[dict] = []
    if result.text:
        blocks.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": call["name"],
                "input": call["arguments"],
            }
        )
    if not blocks:
        blocks = [{"type": "text", "text": ""}]

    output_text = result.text or json.dumps(
        [c["arguments"] for c in result.tool_calls], ensure_ascii=False
    )
    input_tokens, output_tokens, cached = account(model, system, messages, tools, output_text)
    stop_reason = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}.get(
        result.finish_reason, "end_turn"
    )
    ttft_s, per_token_s = simulate_latency(model, input_tokens, cached, output_tokens)

    body = {
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        },
    }
    if payload.get("stream"):
        return StreamingResponse(
            _sse_anthropic(body, ttft_s, per_token_s), media_type="text/event-stream"
        )
    _sleep(ttft_s + per_token_s * output_tokens)
    return body


def _sse_anthropic(body: dict, ttft_s: float, per_token_s: float):
    def event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n"

    start = {k: v for k, v in body.items() if k != "content"}
    start["content"] = []
    start["usage"] = {"input_tokens": body["usage"]["input_tokens"], "output_tokens": 0}
    _sleep(ttft_s)
    yield event("message_start", {"type": "message_start", "message": start})
    for index, block in enumerate(body["content"]):
        if block["type"] == "text":
            yield event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            words = block["text"].split(" ")
            for i, word in enumerate(words):
                _sleep(per_token_s * 1.4)
                yield event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": word + (" " if i < len(words) - 1 else ""),
                        },
                    },
                )
        else:
            yield event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"], ensure_ascii=False),
                    },
                },
            )
        yield event("content_block_stop", {"type": "content_block_stop", "index": index})
    yield event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": body["usage"]["output_tokens"]},
        },
    )
    yield event("message_stop", {"type": "message_stop"})
