"""The Anthropic Messages dialect, with the differences made explicit.

Four of them are load-bearing, and an adapter that pretends otherwise produces
subtle bugs (Module 2 §1):

1. ``system`` is a **top-level parameter**, not a message in ``messages``.
2. ``max_tokens`` is **required**, not optional.
3. Content is a list of typed **blocks** (``text``, ``tool_use``, ``tool_result``),
   not a string.
4. The finish signal is ``stop_reason`` (``end_turn``/``max_tokens``/``tool_use``/
   ``refusal``), not ``finish_reason``.

This is exactly why the normalised ``LLMRequest``/``LLMResponse`` exists.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import anthropic
from anthropic import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from retail_support.config import ModelRoute
from retail_support.llm.interfaces import (
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    Usage,
)

RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}


def _normalise_stop(reason: str | None) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "refusal",
    }.get(reason or "", "error")


def _wrap(exc: Exception) -> LLMError:
    if isinstance(exc, RateLimitError):
        retry_after = None
        try:
            value = exc.response.headers.get("retry-after")
            retry_after = float(value) if value is not None else None
        except Exception:  # pragma: no cover
            pass
        return LLMError(str(exc), status=429, retryable=True, retry_after=retry_after)
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return LLMError(str(exc), retryable=True)
    if isinstance(exc, APIStatusError):
        return LLMError(
            str(exc),
            status=exc.status_code,
            retryable=exc.status_code in RETRYABLE_STATUSES,
        )
    return LLMError(str(exc), retryable=False, raw=exc)


def split_system(messages: list[Message]) -> tuple[str, list[dict]]:
    """System messages hoist out; everything else becomes typed content blocks."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns: list[dict] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            turns.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
            continue
        blocks: list[dict] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for call in m.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments or "{}"),
                }
            )
        turns.append({"role": m.role, "content": blocks or [{"type": "text", "text": ""}]})
    return system, turns


def to_anthropic_tools(tools: list[dict] | None) -> list[dict]:
    """OpenAI-dialect tool schemas -> Anthropic's flatter shape."""
    out = []
    for t in tools or []:
        fn = t.get("function", t)
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


class AnthropicClient:
    def __init__(self, route: ModelRoute) -> None:
        self._route = route
        kwargs: dict = {
            "api_key": route.api_key.get_secret_value() or "not-needed",
            "timeout": route.timeout_s,
            "max_retries": 0,  # policy lives in ResilientClient
        }
        if route.base_url:
            kwargs["base_url"] = route.base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.last_ttft_ms: float | None = None

    @property
    def route_name(self) -> str:
        return self._route.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, turns = split_system(request.messages)
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": self._route.resolve(request.model_alias),
            "messages": turns,
            "max_tokens": request.max_tokens,  # REQUIRED here, unlike OpenAI-compat
        }
        # A fifth difference, and a live one: as of anthropic 1.x the Messages API
        # no longer takes `temperature` or `top_p` at all — sampling moved behind
        # `output_config.effort`. Forwarding request.temperature here raises a
        # TypeError, which is how this was found. The normalised LLMRequest keeps
        # the field because the OpenAI dialect still has it; the adapter is the
        # right place for the divergence to stop. Do not "fix" this by deleting
        # temperature from LLMRequest, and do not smuggle it through extra_body.
        if system:
            if request.cache_prefix_messages:
                # Module 6 §3: mark the stable prefix explicitly on this dialect.
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = to_anthropic_tools(request.tools)
        if request.response_format:
            # Structured outputs on this dialect: a JSON schema on output_config,
            # not a response_format on the request.
            schema = (request.response_format.get("json_schema") or {}).get("schema")
            if schema:
                kwargs["output_config"] = {
                    "format": {"type": "json_schema", "schema": schema}
                }
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _wrap(exc) from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input, ensure_ascii=False),
                    )
                )
        return LLMResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=_normalise_stop(resp.stop_reason),
            model_id=resp.model,
            usage=Usage(
                input_tokens=resp.usage.input_tokens or 0,
                output_tokens=resp.usage.output_tokens or 0,
                cached_input_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
            route=self._route.name,
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Anthropic streams typed events; we normalise to the same StreamChunk."""
        system, turns = split_system(request.messages)
        t0 = time.perf_counter()
        first_token_at: float | None = None
        model_id = self._route.resolve(request.model_alias)
        usage = Usage()
        stop: str | None = None
        kwargs: dict = {
            "model": model_id,
            "messages": turns,
            "max_tokens": request.max_tokens,
        }
        if system:
            kwargs["system"] = system
        try:
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        self.last_ttft_ms = (first_token_at - t0) * 1000
                    yield StreamChunk(delta=text, model_id=model_id)
                final = stream.get_final_message()
                usage = Usage(
                    input_tokens=final.usage.input_tokens or 0,
                    output_tokens=final.usage.output_tokens or 0,
                    cached_input_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                )
                stop = final.stop_reason
        except Exception as exc:
            raise _wrap(exc) from exc

        yield StreamChunk(
            final=True,
            usage=usage,
            ttft_ms=self.last_ttft_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            model_id=model_id,
            finish_reason=_normalise_stop(stop),
        )
