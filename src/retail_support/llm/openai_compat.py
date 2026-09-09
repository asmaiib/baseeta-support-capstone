"""One adapter, three deployments: OpenAI cloud, the course gateway, and the
classroom vLLM server. The ``base_url`` is the only difference — that is the whole
of Module 1's boundary argument, and Module 2 proves it with a config edit.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from retail_support.config import ModelRoute
from retail_support.llm.interfaces import (
    LLMError,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

#: 400 is a bug; retrying it is the same bug, three times slower, at 3x the cost.
RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _normalise_finish(reason: str | None) -> str:
    return {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "content_filter": "refusal",
    }.get(reason or "", "error")


def _retry_after(exc: APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def _wrap(exc: Exception) -> LLMError:
    """The error taxonomy every adapter must map (Module 2 §3), in one place."""
    if isinstance(exc, RateLimitError):
        return LLMError(str(exc), status=429, retryable=True, retry_after=_retry_after(exc))
    if isinstance(exc, APITimeoutError):
        return LLMError(str(exc), status=None, retryable=True)
    if isinstance(exc, APIConnectionError):
        return LLMError(str(exc), status=None, retryable=True)
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        return LLMError(
            str(exc),
            status=status,
            retryable=status in RETRYABLE_STATUSES,
            retry_after=_retry_after(exc),
        )
    return LLMError(str(exc), retryable=False, raw=exc)


def _wire_messages(request: LLMRequest) -> list[dict]:
    out: list[dict] = []
    for m in request.messages:
        msg: dict = {"role": m.role, "content": m.content}
        if m.role == "tool":
            msg["tool_call_id"] = m.tool_call_id
        if m.name and m.role != "tool":
            msg["name"] = m.name
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in m.tool_calls
            ]
            # An assistant turn that only requests tools carries no text.
            msg["content"] = m.content or None
        out.append(msg)
    return out


class OpenAICompatClient:
    """Speaks ``POST /v1/chat/completions`` — the de-facto wire standard."""

    def __init__(self, route: ModelRoute) -> None:
        self._route = route
        self._client = OpenAI(
            base_url=route.base_url,
            api_key=route.api_key.get_secret_value() or "not-needed",
            timeout=httpx.Timeout(route.timeout_s, connect=route.connect_timeout_s),
            max_retries=0,  # WE own retry policy (retail_support.llm.resilient), not the SDK
        )
        self.last_ttft_ms: float | None = None

    @property
    def route_name(self) -> str:
        return self._route.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": self._route.resolve(request.model_alias),
            "messages": _wire_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,  # ALWAYS bounded
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if request.response_format:
            kwargs["response_format"] = request.response_format
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _wrap(exc) from exc

        choice = resp.choices[0]
        usage = resp.usage
        cached = 0
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
        return LLMResponse(
            text=choice.message.content,
            tool_calls=[
                ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
                for c in (choice.message.tool_calls or [])
            ],
            finish_reason=_normalise_finish(choice.finish_reason),
            model_id=resp.model,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cached_input_tokens=cached,
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
            route=self._route.name,
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Yields text deltas, then one final frame carrying usage.

        ``stream_options={"include_usage": True}`` is not optional: without it the
        usage never arrives and the cost meter silently undercounts the majority of
        traffic — the classic streaming cost-accounting bug (Module 2 §3).
        """
        t0 = time.perf_counter()
        first_token_at: float | None = None
        usage = Usage()
        model_id = self._route.resolve(request.model_alias)
        finish: str | None = None
        try:
            stream = self._client.chat.completions.create(
                model=model_id,
                messages=_wire_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:  # final frame
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        cached_input_tokens=getattr(
                            getattr(chunk.usage, "prompt_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0,
                    )
                    continue
                if not chunk.choices:
                    continue
                model_id = chunk.model or model_id
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        self.last_ttft_ms = (first_token_at - t0) * 1000
                    yield StreamChunk(delta=delta, model_id=model_id)
        except Exception as exc:
            raise _wrap(exc) from exc

        yield StreamChunk(
            final=True,
            usage=usage,
            ttft_ms=self.last_ttft_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            model_id=model_id,
            finish_reason=_normalise_finish(finish),
        )
