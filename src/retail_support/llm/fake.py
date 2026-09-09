"""``FakeClient`` — the third implementation of the boundary, and the one that
makes the test suite fast, free and deterministic.

It is scripted, not clever: you tell it what to return, in order. Everything a
walkthrough needs to test — a tool call, a malformed argument, an endless loop, a 429
storm, a schema violation the repair loop must fix — is a one-line script here.

This is the payoff of Module 1's protocol you feel first: you can
test the whole application without a network, a key, or a bill.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Iterator

from retail_support.llm.interfaces import (
    LLMError,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)


class FakeClient:
    def __init__(self, model_id: str = "fake-model", route: str = "fake") -> None:
        self.model_id = model_id
        self.route = route
        self._queue: list[LLMResponse | Exception] = []
        self._default: Callable[[LLMRequest], LLMResponse] | None = None
        self._repeat: LLMResponse | Exception | None = None
        self.requests: list[LLMRequest] = []
        self._ids = itertools.count(1)

    # --- scripting -------------------------------------------------------
    def script_text(self, text: str, *, finish: str = "stop", tokens: tuple[int, int] = (100, 40)):
        self._queue.append(
            LLMResponse(
                text=text,
                finish_reason=finish,
                model_id=self.model_id,
                usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1]),
                latency_ms=1.0,
                route=self.route,
            )
        )
        return self

    def script_json(self, payload: dict | str, **kwargs):
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return self.script_text(text, **kwargs)

    def script_tool_call(self, name: str, arguments: dict | str, *, text: str = ""):
        args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        self._queue.append(
            LLMResponse(
                text=text or None,
                tool_calls=[ToolCall(id=f"call_{next(self._ids)}", name=name, arguments=args)],
                finish_reason="tool_calls",
                model_id=self.model_id,
                usage=Usage(input_tokens=120, output_tokens=30),
                latency_ms=1.0,
                route=self.route,
            )
        )
        return self

    def script_endless_tool_calls(self, name: str, arguments: dict | None = None):
        """The stubborn model. Bounds are the only thing that stops it."""
        args = json.dumps(arguments or {"order_reference": "ORD-1000001"}, ensure_ascii=False)
        self._repeat = LLMResponse(
            tool_calls=[ToolCall(id="call_loop", name=name, arguments=args)],
            finish_reason="tool_calls",
            model_id=self.model_id,
            usage=Usage(input_tokens=120, output_tokens=30),
            latency_ms=1.0,
            route=self.route,
        )
        return self

    def script_error(self, error: Exception, times: int = 1):
        for _ in range(times):
            self._queue.append(error)
        return self

    def script_rate_limit(self, times: int = 1, retry_after: float | None = 2.0):
        return self.script_error(
            LLMError("429 rate limit", status=429, retryable=True, retry_after=retry_after),
            times=times,
        )

    def always(self, fn: Callable[[LLMRequest], LLMResponse]):
        """Fall back to a function of the request when the queue empties."""
        self._default = fn
        return self

    # --- the LLMClient protocol -----------------------------------------
    @property
    def route_name(self) -> str:
        return self.route

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._queue:
            item = self._queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._repeat is not None:
            if isinstance(self._repeat, Exception):
                raise self._repeat
            return self._repeat.model_copy()
        if self._default is not None:
            return self._default(request)
        raise AssertionError(
            "FakeClient ran out of scripted responses — script one more, or set .always()"
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        response = self.complete(request)
        t0 = time.perf_counter()
        for word in (response.text or "").split(" "):
            yield StreamChunk(delta=word + " ", model_id=response.model_id)
        yield StreamChunk(
            final=True,
            usage=response.usage,
            ttft_ms=0.5,
            total_ms=(time.perf_counter() - t0) * 1000,
            model_id=response.model_id,
            finish_reason=response.finish_reason,
        )
