"""Reliability policy, once, at the boundary (Module 1 §5).

Retries with exponential backoff and jitter, ``Retry-After`` honoured, retryable
errors only, a capped number of attempts, then a fallback chain ending in a
degraded-but-honest answer. Nowhere else in the codebase mentions retries — grep
for ``sleep`` and this file should be the only hit outside tests.

A *degraded-but-honest* answer beats an error page. But the fallback hop must be
evaluated too (Module 5), or failover becomes silent quality decay.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator

from retail_support.llm.interfaces import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    Usage,
)
from retail_support.observability import get_logger

log = get_logger(__name__)

DEGRADED_ANSWER = {
    "ar": (
        "أعتذر، لا أستطيع الإجابة في هذه اللحظة بسبب عُطل مؤقت. "
        "يمكنك مراجعة صفحة الأسئلة الشائعة أو زيارة أقرب فرع."
    ),
    "en": (
        "I can't answer right now because of a temporary fault. "
        "You can check the FAQ page or visit your nearest store."
    ),
}


class AllHopsExhausted(LLMError):
    """Every hop in the chain failed. The caller decides how to degrade."""


class ResilientClient:
    """Decorator client: retry + fallback chain. Wraps ANY ``LLMClient``."""

    def __init__(
        self,
        chain: list[tuple[str, LLMClient]],
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 20.0,
        sleep=time.sleep,
    ) -> None:
        if not chain:
            raise ValueError("ResilientClient needs at least one hop")
        self._chain = chain  # [("primary", client), ("on_prem", client), ...]
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep  # injected so tests do not actually wait
        self.attempts = 0
        self.fallbacks = 0

    @property
    def route_name(self) -> str:
        return self._chain[0][0]

    def _delay_for(self, attempt: int, error: LLMError) -> float:
        if error.retry_after is not None:  # honour the header before guessing
            return min(error.retry_after, self._max_delay)
        delay = self._base_delay * (2 ** (attempt - 1))
        return min(delay * random.uniform(0.5, 1.5), self._max_delay)  # noqa: S311 - jitter

    def complete(self, request: LLMRequest) -> LLMResponse:
        last_error: LLMError | None = None
        for hop_index, (hop_name, client) in enumerate(self._chain):
            for attempt in range(1, self._max_attempts + 1):
                self.attempts += 1
                try:
                    response = client.complete(request)
                except LLMError as exc:
                    last_error = exc
                    if not exc.retryable:
                        log.warning(
                            "llm_not_retryable",
                            hop=hop_name,
                            status=exc.status,
                            error=type(exc).__name__,
                        )
                        break  # a 400 is a bug: move on, do not hammer
                    if attempt == self._max_attempts:
                        log.warning("llm_hop_exhausted", hop=hop_name, attempts=attempt)
                        break
                    delay = self._delay_for(attempt, exc)
                    log.warning(
                        "llm_retry",
                        hop=hop_name,
                        attempt=attempt,
                        status=exc.status,
                        retry_after=exc.retry_after,
                        delay_s=round(delay, 2),
                    )
                    self._sleep(delay)
                except Exception as exc:  # unknown failure: treat as non-retryable
                    last_error = LLMError(str(exc), retryable=False, raw=exc)
                    log.warning("llm_unexpected_error", hop=hop_name, error=type(exc).__name__)
                    break
                else:
                    if hop_index > 0:
                        self.fallbacks += 1
                        log.warning(
                            "fallback_served", hop=hop_name, model_id=response.model_id
                        )
                    return response
        raise AllHopsExhausted(
            f"all {len(self._chain)} model hops exhausted",
            status=getattr(last_error, "status", None),
            retryable=False,
            raw=last_error,
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Streaming fails over only *before* the first token.

        Once the user has seen half an answer, silently restarting on another model
        would splice two different answers together. After first token the failure
        surfaces, and the UI shows its designed "generation failed" state.
        """
        for hop_index, (hop_name, client) in enumerate(self._chain):
            emitted = False
            try:
                for chunk in client.stream(request):
                    emitted = emitted or bool(chunk.delta)
                    yield chunk
                if hop_index > 0:
                    self.fallbacks += 1
                    log.warning("fallback_served", hop=hop_name, streaming=True)
                return
            except LLMError as exc:
                if emitted:
                    raise
                log.warning("llm_stream_hop_failed", hop=hop_name, status=exc.status)
                continue
        raise AllHopsExhausted("all model hops exhausted (streaming)")


def degraded_response(language: str = "en") -> LLMResponse:
    """The last hop of every chain: honest, cheap, and never an error page."""
    return LLMResponse(
        text=DEGRADED_ANSWER.get(language, DEGRADED_ANSWER["en"]),
        finish_reason="stop",
        model_id="degraded-static",
        usage=Usage(),
        latency_ms=0.0,
        route="degraded",
    )
