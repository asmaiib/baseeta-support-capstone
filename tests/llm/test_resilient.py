"""Reliability policy, tested where it lives.

Every assertion here is a rule from Module 1 §5 that costs money or uptime when it
is wrong: retry only what is retryable, honour ``Retry-After``, cap the attempts,
fall over to the next hop, and never restart a stream that has already spoken.
"""

from __future__ import annotations

import pytest

from retail_support.llm.fake import FakeClient
from retail_support.llm.interfaces import LLMError, LLMRequest, LLMResponse, Message
from retail_support.llm.resilient import AllHopsExhausted, ResilientClient, degraded_response

REQUEST = LLMRequest(messages=[Message(role="user", content="hello")], max_tokens=64)


def ok(text: str = "answer", model: str = "m") -> LLMResponse:
    return LLMResponse(text=text, model_id=model, finish_reason="stop")


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def test_retries_a_429_then_succeeds():
    sleep = RecordingSleep()
    client = FakeClient()
    client.script_rate_limit(times=2, retry_after=2.0)
    client.script_text("answer")
    resilient = ResilientClient([("primary", client)], sleep=sleep)

    response = resilient.complete(REQUEST)

    assert response.text == "answer"
    assert client.call_count == 3
    assert sleep.delays == [2.0, 2.0], "Retry-After is honoured, not guessed around"


def test_does_not_retry_a_400():
    """A malformed request retried three times is the same bug, three times
    slower, at three times the cost."""
    sleep = RecordingSleep()
    client = FakeClient().script_error(
        LLMError("bad request", status=400, retryable=False), times=1
    )
    resilient = ResilientClient([("primary", client)], sleep=sleep)

    with pytest.raises(AllHopsExhausted):
        resilient.complete(REQUEST)

    assert client.call_count == 1
    assert sleep.delays == []


def test_caps_attempts_then_falls_over_to_the_next_hop():
    sleep = RecordingSleep()
    primary = FakeClient(model_id="primary-model").script_rate_limit(times=9, retry_after=None)
    fallback = FakeClient(model_id="onprem-model").script_text("degraded but honest")
    resilient = ResilientClient(
        [("primary", primary), ("on_prem", fallback)], max_attempts=3, sleep=sleep
    )

    response = resilient.complete(REQUEST)

    assert primary.call_count == 3, "attempts are capped: retries multiply cost and tail latency"
    assert response.text == "degraded but honest"
    assert resilient.fallbacks == 1


def test_backoff_is_exponential_and_jittered():
    sleep = RecordingSleep()
    client = FakeClient().script_error(
        LLMError("overloaded", status=529, retryable=True), times=5
    )
    resilient = ResilientClient([("primary", client)], max_attempts=3, base_delay=1.0, sleep=sleep)

    with pytest.raises(AllHopsExhausted):
        resilient.complete(REQUEST)

    assert len(sleep.delays) == 2
    assert 0.5 <= sleep.delays[0] <= 1.5
    assert 1.0 <= sleep.delays[1] <= 3.0
    assert sleep.delays[1] > sleep.delays[0] * 0.6


def test_all_hops_exhausted_is_raised_not_swallowed():
    a = FakeClient().script_error(LLMError("boom", status=500, retryable=True), times=9)
    b = FakeClient().script_error(LLMError("boom", status=500, retryable=True), times=9)
    resilient = ResilientClient([("primary", a), ("second", b)], sleep=lambda _s: None)

    with pytest.raises(AllHopsExhausted):
        resilient.complete(REQUEST)


def test_streaming_does_not_fail_over_after_the_first_token():
    """Splicing two models' answers together mid-stream is worse than an error."""

    class HalfStream:
        route_name = "primary"

        def complete(self, request):  # pragma: no cover - not used here
            raise NotImplementedError

        def stream(self, request):
            from retail_support.llm.interfaces import StreamChunk

            yield StreamChunk(delta="half an ")
            raise LLMError("died mid-stream", status=500, retryable=True)

    fallback = FakeClient().script_text("a completely different answer")
    resilient = ResilientClient([("primary", HalfStream()), ("second", fallback)])

    with pytest.raises(LLMError):
        list(resilient.stream(REQUEST))


def test_degraded_response_is_honest_and_bilingual():
    for language in ("en", "ar"):
        response = degraded_response(language)
        assert response.text
        assert response.model_id == "degraded-static"
        assert response.usage.input_tokens == 0
