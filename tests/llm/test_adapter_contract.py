"""Protocol compliance, made executable.

The same test class runs against **every** adapter. That is what "implements the
protocol" has to mean in a codebase where swapping providers is supposed to be a
config change: not a docstring claim, a test that fails when it stops being true.

Adapters that need a server (OpenAI-compatible, Anthropic) run against the course
gateway and skip when it is not up. ``FakeClient`` always runs.
"""

from __future__ import annotations

import pytest

from retail_support.llm.fake import FakeClient
from retail_support.llm.interfaces import LLMClient, LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SIMPLE = LLMRequest(
    messages=[Message(role="user", content="What is your return policy for electronics?")],
    max_tokens=200,
    model_alias="retail_support-small",
)


def build_gateway_clients():
    """Adapters pointed at the course gateway, if it is running."""
    from conftest import gateway_up

    if not gateway_up():
        return []
    from retail_support.app import build_client
    from retail_support.config import get_settings

    settings = get_settings()
    return [
        pytest.param(build_client(settings, "primary"), id="openai_compat"),
        pytest.param(build_client(settings, "comparison"), id="anthropic"),
        pytest.param(build_client(settings, "vllm"), id="vllm"),
    ]


def all_clients():
    fake = FakeClient().always(
        lambda request: LLMResponse(
            text="ok", model_id="fake-model", finish_reason="stop", route="fake"
        )
    )
    return [pytest.param(fake, id="fake")] + build_gateway_clients()


@pytest.mark.parametrize("client", all_clients())
class TestAdapterContract:
    def test_satisfies_the_protocol(self, client):
        assert isinstance(client, LLMClient)

    def test_complete_returns_a_normalised_response(self, client):
        response = client.complete(SIMPLE)
        assert isinstance(response, LLMResponse)
        assert response.finish_reason in {"stop", "length", "tool_calls", "refusal", "error"}
        assert response.model_id, "every response names the CONCRETE model that answered"
        assert response.latency_ms >= 0

    def test_usage_is_always_accounted(self, client):
        response = client.complete(SIMPLE)
        assert response.usage.input_tokens >= 0
        assert response.usage.output_tokens >= 0
        assert response.usage.cached_input_tokens >= 0

    def test_streaming_ends_with_usage(self, client):
        chunks = list(client.stream(SIMPLE))
        assert chunks, "a stream must yield at least the final frame"
        final = chunks[-1]
        assert final.final is True
        assert final.usage is not None, (
            "losing usage on the final frame is the classic streaming "
            "cost-accounting bug — the meter silently undercounts most traffic"
        )
