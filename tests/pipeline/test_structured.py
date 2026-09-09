"""The validate → retry → repair loop.

Three behaviours, each of which is a design decision rather than an accident:
one repair and no more, feedback that carries the *located* errors, and a designed
escalation instead of an unbounded loop or a relaxed schema.
"""

from __future__ import annotations

import pytest

from retail_support.domain.order import ReturnCase
from retail_support.llm.fake import FakeClient
from retail_support.pipeline.extract import ExtractionFailed, extract_ticket
from retail_support.pipeline.structured import StructuredExtractionFailed, extract_structured

GOOD = {
    "issue_type": "return",
    "order_reference": "ORD-1000001",
    "category": "electronics",
    "summary_en": "Customer wants to return a laptop that arrived damaged.",
    "city": "Riyadh",
    "urgency": "routine",
    "language": "ar",
    "customer": {"full_name": "Unnamed customer", "phone": None},
    "needs_human": False,
}
BAD_CITY = {**GOOD, "city": "Al Khobar"}
BAD_REF = {**GOOD, "order_reference": "12345678"}


def test_a_valid_first_try_costs_one_call():
    client = FakeClient().script_json(GOOD)
    case, outcome = extract_ticket(client, "أبغى أرجع لابتوب وصل تالف")
    assert isinstance(case, ReturnCase)
    assert outcome.first_try is True
    assert client.call_count == 1


def test_one_repair_recovers_a_semantic_violation():
    client = FakeClient().script_json(BAD_CITY).script_json(GOOD)
    case, outcome = extract_ticket(client, "message")
    assert case.city == "Riyadh"
    assert outcome.first_try is False
    assert outcome.attempts == 2


def test_the_repair_message_carries_the_located_errors():
    """"Please try again" barely moves the pass rate. The located error does."""
    client = FakeClient().script_json(BAD_REF).script_json(GOOD)
    extract_ticket(client, "message")

    repair_turn = client.requests[1].messages[-1].content
    assert "failed validation" in repair_turn
    assert "order_reference" in repair_turn


def test_two_failures_escalate_by_design_and_carry_the_evidence():
    client = FakeClient().script_json(BAD_CITY).script_json(BAD_CITY)
    with pytest.raises(ExtractionFailed) as raised:
        extract_ticket(client, "message")

    assert client.call_count == 2, "one repair, never an unbounded loop"
    assert raised.value.raw is not None, "the human-review queue needs the raw output"
    assert raised.value.errors


def test_nothing_downstream_ever_sees_raw_model_output():
    client = FakeClient().script_text("Sure! Here is your case: {oops")
    with pytest.raises(StructuredExtractionFailed):
        extract_structured(
            client,
            ReturnCase,
            system="s",
            user="u",
            schema_name="return_case",
            max_attempts=1,
        )


def test_the_request_carries_the_schema_and_a_stable_prefix():
    client = FakeClient().script_json(GOOD)
    extract_ticket(client, "message")
    request = client.requests[0]
    assert request.response_format["json_schema"]["strict"] is True
    assert request.response_format["json_schema"]["name"] == "return_case"
    assert request.temperature == 0.0, "extraction is validated downstream: sample greedily"
    assert request.cache_prefix_messages == 1
