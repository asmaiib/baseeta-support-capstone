"""The output contract's semantics, and the state that holds authority.

Two halves of the same lesson. The schema guarantees the shape; the validators
guarantee the meaning; the session guarantees who it is for. Nothing in a token
stream guarantees any of the three.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from retail_support.domain.session import ConversationState, Session, contains_pii, mask_pii
from retail_support.domain.order import (
    Customer,
    ReturnCase,
    ReturnRequest,
    schema_violations,
    strict_schema,
)


def valid_case(**overrides) -> dict:
    payload = {
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
    payload.update(overrides)
    return payload


def test_a_well_formed_case_validates():
    case = ReturnCase(**valid_case())
    assert case.city == "Riyadh"
    assert case.order_reference == "ORD-1000001"


@pytest.mark.parametrize(
    "order_reference",
    ["ORD1000001", "ord-1000001", "ORD-12", "ORD-", "12345678"],
)
def test_the_order_reference_validator_rejects_what_the_schema_cannot(order_reference: str):
    """Constrained decoding produces a string. Only a validator knows what a
    valid order reference looks like."""
    with pytest.raises(ValidationError):
        ReturnCase(**valid_case(order_reference=order_reference))


def test_a_valid_order_reference_passes():
    assert ReturnCase(**valid_case(order_reference="ORD-1000009")).order_reference == "ORD-1000009"


@pytest.mark.parametrize(
    "phone",
    ["12345", "+9664123456", "0512345", "notaphone"],
)
def test_the_phone_validator_rejects_non_saudi_numbers(phone: str):
    with pytest.raises(ValidationError):
        Customer(full_name="x", phone=phone)


@pytest.mark.parametrize("city", ["Al Khobar", "riyadh", "RIYADH", ""])
def test_an_out_of_enum_city_is_rejected(city: str):
    with pytest.raises(ValidationError):
        ReturnCase(**valid_case(city=city))


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        ReturnCase(**valid_case(confidence=0.9))


def test_the_wire_schema_fits_the_strict_subset():
    assert schema_violations(strict_schema()) == []
    schema = strict_schema()["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def working_day(days: int = 15) -> date:
    day = date.today() + timedelta(days=days)
    while day.weekday() in (4, 5):
        day += timedelta(days=1)
    return day


def test_the_return_request_contract_enforces_business_rules():
    assert ReturnRequest(
        order_reference="ORD-1000001",
        category="electronics",
        resolution="refund",
        dropoff_city="Riyadh",
        dropoff_date=working_day(),
    ).dropoff_city == "Riyadh"

    with pytest.raises(ValidationError, match="future"):
        ReturnRequest(
            order_reference="ORD-1000001", category="electronics", resolution="refund",
            dropoff_city="Riyadh", dropoff_date=date(2019, 1, 2),
        )

    friday = date.today() + timedelta(days=(4 - date.today().weekday()) % 7 + 7)
    with pytest.raises(ValidationError, match="closed"):
        ReturnRequest(
            order_reference="ORD-1000001", category="electronics", resolution="refund",
            dropoff_city="Riyadh", dropoff_date=friday,
        )

    with pytest.raises(ValidationError, match="30 days"):
        ReturnRequest(
            order_reference="ORD-1000001", category="electronics", resolution="refund",
            dropoff_city="Riyadh", dropoff_date=working_day(60),
        )

    with pytest.raises(ValidationError):
        ReturnRequest(
            order_reference="BAD-REF", category="electronics", resolution="refund",
            dropoff_city="Riyadh", dropoff_date=working_day(),
        )


# --- conversation state ---------------------------------------------------


def test_the_window_forgets_the_oldest_turn():
    """The moment statelessness lands: turn one is gone after turn nine."""
    state = ConversationState(max_turns=8)
    for i in range(9):
        state.add_user(f"question {i}")
        state.add_assistant(f"answer {i}")

    contents = [m.content for m in state.turns]
    assert "question 0" not in contents
    assert "question 8" in contents
    assert len(state.turns) == 16


def test_history_replays_oldest_to_newest_after_the_system_prompt():
    state = ConversationState()
    state.add_user("first")
    state.add_assistant("reply")
    messages = state.messages(system="SYSTEM")
    assert [m.role for m in messages] == ["system", "user", "assistant"]
    assert messages[1].content == "first"


# --- authority ------------------------------------------------------------


def test_the_session_not_the_argument_decides_whose_return_it_is():
    session = Session(customer_id="customer-A")
    verdict = session.authorize(
        "create_return_request", {"dropoff_city": "Riyadh", "on_behalf_of": "customer-B"}
    )
    assert not verdict.allowed
    assert verdict.reason == "cross_customer"
    assert verdict.user_hint


def test_stale_verification_blocks_a_side_effect():
    session = Session(customer_id="customer-A", identity_verified=False)
    assert not session.authorize("create_return_request", {"dropoff_city": "Riyadh"}).allowed
    session.verify_identity()
    assert session.authorize("create_return_request", {"dropoff_city": "Riyadh"}).allowed


def test_idempotency_blocks_the_second_identical_return_request():
    session = Session()
    args = {
        "order_reference": "ORD-1000001", "category": "electronics",
        "resolution": "refund", "dropoff_city": "Riyadh", "dropoff_date": "2026-10-14",
    }
    assert session.authorize("create_return_request", args).allowed
    session.record_side_effect("create_return_request", args, {"case_id": "RC1"})
    assert not session.authorize("create_return_request", args).allowed
    assert session.replay_side_effect("create_return_request", args) == {"case_id": "RC1"}


def test_the_pii_vault_round_trips_inside_the_boundary():
    session = Session()
    masked = mask_pii("id 1098765432 phone +966512345678 iban SA0380000000608010167519", session)
    assert contains_pii(masked) is None
    assert "1098765432" in session.pii_vault.unmask(masked)
