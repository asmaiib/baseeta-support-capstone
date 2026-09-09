"""The negative tests. This file is the security model, executable.

Four attacks, four rules:

1. a conversation cannot file a return for somebody else — the *session*
   decides, not the model's arguments;
2. a hallucinated tool name is an error the model can recover from, not a crash;
3. malformed arguments never reach a function;
4. a stubborn model meets a bound, and the bound degrades by design.

Plus the one that costs real money when it is missing: a retried turn does not
file the same return twice.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from retail_support.llm.fake import FakeClient
from retail_support.llm.interfaces import Message
from retail_support.pipeline.tool_loop import run_with_tools
from retail_support.tools.services import return_service


def seed_messages() -> list[Message]:
    return [
        Message(role="system", content="You are Baseeta Support."),
        Message(role="user", content="I'd like to return an item."),
    ]


def next_working_day(days_ahead: int = 15) -> str:
    day = date.today() + timedelta(days=days_ahead)
    while day.weekday() in (4, 5):  # Friday, Saturday
        day += timedelta(days=1)
    return day.isoformat()


def return_args(**overrides) -> dict:
    args = {
        "order_reference": "ORD-1000001",
        "category": "electronics",
        "resolution": "refund",
        "dropoff_city": "Riyadh",
        "dropoff_date": next_working_day(),
    }
    args.update(overrides)
    return args


def test_return_for_someone_else_is_denied(fake_client: FakeClient, session):
    """The injected conversation asks to file a return on behalf of customer B.

    The model's arguments are user input by proxy — they may have arrived
    through forty patient turns of conversation. The session, and only the
    session, says whose return this is.
    """
    fake_client.script_tool_call(
        "create_return_request", return_args(on_behalf_of="customer-B")
    )
    fake_client.script_text("I could not do that.")

    result = run_with_tools(fake_client, seed_messages(), session)

    assert return_service.cases_for("customer-B") == []
    assert return_service.cases_for("customer-A") == []
    assert result.calls == [], "a denied call never reaches the function"


def test_hallucinated_tool_name_is_an_error_not_a_crash(fake_client: FakeClient, session):
    fake_client.script_tool_call("cancel_everything", {"reason": "why not"})
    fake_client.script_text("Sorry, I cannot do that.")

    result = run_with_tools(fake_client, seed_messages(), session)

    assert result.text == "Sorry, I cannot do that."
    assert result.calls == []


def test_malformed_arguments_never_reach_the_function(fake_client: FakeClient, session):
    fake_client.script_tool_call("check_order_status", "{not json at all")
    fake_client.script_text("Could you confirm the order reference?")

    result = run_with_tools(fake_client, seed_messages(), session)

    assert "reference" in result.text.lower()
    assert result.calls == []


def test_loop_bound_trips_to_designed_degradation(fake_client: FakeClient, session):
    """A failing tool retried thirty times is a denial-of-wallet attack you wrote."""
    fake_client.script_endless_tool_calls("check_order_status")

    result = run_with_tools(fake_client, seed_messages(), session, max_iterations=6)

    assert result.bound_hit is True
    assert "transferring you" in result.text
    assert fake_client.call_count == 6, "the bound is 6, not 30"


def test_a_retried_turn_does_not_file_a_return_twice(fake_client: FakeClient, session):
    """Side effects plus retries need idempotency, exactly as in any other
    distributed system. The second execution is blocked, and one case exists."""
    args = return_args()
    fake_client.script_tool_call("create_return_request", args)
    fake_client.script_text("Filed.")
    run_with_tools(fake_client, seed_messages(), session)

    replay = FakeClient()
    replay.script_tool_call("create_return_request", args)
    replay.script_text("Filed.")
    run_with_tools(replay, seed_messages(), session)

    assert len(return_service.cases_for("customer-A")) == 1


def test_a_past_date_is_rejected_by_the_argument_contract(fake_client: FakeClient, session):
    """Constrained decoding guarantees the shape. It cannot know that 2019 is over."""
    fake_client.script_tool_call("create_return_request", return_args(dropoff_date="2019-01-02"))
    fake_client.script_text("That date has passed — which day would suit you?")

    result = run_with_tools(fake_client, seed_messages(), session)

    assert return_service.cases_for("customer-A") == []
    assert "date" in result.text.lower()


def test_unverified_identity_blocks_a_side_effecting_tool(fake_client: FakeClient):
    from retail_support.domain.session import Session

    unverified = Session(customer_id="customer-A", identity_verified=False)
    fake_client.script_tool_call("create_return_request", return_args())
    fake_client.script_text("I need to verify your identity first.")

    run_with_tools(fake_client, seed_messages(), unverified)

    assert return_service.cases_for("customer-A") == []


def test_read_only_tools_need_no_gate(fake_client: FakeClient, session):
    fake_client.script_tool_call("check_order_status", {"order_reference": "ORD-1000001"})
    fake_client.script_text("It is shipped.")

    result = run_with_tools(fake_client, seed_messages(), session)

    assert [call["tool"] for call in result.calls] == ["check_order_status"]
    assert result.calls[0]["risk"] == "read_only"


@pytest.mark.parametrize("tool_name", ["check_order_status", "create_return_request"])
def test_every_tool_call_is_traced_with_its_risk_class(fake_client: FakeClient, session, tool_name):
    arguments = (
        {"order_reference": "ORD-1000001"}
        if tool_name == "check_order_status"
        else return_args()
    )
    fake_client.script_tool_call(tool_name, arguments)
    fake_client.script_text("done")

    run_with_tools(fake_client, seed_messages(), session)

    assert session.tool_trace, "governance: 100% of calls logged with risk class and iteration"
    assert session.tool_trace[0]["tool"] == tool_name
    assert "risk" in session.tool_trace[0]
    assert session.tool_trace[0]["iteration"] == 1
