"""Module 3: a scripted conversation that should trigger exactly one tool call.

Exactly one. A tool description that says "use for any question about applications"
fires on everything; a description written as API documentation fires on nothing.
Descriptions route — this script is how you find out which of those you wrote.

    python scripts/tool_smoke.py
"""

from __future__ import annotations

from _common import bootstrap, rule, write_json

bootstrap()

from retail_support.app import build_assistant  # noqa: E402
from retail_support.domain.session import Session  # noqa: E402

CASES = [
    {
        "name": "order status lookup with a reference",
        "text": "What is the status of my order ORD-1000001?",
        "expect_tools": ["check_order_status"],
    },
    {
        "name": "return-policy question — must NOT call a tool",
        "text": "What is your return policy for electronics?",
        "expect_tools": [],
    },
    {
        "name": "status question without a reference — must ask, not guess",
        "text": "Can you check my order please?",
        "expect_tools": [],
    },
    {
        "name": "return with everything confirmed",
        "text": "Yes, please file a return for order ORD-1000001, refund, drop-off in Riyadh on 2026-10-14, I confirm.",
        "expect_tools": ["create_return_request"],
    },
    {
        "name": "asks for a human",
        "text": "I want to speak to a human agent about my complaint",
        "expect_tools": ["escalate_to_agent"],
    },
]


def main() -> int:
    assistant = build_assistant()
    failures = 0
    rows = []
    rule("tool-smoke")
    for case in CASES:
        session = Session()
        reply = assistant.ask(case["text"], session, remember=False)
        called = [call["tool"] for call in reply.tool_calls if "tool" in call]
        ok = called == case["expect_tools"]
        failures += 0 if ok else 1
        rows.append({**case, "called": called, "ok": ok, "reply": reply.text})
        print(
            f"  {'OK ' if ok else 'BAD'} {case['name']:<48} "
            f"called={called or '[]'} expected={case['expect_tools'] or '[]'}"
        )
        if not ok:
            print(f"      reply: {reply.text[:100]}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} as expected")
    write_json("tool_smoke.json", {"failures": failures, "rows": rows})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
