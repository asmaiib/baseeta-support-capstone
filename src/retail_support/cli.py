"""The command line: ``ask``, ``chat``, ``stream`` and ``doctor``.

``doctor`` is the one to run first. It checks the things that actually break a
session — the Python version, the config, the reachability of every configured route,
and whether this terminal can render Arabic — and prints a tick or a cross for
each. On Windows, run it *now* rather than on the morning of day one.
"""

from __future__ import annotations

import argparse
import sys
import time

from retail_support.app import build_assistant, build_clients
from retail_support.config import get_settings
from retail_support.domain.session import Session
from retail_support.llm.interfaces import LLMRequest, Message
from retail_support.observability import configure_logging


def _force_utf8() -> None:
    """Windows consoles default to cp1252, which cannot print Arabic at all."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - non-reconfigurable stream
            pass


def cmd_ask(args) -> int:
    assistant = build_assistant(route=args.route)
    session = Session(customer_id=args.customer, max_turns=get_settings().pipeline.max_history_turns)
    reply = assistant.ask(args.question, session)
    print(
        f"[{reply.intent} → {reply.model_id or 'n/a'}"
        f"{' via ' + reply.route if reply.route else ''}"
        f"{' · cache:' + reply.cache_tier if reply.cache_tier else ''}] "
        f"{reply.latency_ms:.0f}ms, {reply.input_tokens} in "
        f"({reply.cached_tokens} cached) / {reply.output_tokens} out, "
        f"{reply.cost_halalas:.3f} halalas"
    )
    print(reply.text)
    return 0


def cmd_chat(args) -> int:
    assistant = build_assistant(route=args.route)
    session = Session(customer_id=args.customer, max_turns=get_settings().pipeline.max_history_turns)
    print("RetailSupport — type 'exit' to leave, '/state' to see the window.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text in {"exit", "quit"}:
            return 0
        if text == "/state":
            print(f"  window: {len(session.state.turns)} messages, max {session.state.max_turns} turns")
            for message in session.state.turns:
                print(f"   {message.role}: {message.content[:70]}")
            continue
        if not text:
            continue
        reply = assistant.ask(text, session)
        marker = "⛔" if reply.blocked else "retail_support"
        print(f"{marker}> {reply.text}")
        print(
            f"   [{reply.intent} · {reply.model_id or '-'} · {reply.latency_ms:.0f}ms · "
            f"{reply.cost_halalas:.3f} hal"
            + (f" · tools: {[c['tool'] for c in reply.tool_calls]}" if reply.tool_calls else "")
            + "]"
        )


def cmd_stream(args) -> int:
    """Streaming is UX, not decoration: watch TTFT and total diverge."""
    settings = get_settings()
    clients = build_clients(settings)
    client = clients[args.route or settings.primary_route]
    from retail_support.domain.catalog import rendered_catalog
    from retail_support.prompts.registry import load_prompt

    prompt = load_prompt("answer_faq.v1")
    messages = [
        Message(role="system", content=prompt.render(product_catalog=rendered_catalog("en"))),
        Message(role="user", content=f"<customer_message>\n{args.question}\n</customer_message>"),
    ]
    t0 = time.perf_counter()
    ttft = None
    for chunk in client.stream(LLMRequest(messages=messages, max_tokens=600)):
        if chunk.final:
            print(
                f"\n\n[TTFT {chunk.ttft_ms:.0f}ms · total {chunk.total_ms:.0f}ms · "
                f"{chunk.usage.input_tokens} in ({chunk.usage.cached_input_tokens} cached) / "
                f"{chunk.usage.output_tokens} out]"
            )
            continue
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        print(chunk.delta, end="", flush=True)
    return 0


def cmd_doctor(_args) -> int:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  {'✓' if passed else '✗'} {label}" + (f" — {detail}" if detail else ""))

    print("retail_support doctor\n")
    check("python 3.12+", sys.version_info >= (3, 12), sys.version.split()[0])

    try:
        settings = get_settings()
        check("config loads", True, f"{len(settings.routes)} routes")
    except Exception as exc:  # noqa: BLE001
        check("config loads", False, str(exc))
        return 1

    try:
        "كيف أجدد رخصتي التجارية؟".encode(sys.stdout.encoding or "utf-8")
        check("terminal renders Arabic", True, "كيف أجدد رخصتي التجارية؟")
    except Exception:
        check(
            "terminal renders Arabic",
            False,
            "use Windows Terminal or the VS Code terminal, and set PYTHONUTF8=1",
        )

    from retail_support.prompts.registry import list_prompts

    prompts = list_prompts()
    check("prompt registry", bool(prompts), f"{sum(len(v) for v in prompts.values())} versions")

    for name in settings.routes:
        try:
            client = build_clients(settings)[name]
            t0 = time.perf_counter()
            response = client.complete(
                LLMRequest(
                    messages=[Message(role="user", content="ping")],
                    max_tokens=16,
                    model_alias="retail_support-small",
                )
            )
            check(
                f"route {name}",
                True,
                f"{response.model_id} in {(time.perf_counter() - t0) * 1000:.0f}ms",
            )
        except Exception as exc:  # noqa: BLE001
            check(f"route {name}", False, f"{type(exc).__name__}: {str(exc)[:90]}")

    print("\n" + ("all good — you are ready for Module 1" if ok else "fix the crosses above"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    configure_logging()
    parser = argparse.ArgumentParser(prog="retail_support", description="RetailSupport CLI")
    parser.add_argument("--route", help="pin every model call to one configured route")
    parser.add_argument("--customer", default="customer-A", help="the authenticated customer id")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="one question, one answer")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="a windowed conversation")
    chat.set_defaults(func=cmd_chat)

    stream = sub.add_parser("stream", help="stream one answer and report TTFT")
    stream.add_argument("question")
    stream.set_defaults(func=cmd_stream)

    doctor = sub.add_parser("doctor", help="check the environment before a session")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
