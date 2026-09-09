"""Module 6: replay 200 conversations through the real pipeline and meter everything.

    python scripts/replay.py --label before
    python scripts/replay.py --label after --cache --semantic --routing

Task 1's ritual, and it is not optional: run the replay, aggregate the cost log,
and **say out loud where the money goes before touching anything**. Pairs who skip
it optimise the 4% line item while the 60% one sits unexamined, and the leaderboard
shows it.

    jq -s 'group_by(.intent) | map({intent: .[0].intent, halalas: (map(.cost_halalas) | add)})' \\
       logs/llm_cost.jsonl
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict

from _common import ROOT, bootstrap, percentile, read_jsonl, rule, write_json

bootstrap()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--cache", action="store_true", help="exact response cache on")
    parser.add_argument("--semantic", action="store_true", help="semantic tier on (implies --cache)")
    parser.add_argument("--routing", action="store_true", help="apply the routing table")
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="cheap-first, escalate on an unsupported amount (Module 6 §5)",
    )
    parser.add_argument("--prompt", default=None, help="e.g. answer_faq.v0 (the cache-killer)")
    parser.add_argument("--route", default=None, help="pin every call to one route")
    args = parser.parse_args()

    # Environment first: the settings object and the FAQ prompt are read at import
    # time by design, so that a walkthrough can change behaviour without editing code.
    if args.prompt:
        os.environ["RETAIL_SUPPORT_FAQ_PROMPT"] = args.prompt
    if args.cache or args.semantic:
        os.environ["RETAIL_SUPPORT_CACHE_ENABLED"] = "1"
    if args.semantic:
        os.environ["RETAIL_SUPPORT_SEMANTIC_CACHE_ENABLED"] = "1"
    if args.routing:
        os.environ["RETAIL_SUPPORT_ROUTING_ENABLED"] = "1"
    if args.cascade:
        os.environ["RETAIL_SUPPORT_CASCADE_ENABLED"] = "1"
    cost_log = ROOT / "logs" / f"llm_cost_{args.label}.jsonl"
    if cost_log.exists():
        cost_log.unlink()

    from retail_support.app import build_assistant  # noqa: PLC0415 - after env is set
    from retail_support.config import load_settings
    from retail_support.domain.session import Session
    from retail_support.observability.cost import CostMeter

    settings = load_settings()
    meter = CostMeter(settings.prices, sink=cost_log)
    assistant = build_assistant(settings, route=args.route, meter=meter)

    conversations = read_jsonl("replay_200.jsonl")[: args.limit]
    turn_latencies: list[float] = []
    first_turn_latencies: list[float] = []
    conversation_totals: list[float] = []
    intents: Counter[str] = Counter()
    blocked = 0
    tool_calls = 0
    started = time.perf_counter()

    for conversation in conversations:
        session = Session(
            customer_id=f"customer-{conversation['id']}",
            max_turns=settings.pipeline.max_history_turns,
        )
        total = 0.0
        for index, text in enumerate(conversation["turns"]):
            reply = assistant.ask(text, session)
            turn_latencies.append(reply.latency_ms)
            if index == 0:
                first_turn_latencies.append(reply.latency_ms)
            total += reply.latency_ms
            intents[reply.intent] += 1
            blocked += 1 if reply.blocked else 0
            tool_calls += len(reply.tool_calls)
        conversation_totals.append(total)

    wall = time.perf_counter() - started
    total_halalas = meter.total_halalas
    per_conversation = total_halalas / max(len(conversations), 1)
    by_intent = meter.by("intent")
    cache = getattr(assistant, "cache", None)

    rule(f"replay | label={args.label}")
    flags = ", ".join(
        [
            f"cache={'on' if args.cache or args.semantic else 'off'}",
            f"semantic={'on' if args.semantic else 'off'}",
            f"routing={'on' if args.routing else 'off'}",
            f"cascade={'on' if args.cascade else 'off'}",
            f"faq_prompt={os.environ.get('RETAIL_SUPPORT_FAQ_PROMPT', 'answer_faq.v1')}",
        ]
    )
    print(f"  {flags}")
    print(
        f"{len(conversations)} conversations | cost/conv: {per_conversation:.2f} halalas | "
        f"p50 turn {percentile(turn_latencies, 50):.0f}ms | "
        f"p95 conversation {percentile(conversation_totals, 95):.0f}ms | wall {wall:.1f}s"
    )
    share = {k: f"{v / total_halalas:.0%}" for k, v in by_intent.items()} if total_halalas else {}
    print(f"by intent (spend): {share}")
    print(f"by intent (turns): {dict(intents)}   blocked: {blocked}   tool calls: {tool_calls}")
    print(
        f"prompt cache: {meter.cached_input_share():.0%} of input tokens at the cached rate"
    )
    if cache is not None:
        print(f"response cache: {cache.stats.render()}")
    escalations = getattr(assistant.deps.faq_handler, "escalations", 0)
    if args.cascade:
        print(
            f"cascade: {escalations} escalations "
            f"({escalations / max(intents.get('faq', 1), 1):.0%} of FAQ turns paid twice)"
        )

    payload = {
        "label": args.label,
        "conversations": len(conversations),
        "cost_halalas_total": round(total_halalas, 4),
        "cost_halalas_per_conversation": round(per_conversation, 4),
        "by_intent_spend": by_intent,
        "by_intent_turns": dict(intents),
        "p50_turn_ms": round(percentile(turn_latencies, 50)),
        "p95_turn_ms": round(percentile(turn_latencies, 95)),
        "p50_first_turn_ms": round(percentile(first_turn_latencies, 50)),
        "p95_conversation_ms": round(percentile(conversation_totals, 95)),
        "prompt_cache_share": round(meter.cached_input_share(), 4),
        "blocked": blocked,
        "tool_calls": tool_calls,
        "flags": {
            "cache": args.cache or args.semantic,
            "semantic": args.semantic,
            "routing": args.routing,
            "cascade": args.cascade,
            "faq_prompt": os.environ.get("RETAIL_SUPPORT_FAQ_PROMPT", "answer_faq.v1"),
            "route": args.route,
        },
    }
    if cache is not None:
        payload["response_cache"] = {
            "lookups": cache.stats.lookups,
            "exact_hits": cache.stats.exact_hits,
            "semantic_hits": cache.stats.semantic_hits,
            "hit_rate": round(cache.stats.hit_rate, 4),
        }
    path = write_json(f"replay_{args.label}.json", payload)
    print(f"\n  written: {path.name}   cost log: {cost_log.relative_to(ROOT)}")

    stage_spend: dict[str, float] = defaultdict(float)
    for record in meter.records:
        stage_spend[record.stage or "?"] += record.cost_halalas
    print("  spend by stage: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(stage_spend.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
