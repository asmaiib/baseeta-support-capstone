"""Module 4: the two numbers, side by side, always.

    python scripts/guard_eval.py

A guard that blocks 100% of attacks and 12% of legitimate Arabic questions is a
broken product with excellent security numbers. Block rate alone can be gamed by
blocking everything; false-positive rate alone by blocking nothing. Report both or
report neither — and when a pair proudly announces 100% block, ask for the other
number before congratulating them.
"""

from __future__ import annotations

import argparse
from collections import Counter

from _common import bootstrap, read_jsonl, rule, write_json

bootstrap()

from retail_support.app import build_client  # noqa: E402
from retail_support.config import get_settings  # noqa: E402
from retail_support.domain.session import Session  # noqa: E402
from retail_support.guards.input_guards import InputGuard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default=None)
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="deterministic layer only — the 'before' row of the benchmark table",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = build_client(settings, args.route or settings.cheap_route)
    guard = InputGuard(
        client,
        max_chars=settings.guards.max_input_chars,
        classifier_enabled=not args.no_classifier,
        classifier_alias=settings.guards.classifier_alias,
    )

    attacks = read_jsonl("attack_corpus_40.jsonl")
    legit = read_jsonl("legit_corpus_60.jsonl")

    blocked_by_layer: Counter[str] = Counter()
    missed: list[dict] = []
    latencies: dict[str, list[float]] = {"deterministic": [], "classifier": []}

    for row in attacks:
        guarded = guard.check(row["text"], Session())
        for layer, value in guard.last_layer_timings.items():
            latencies.setdefault(layer, []).append(value)
        if guarded.blocked:
            blocked_by_layer[guarded.verdict.layer] += 1
        else:
            missed.append({"id": row["id"], "family": row["family"], "language": row["language"]})

    false_positives: list[dict] = []
    for row in legit:
        guarded = guard.check(row["text"], Session())
        for layer, value in guard.last_layer_timings.items():
            latencies.setdefault(layer, []).append(value)
        if guarded.blocked:
            false_positives.append(
                {
                    "id": row["id"],
                    "category": guarded.verdict.category,
                    "layer": guarded.verdict.layer,
                    "trap": row.get("trap", ""),
                    "text": row["text"],
                }
            )

    blocked = sum(blocked_by_layer.values())
    rule("guard-eval")
    layers = ", ".join(f"{layer} {count}" for layer, count in sorted(blocked_by_layer.items()))
    print(
        f"attack_corpus_40:  blocked {blocked}/{len(attacks)} "
        f"({blocked / len(attacks):.0%})  [{layers}]"
    )
    if missed:
        families = Counter(m["family"] for m in missed)
        print(f"                   missed: {len(missed)} ({dict(families)})")
        for row in missed:
            print(f"                     {row['id']} {row['language']}/{row['family']}")
    passed = len(legit) - len(false_positives)
    print(
        f"legit_corpus_60:   passed {passed}/{len(legit)} "
        f"(FP rate {len(false_positives) / len(legit):.0%})"
    )
    for row in false_positives:
        print(f"                     {row['id']} [{row['category']}] {row['text'][:56]}")
    timing = " | ".join(
        f"{layer} {sum(values) / len(values):.1f}ms"
        for layer, values in latencies.items()
        if values
    )
    print(f"guard latency:     {timing}")

    payload = {
        "attacks": len(attacks),
        "blocked": blocked,
        "block_rate": round(blocked / len(attacks), 4),
        "by_layer": dict(blocked_by_layer),
        "missed": missed,
        "legit": len(legit),
        "false_positives": false_positives,
        "fp_rate": round(len(false_positives) / len(legit), 4),
        "latency_ms": {
            layer: round(sum(values) / len(values), 2)
            for layer, values in latencies.items()
            if values
        },
        "classifier_enabled": not args.no_classifier,
    }
    write_json("guard_eval.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
