"""The evaluation harness: the golden set, through the real pipeline.

    python eval/harness.py                     # default route
    python eval/harness.py --route vllm        # the open-weight comparison
    python eval/harness.py --judge --rubric groundedness.v2

Two design rules, both load-bearing:

1. **It runs the real pipeline.** ``make eval`` goes through the same
   ``build_assistant`` the CLI and the API use — guards, router, tools and all. A
   harness with its own simplified copy of the request path drifts within weeks,
   and then it is measuring the copy.
2. **Slices, never only the average.** 94% overall with 71% on Arabic emergency
   cases is a failing system with a passing headline. Every report here is sliced
   by language, intent, difficulty and risk class, and the gate reads the slices.

Safety asserts are deterministic by construction. Judges appear only as *tracking*
metrics: they add signal, they drift, and they never own a gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from asserts.checks import CHECKS  # noqa: E402

from retail_support.domain.catalog import rendered_catalog  # noqa: E402
from retail_support.domain.session import Session  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover
        pass

GOLDEN = ROOT / "eval" / "golden" / "regression_set.yaml"
OUT = ROOT / "eval" / "out"
SLICE_KEYS = ("language", "intent", "difficulty", "risk")


def evaluate_assert(spec: dict, reply, case: dict) -> tuple[bool, str]:
    kind = spec["type"]
    text = reply.text or ""

    if kind == "contains":
        return (spec["value"] in text), f"missing {spec['value']!r}"
    if kind == "not-contains":
        return (spec["value"] not in text), f"contains {spec['value']!r}"
    if kind == "regex":
        return bool(re.search(spec["value"], text)), f"no match for {spec['value']!r}"
    if kind == "intent":
        return (reply.intent == spec["value"]), f"intent was {reply.intent!r}"
    if kind == "blocked":
        return reply.blocked, "was not blocked"
    if kind == "not-blocked":
        return (not reply.blocked), f"blocked as {reply.guard_category!r}"
    if kind == "escalated":
        return (reply.escalated or reply.intent == "escalate"), "did not escalate"
    if kind == "tool-called":
        called = [c.get("tool") for c in reply.tool_calls]
        return (spec["value"] in called), f"tools called: {called}"
    if kind == "no-tool-called":
        called = [c.get("tool") for c in reply.tool_calls]
        return (not called), f"tools called: {called}"
    if kind == "no-pii-out":
        from retail_support.domain.session import contains_pii

        found = contains_pii(text)
        return (found is None), f"unmasked {found} in the answer"
    if kind == "latency":
        return (reply.latency_ms <= spec["value"]), f"{reply.latency_ms:.0f}ms"
    if kind == "python":
        check = CHECKS[spec["value"]]
        return check(reply, case)
    if kind == "llm-rubric":
        return True, ""  # scored separately; judges track, they do not gate
    raise ValueError(f"unknown assert type {kind!r}")


def judge_case(client, reply, case: dict, rubric_text: str, model_alias: str) -> dict:
    from pydantic import BaseModel

    from retail_support.pipeline.structured import extract_structured
    from retail_support.prompts.registry import load_prompt

    class JudgeVerdict(BaseModel):
        score: float
        evidence: str = ""

    language = case.get("strata", {}).get("language", "en")
    directory = rendered_catalog("ar" if language == "ar" else "en")
    system = load_prompt("judge_groundedness.v1").render()
    user = (
        f"<rubric>\n{rubric_text}\n</rubric>\n\n"
        f"<context>\n{directory}\n</context>\n\n"
        f"<answer>\n{reply.text}\n</answer>"
    )
    verdict, _ = extract_structured(
        client,
        JudgeVerdict,
        system=system,
        user=user,
        schema_name="judge_verdict",
        model_alias=model_alias,
        temperature=0.0,
        max_tokens=200,
    )
    return {"score": verdict.score, "evidence": verdict.evidence}


def run(
    *,
    route: str | None,
    limit: int | None,
    use_judge: bool,
    rubric: str,
    label: str,
) -> dict:
    from retail_support.app import build_assistant, build_client
    from retail_support.config import get_settings

    settings = get_settings()
    assistant = build_assistant(route=route)
    judge_client = build_client(settings, settings.primary_route) if use_judge else None
    rubric_text = (ROOT / "eval" / "rubrics" / rubric).read_text(encoding="utf-8")

    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]

    results = []
    started = time.perf_counter()
    for case in cases:
        session = Session(customer_id=f"customer-{case['id']}")
        reply = assistant.ask(case["vars"]["customer_message"], session, remember=False)
        failures = []
        for spec in case["assert"]:
            passed, detail = evaluate_assert(spec, reply, case)
            if not passed:
                failures.append({"type": spec["type"], "value": spec.get("value"), "detail": detail})
        judge = None
        if use_judge and any(a["type"] == "llm-rubric" for a in case["assert"]):
            judge = judge_case(
                judge_client, reply, case, rubric_text, settings.guards.classifier_alias
            )
            judge["threshold"] = next(
                a.get("threshold", 0.67) for a in case["assert"] if a["type"] == "llm-rubric"
            )
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "strata": case["strata"],
                "passed": not failures,
                "failures": failures,
                "judge": judge,
                "reply": reply.text,
                "intent": reply.intent,
                "blocked": reply.blocked,
                "latency_ms": round(reply.latency_ms, 1),
                "cost_halalas": reply.cost_halalas,
                "model_id": reply.model_id,
                "cache_tier": reply.cache_tier,
            }
        )

    wall = time.perf_counter() - started
    return summarise(results, route=route, label=label, wall_s=round(wall, 1), rubric=rubric)


def summarise(results: list[dict], *, route, label, wall_s, rubric) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    slices: dict[str, dict[str, dict[str, int]]] = {key: {} for key in SLICE_KEYS}
    for r in results:
        for key in SLICE_KEYS:
            value = r["strata"].get(key, "?")
            bucket = slices[key].setdefault(value, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += 1 if r["passed"] else 0
    judged = [r for r in results if r.get("judge")]
    judge_mean = (
        round(sum(r["judge"]["score"] for r in judged) / len(judged), 3) if judged else None
    )
    return {
        "label": label,
        "route": route or "default",
        "rubric": rubric,
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "slices": {
            key: {
                value: round(bucket["passed"] / bucket["total"], 4)
                for value, bucket in sorted(values.items())
            }
            for key, values in slices.items()
        },
        "slice_counts": {
            key: {value: bucket["total"] for value, bucket in sorted(values.items())}
            for key, values in slices.items()
        },
        "judge_mean": judge_mean,
        "judged_cases": len(judged),
        "wall_s": wall_s,
        "total_halalas": round(sum(r["cost_halalas"] for r in results), 4),
        "p50_latency_ms": sorted(r["latency_ms"] for r in results)[total // 2] if total else 0,
        "results": results,
    }


def render(report: dict) -> None:
    print(f"\n{'─' * 72}")
    print(
        f"eval | route={report['route']} | {report['cases']} cases | "
        f"pass {report['passed']}/{report['cases']} ({report['pass_rate']:.0%}) | "
        f"{report['wall_s']}s | {report['total_halalas']:.1f} halalas"
    )
    print("─" * 72)
    for key in SLICE_KEYS:
        parts = [
            f"{value} {rate:.0%}" for value, rate in report["slices"][key].items()
        ]
        print(f"  {key:<11} " + " | ".join(parts))
    if report["judge_mean"] is not None:
        print(f"  judge      groundedness mean {report['judge_mean']} over {report['judged_cases']} cases (tracking)")
    failing = [r for r in report["results"] if not r["passed"]]
    if failing:
        print(f"\n  {len(failing)} failing:")
        for row in failing[:12]:
            reasons = ", ".join(f"{f['type']}({f['detail']})" for f in row["failures"])
            print(f"    {row['id']} [{row['strata']['risk']}] {row['description'][:44]} — {reasons[:70]}")
        if len(failing) > 12:
            print(f"    ... and {len(failing) - 12} more")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true", help="run the groundedness judge")
    parser.add_argument("--rubric", default="groundedness.v2.md")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = run(
        route=args.route,
        limit=args.limit,
        use_judge=args.judge,
        rubric=args.rubric,
        label=args.label,
    )
    render(report)
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else OUT / f"eval_{args.label}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
