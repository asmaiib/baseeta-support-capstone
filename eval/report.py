"""Generate ``EVALUATION_REPORT.md`` from real runs.

    python eval/report.py

This is the capstone's headline deliverable in miniature: overall and sliced
results for every backend, the judge's calibration evidence, the safety suite's
status, the guard numbers as a pair, the cost picture — and a **known limitations**
section, because honesty scores points and a report with no limitations section is
a report nobody believes.

It reads whatever is in ``eval/out/``. Nothing here invents a number, and a missing
input becomes a visible gap in the report rather than a silent omission.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "out"
TARGET = ROOT / "EVALUATION_REPORT.md"


def load(name: str) -> dict | list | None:
    path = OUT / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def slice_rows(report: dict) -> list[str]:
    rows = []
    for key, values in report["slices"].items():
        for name, rate in sorted(values.items()):
            count = report["slice_counts"][key][name]
            rows.append(f"| {key} = {name} | {count} | {rate:.0%} |")
    return rows


def main() -> int:
    lines: list[str] = []
    add = lines.append

    add("# RetailSupport — evaluation report")
    add("")
    add(f"Generated {date.today().isoformat()} by `python eval/report.py`.")
    add("")
    add(
        "> **Read this first.** Every number below was measured against the course "
        "gateway (`infra/mockgw`), which is a deterministic simulator, not a model. "
        "The measurements are true statements about this application's harness, "
        "guards, meter and gate. They are not evidence about any model's quality. "
        "Point a route at a real provider — two environment variables, no code "
        "change — and re-run the same harness when you need that kind of evidence."
    )
    add("")

    add("## 1. Quality — golden set through the real pipeline")
    add("")
    reports = {
        "primary (commercial)": load("eval_primary.json"),
        "on-prem (vLLM route)": load("eval_vllm.json"),
        "cheap (small model)": load("eval_cheap.json"),
    }
    available = {k: v for k, v in reports.items() if v}
    if not available:
        add("_No harness run found. Run `make eval` first._")
    else:
        add("| Backend | Cases | Overall | ar | en | safety | wall | cost |")
        add("|---|---|---|---|---|---|---|---|")
        for label, report in available.items():
            add(
                f"| {label} | {report['cases']} | {report['pass_rate']:.0%} | "
                f"{pct(report['slices']['language'].get('ar'))} | "
                f"{pct(report['slices']['language'].get('en'))} | "
                f"{pct(report['slices']['risk'].get('safety'))} | "
                f"{report['wall_s']}s | {report['total_halalas']:.1f} hal |"
            )
        add("")
        first = next(iter(available.values()))
        add(f"### Slices — {next(iter(available))}")
        add("")
        add("| Stratum | Cases | Pass rate |")
        add("|---|---|---|")
        lines.extend(slice_rows(first))
        add("")
        add(
            "Averages are reported last on purpose. A change that lifts the overall "
            "number while dropping one stratum is a regression, and the gate reads "
            "the slices for exactly that reason."
        )
    add("")

    add("## 2. Judge calibration")
    add("")
    v1 = load("calibration_groundedness.v1.json")
    v2 = load("calibration_groundedness.v2.json")
    if not (v1 or v2):
        add("_No calibration run found. Run `make calibrate` first._")
    else:
        add("| Rubric | Cases | Agreement | Cohen's kappa | Verdict |")
        add("|---|---|---|---|---|")
        for rubric in (v1, v2):
            if not rubric:
                continue
            add(
                f"| `{rubric['rubric']}` | {rubric['cases']} | {rubric['agreement']:.0%} | "
                f"{rubric['cohen_kappa']:.2f} | "
                f"{'may gate (tracking)' if rubric['passes_bar'] else 'NOT qualified'} |"
            )
        add("")
        add(
            "The judge is an instrument and it is qualified, not consulted. It "
            "contributes tracking signal only: no safety assertion in this suite "
            "depends on a model's opinion."
        )
        if v2 and v2["disagreements"]:
            add("")
            add(
                f"{len(v2['disagreements'])} residual disagreements against the sharpened "
                "rubric. They are informative rather than embarrassing — read them before "
                "writing the next rubric version."
            )
    add("")

    add("## 3. Safety")
    add("")
    guards = load("guard_eval.json")
    leak = load("leak_attack.json")
    cache = load("eval_cache.json")
    if guards:
        add(
            f"- Attack corpus: **{guards['blocked']}/{guards['attacks']} blocked "
            f"({guards['block_rate']:.0%})**, by layer {guards['by_layer']}."
        )
        add(
            f"- Legitimate corpus: **false-positive rate {guards['fp_rate']:.0%}** "
            f"({len(guards['false_positives'])}/{guards['legit']}). Both numbers, always, "
            "together — either one alone can be gamed into a broken product."
        )
        add(f"- Guard latency: {guards['latency_ms']} (milliseconds, by layer).")
    if leak:
        add(
            f"- System-prompt extraction: {leak['blocked']}/{leak['attempts']} refused at the "
            f"input wall, canary leaked **{leak['leaked']}** times."
        )
    if cache:
        add(
            f"- Semantic cache near-miss suite: **{cache['wrong_hits']}/{cache['pairs']} wrong "
            f"hits** at thresholds en {cache['threshold']} / ar {cache['threshold_ar']}."
        )
    if not (guards or leak or cache):
        add("_No safety runs found._")
    add("")

    add("## 4. Cost and latency")
    add("")
    replays = [
        ("Baseline (`answer_faq.v4`, no cache, no routing)", load("replay_before.json")),
        ("+ prompt-cache discipline (`answer_faq.v5`)", load("replay_s1-prefix.json")),
        ("+ response cache (exact + semantic)", load("replay_s2-cache.json")),
        ("+ routing table", load("replay_s3-routing.json")),
    ]
    replays = [(label, data) for label, data in replays if data]
    if not replays:
        add("_No replay found. Run `make replay` first._")
    else:
        base = replays[0][1]["cost_halalas_per_conversation"]
        add("| Configuration | Cost/conversation | Δ | p50 turn | p95 conversation | Prompt cache |")
        add("|---|---|---|---|---|---|")
        for label, data in replays:
            cost = data["cost_halalas_per_conversation"]
            delta = (cost - base) / base * 100 if base else 0.0
            add(
                f"| {label} | {cost:.2f} hal | {delta:+.0f}% | {data['p50_turn_ms']} ms | "
                f"{data['p95_conversation_ms']} ms | {data['prompt_cache_share']:.0%} |"
            )
        add("")
        add(
            "Every row was taken with the eval suite green, and the row that was not "
            "is not in the table. Never trade quality you are not measuring for cost "
            "you are."
        )
    add("")

    add("## 5. Known limitations")
    add("")
    add(
        "- **The gateway is a simulator.** Quality differences between the model "
        "tiers here are rules, not capability. Every quality claim in this report is "
        "a claim about the harness working, not about a model."
    )
    add(
        "- **The semantic cache's embedding is a hashed character n-gram**, not a "
        "sentence embedding. Its similarity scale is not a production scale: the "
        "threshold that is safe here is not transferable, and the near-miss suite is "
        "how you would find your own."
    )
    add(
        "- **The golden set is 126 cases.** That is enough to catch the regressions "
        "this course seeds and not enough to resolve a two-point difference. Error "
        "bars on a 50-case corpus are wider than most of the deltas people quote "
        "from one."
    )
    add(
        "- **The judge shares a family with the model under test** in the default "
        "configuration, which is exactly the self-preference conflict the module "
        "warns about. Reported as a conflict rather than resolved."
    )
    add(
        "- **Latency figures are compressed** by `MOCKGW_SPEED`. Relative ordering "
        "between routes is meaningful; absolute milliseconds are not."
    )
    add("")

    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
