"""Judge calibration. A judge is an instrument: no kappa, no authority.

    python eval/calibrate_judge.py --rubric groundedness.v1.md
    python eval/calibrate_judge.py --rubric groundedness.v2.md

Percent agreement is not enough on its own. When 90% of the labels are 1.0, an
instrument that always says 1.0 scores 90% agreement and knows nothing. Cohen's
kappa corrects for the agreement you would get by chance, which is why the course
bar is stated in kappa (>= 0.6) and not in percent.

Two things to notice while running this:

* the inter-*human* agreement measured in the labelling bee is the ceiling for any
  judge. Seeing kappa(human, human) near 0.8 permanently recalibrates expectations
  about kappa(judge, human).
* when the judge disagrees with the humans, **fix the rubric, not the humans**. The
  jump from v1 to v2 is a rubric edit and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover
        pass

LABELS = ROOT / "eval" / "golden" / "human_labels_40.jsonl"
OUT = ROOT / "eval" / "out"


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Pure Python, twelve lines, no scikit-learn. It is a contingency table."""
    categories = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum(
        (a.count(c) / n) * (b.count(c) / n) for c in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubric", default="groundedness.v2.md")
    parser.add_argument("--route", default=None)
    parser.add_argument("--bar", type=float, default=0.6)
    args = parser.parse_args()

    from harness import judge_case  # noqa: PLC0415

    from retail_support.app import build_client
    from retail_support.config import get_settings
    from retail_support.pipeline.types import Reply

    settings = get_settings()
    client = build_client(settings, args.route or settings.primary_route)
    rubric_text = (ROOT / "eval" / "rubrics" / args.rubric).read_text(encoding="utf-8")

    rows = [json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines() if line.strip()]
    human: dict[str, float] = {}
    judge: dict[str, float] = {}
    evidence: dict[str, str] = {}

    for row in rows:
        reply = Reply(text=row["answer"], language=row["language"])
        case = {"strata": {"language": row["language"]}}
        verdict = judge_case(
            client, reply, case, rubric_text, settings.guards.classifier_alias
        )
        human[row["case_id"]] = float(row["human_score"])
        judge[row["case_id"]] = float(verdict["score"])
        evidence[row["case_id"]] = verdict["evidence"]

    ids = sorted(human)
    h = [str(human[i]) for i in ids]
    j = [str(judge[i]) for i in ids]
    agreement = sum(1 for x, y in zip(h, j, strict=True) if x == y) / len(ids)
    kappa = cohen_kappa(h, j)

    print(f"\nrubric: {args.rubric}")
    print(f"  agreement: {agreement:.0%} | cohen_kappa: {kappa:.2f} over {len(ids)} cases")
    verdict_line = (
        "judge may gate (tracking)"
        if kappa >= args.bar
        else "rubric needs work — do NOT wire this judge to anything"
    )
    print(f"  VERDICT: {verdict_line}")

    disagreements = [
        {
            "case_id": i,
            "human": human[i],
            "judge": judge[i],
            "judge_evidence": evidence[i],
        }
        for i in ids
        if human[i] != judge[i]
    ]
    if disagreements:
        print(f"\n  {len(disagreements)} disagreements — read them, then fix the RUBRIC:")
        for row in disagreements[:6]:
            print(
                f"    {row['case_id']}: human {row['human']} vs judge {row['judge']} — "
                f"{row['judge_evidence'][:70]}"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"calibration_{Path(args.rubric).stem}.json").write_text(
        json.dumps(
            {
                "rubric": args.rubric,
                "cases": len(ids),
                "agreement": round(agreement, 4),
                "cohen_kappa": round(kappa, 4),
                "bar": args.bar,
                "passes_bar": kappa >= args.bar,
                "disagreements": disagreements,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if kappa >= args.bar else 1


if __name__ == "__main__":
    raise SystemExit(main())
