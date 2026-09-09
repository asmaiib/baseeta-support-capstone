"""Promote a harness run to the baseline. A governed act, not a convenience.

    python eval/promote_baseline.py primary        # or: make baseline LABEL=primary

Two things this does that a `cp` does not:

* it writes a **summary only**. The gate reads the pass rate and the slices; the
  per-case replies are run artefacts. A 100 kB blob in git makes every promotion an
  unreviewable diff, and an unreviewable diff is how a bad baseline gets in.
* it refuses a run whose safety stratum is not at 100%, because a baseline is the
  definition of "known good", and a baseline with a red safety slice quietly
  legitimises every later run that inherits it.

Say why you promoted it in the commit message. "Regenerated" is not why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "eval" / "baseline.json"

NOTE = (
    "Summary only. The gate reads pass_rate and slices; the per-case replies live in "
    "eval/out/ and are run artefacts, not evidence to be committed. Promote a new "
    "baseline with `make baseline LABEL=<run>` and say why in the commit message."
)


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    source = ROOT / "eval" / "out" / f"eval_{label}.json"
    if not source.exists():
        print(f"no run found at {source.relative_to(ROOT)} — run `make eval LABEL={label}` first")
        return 1

    report = json.loads(source.read_text(encoding="utf-8"))
    safety = report.get("slices", {}).get("risk", {}).get("safety")
    if safety is not None and safety < 1.0:
        print(
            f"refusing to promote: the safety stratum is {safety:.0%}, not 100%.\n"
            "A baseline is the definition of known-good. Fix the safety cases first."
        )
        return 1

    if BASELINE.exists():
        current = json.loads(BASELINE.read_text(encoding="utf-8"))
        delta = report["pass_rate"] - current.get("pass_rate", 0.0)
        if delta < 0:
            print(
                f"note: this run is {delta * 100:.1f}pt BELOW the current baseline "
                f"({report['pass_rate']:.0%} vs {current['pass_rate']:.0%}). That is "
                "allowed — adding hard cases legitimately lowers a baseline — but it "
                "is not something to do by accident. Make the commit message earn it."
            )

    summary = {k: v for k, v in report.items() if k != "results"}
    summary["note"] = NOTE
    BASELINE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"baseline is now eval_{label}.json — {report['passed']}/{report['cases']} "
        f"({report['pass_rate']:.0%}), safety {safety:.0%}\n"
        "Say why in the commit message."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
