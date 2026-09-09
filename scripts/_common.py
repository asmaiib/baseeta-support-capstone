"""Shared plumbing for the measurement scripts. Nothing clever lives here on purpose."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "eval" / "out"


def bootstrap() -> None:
    """Make ``python scripts/x.py`` work from a clone with no install step."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass


def read_jsonl(name: str) -> list[dict]:
    path = DATA / name if not Path(name).is_absolute() else Path(name)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarise_latency(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50_ms": round(statistics.median(values)) if values else 0,
        "p95_ms": round(percentile(values, 95)) if values else 0,
        "max_ms": round(max(values)) if values else 0,
    }


def rule(title: str = "") -> None:
    print(f"\n{'─' * 72}")
    if title:
        print(title)
        print("─" * 72)
