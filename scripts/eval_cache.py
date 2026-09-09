"""Module 6: the semantic cache's safety suite.

    python scripts/eval_cache.py
    python scripts/eval_cache.py --threshold 0.75      # watch a wrong hit appear

Twelve near-miss pairs — renew/cancel, fees-for-X/fees-for-Y, my-record/my-son's-
record. For each pair the first question is cached, then the second is asked. A
*hit* on the second is a **wrong hit**: two different questions served one answer.

Two numbers, again: hit rate and wrong-hit rate. A 40% hit rate with one wrong hit
per thousand is not a saving for a government assistant, and the target here is
zero — the threshold rises until it is.

An honest note about this course's embedding. ``retail_support.caching.embeddings`` is a
hashed-character-n-gram vector, not a sentence embedding, so the danger zone sits
lower on the scale than it would with a real embedding model: renew/cancel lands
near 0.78 here and nearer 0.95 with a production embedding. The *shape* of the
finding transfers; the *number* does not. Which is the point — measure your
threshold on your embedding and your traffic, and never inherit one from a slide.
"""

from __future__ import annotations

import argparse

from _common import bootstrap, read_jsonl, rule, write_json

bootstrap()

from retail_support.caching.embeddings import similarity  # noqa: E402
from retail_support.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=settings.cache.semantic_threshold)
    parser.add_argument(
        "--threshold-ar",
        type=float,
        default=settings.cache.semantic_threshold_by_language.get(
            "ar", settings.cache.semantic_threshold
        ),
    )
    args = parser.parse_args()

    pairs = read_jsonl("near_miss_pairs.jsonl")
    rule("eval-cache | near-miss suite")
    print(f"  thresholds: en {args.threshold} | ar {args.threshold_ar}\n")

    wrong = 0
    rows = []
    for pair in pairs:
        threshold = args.threshold_ar if pair["language"] == "ar" else args.threshold
        score = similarity(pair["a"], pair["b"])
        wrong_hit = score >= threshold
        wrong += 1 if wrong_hit else 0
        rows.append({**pair, "score": round(score, 3), "wrong_hit": wrong_hit})
        print(
            f"  {'WRONG HIT' if wrong_hit else 'ok       '} {score:.3f} "
            f"[{pair['language']}] {pair['a'][:34]} || {pair['b'][:34]}"
        )

    print(f"\nnear-miss suite: wrong hits {wrong}/{len(pairs)}")
    if wrong:
        print(
            "  Raise the threshold for the affected language and re-run. Note the\n"
            "  asymmetry in BENCHMARKS.md: Arabic spelling variants of the *same*\n"
            "  question score higher than English paraphrases do, so Arabic needs a\n"
            "  higher threshold, not a lower one."
        )
    write_json(
        "eval_cache.json",
        {
            "threshold": args.threshold,
            "threshold_ar": args.threshold_ar,
            "pairs": len(pairs),
            "wrong_hits": wrong,
            "rows": rows,
        },
    )
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
