"""The Arabic token premium, measured rather than asserted.

    python scripts/token_report.py

Course handouts have been repeating "Arabic costs 1.5-2.5x its English twin" for
years. Measure it and the picture is sharper than that: the premium is a property
of the *tokenizer generation*, not of the language. On the GPT-4-era vocabulary it
is around 2.5x. On the GPT-4o-era vocabulary, on the same 100 sentence pairs, it
is close to 1.0.

Which is the actual lesson: **count with the route's own tokenizer.** Not with a
convenient one, and not with a number from a slide — including this one, next year.
"""

from __future__ import annotations

import csv

from _common import DATA, bootstrap, rule, write_json

bootstrap()

from retail_support.llm.tokens import MODEL_ENCODINGS, count_with  # noqa: E402

ENCODINGS = ["cl100k_base", "o200k_base"]


def main() -> int:
    with (DATA / "tokenizer_pairs.csv").open(encoding="utf-8") as fh:
        pairs = [(row["en"], row["ar"]) for row in csv.DictReader(fh)]

    rule(f"token-report | {len(pairs)} parallel sentence pairs")
    results = []
    for encoding in ENCODINGS:
        en = sum(count_with(a, encoding) for a, _ in pairs)
        ar = sum(count_with(b, encoding) for _, b in pairs)
        chars_en = sum(len(a) for a, _ in pairs)
        chars_ar = sum(len(b) for _, b in pairs)
        row = {
            "encoding": encoding,
            "en_tokens": en,
            "ar_tokens": ar,
            "ar_over_en": round(ar / en, 2),
            "chars_per_token_en": round(chars_en / en, 2),
            "chars_per_token_ar": round(chars_ar / ar, 2),
            "models": [m for m, e in MODEL_ENCODINGS.items() if e == encoding],
        }
        results.append(row)
        print(
            f"  {encoding:<14} en={en:<6} ar={ar:<6} ar/en={row['ar_over_en']:<5} "
            f"chars/token en={row['chars_per_token_en']:<5} ar={row['chars_per_token_ar']:<5} "
            f"routes: {', '.join(row['models']) or '-'}"
        )

    worst = max(results, key=lambda r: r["ar_over_en"])
    best = min(results, key=lambda r: r["ar_over_en"])
    print(
        f"\n  The same corpus costs {worst['ar_over_en']}x on {worst['encoding']} and "
        f"{best['ar_over_en']}x on {best['encoding']}."
    )
    print(
        "  Budget per route, not per language. A context-budget check that counts with\n"
        "  the wrong tokenizer under- or over-budgets by tens of percent — which is the\n"
        "  whole of the sim-tokenizer-mismatch failure, in one table."
    )
    write_json("token_report.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
