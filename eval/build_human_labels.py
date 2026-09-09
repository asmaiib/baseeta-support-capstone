"""Build the 40-case calibration set that the judge is measured against.

    python eval/build_human_labels.py

In a real labelling round these labels come from the team's own pass: ten answers
each, the pool argued out loud, and the disagreements between *humans* settled
before anyone looks at the judge. The file this script writes is the reference set
to compare pooled labels against, and it is what keeps this repository's own
calibration reproducible.

Construction matters. The labels are deliberately **imbalanced but not degenerate**
(20 scores of 1.0, 6 of 0.5, 12 of 0.0, 2 of 1.0 for correct refusals), because a
calibration set where almost everything scores 1.0 produces a flattering percent
agreement and a meaningless kappa. Hard negatives are the whole point: an
instrument is qualified by what it gets wrong, not by what it gets right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retail_support.domain.catalog import load_catalog  # noqa: E402

OUT = ROOT / "eval" / "golden" / "human_labels_40.jsonl"

#: (entry_id, field) — entry_id is a category id or a general_policy key;
#: field is None for a general_policy fact. Ten distinct grounded facts,
#: reused across the four label classes exactly as the module intends.
FACTS: list[tuple[str, str | None]] = [
    ("cat-electronics", "return_window"),
    ("cat-electronics", "warranty"),
    ("cat-electronics", "shipping_fee"),
    ("cat-home", "return_window"),
    ("cat-home", "shipping_fee"),
    ("cat-home", "warranty"),
    ("cat-fashion", "return_window"),
    ("cat-fashion", "shipping_fee"),
    ("refund_method", None),
    ("exchange", None),
]

#: Subset with a monetary amount in the true value, so an "invented number"
#: substitution actually reads as a plausible-but-wrong fee.
MONETARY_FACTS: list[tuple[str, str | None]] = [
    ("cat-electronics", "shipping_fee"),
    ("cat-electronics", "restocking_fee"),
    ("cat-home", "shipping_fee"),
    ("cat-fashion", "shipping_fee"),
    ("cat-electronics", "shipping_fee"),
    ("cat-home", "shipping_fee"),
]

INVENTED = {
    "en": ["SAR 350", "SAR 90", "SAR 45", "SAR 640", "SAR 75", "SAR 720"],
    "ar": ["٣٥٠ ريالاً", "٩٠ ريالاً", "٤٥ ريالاً", "٦٤٠ ريالاً", "٧٥ ريالاً", "٧٢٠ ريالاً"],
}

DONT_KNOW = {
    "en": (
        "I don't have that information in the product catalog, so I won't guess. "
        "Please check with your nearest store."
    ),
    "ar": (
        "لا تتوفر لدي هذه المعلومة في كتالوج المنتجات، ولن أخمّن. "
        "يرجى مراجعة أقرب فرع."
    ),
}

IMPRECISE = {
    "en": "The {label} is roughly a couple of hundred riyals and usually takes a few days either way.",
    "ar": "{label} تقريباً حوالي مئتي ريال وتستغرق بضعة أيام عادةً في الاتجاهين.",
}

FIELD_LABEL = {
    "return_window": {"en": "return window", "ar": "مدة الإرجاع"},
    "warranty": {"en": "warranty", "ar": "الضمان"},
    "shipping_fee": {"en": "shipping fee", "ar": "رسوم الشحن"},
    "restocking_fee": {"en": "restocking fee", "ar": "رسوم إعادة التخزين"},
}


def _title(entry_id: str, field: str | None, language: str) -> str:
    if field is None:
        return {"refund_method": "refunds", "exchange": "exchanges"}[entry_id] if language == "en" else (
            "الاسترداد" if entry_id == "refund_method" else "الاستبدال"
        )
    return FIELD_LABEL[field][language]


def _true_value(catalog, entry_id: str, field: str | None, language: str) -> str:
    if field is None:
        return catalog.general_policy[entry_id].get(language)
    entry = catalog.by_id(entry_id)
    assert entry is not None, entry_id
    return getattr(entry, field).get(language)


def grounded_answer(catalog, entry_id: str, field: str | None, language: str) -> str:
    title = _title(entry_id, field, language)
    value = _true_value(catalog, entry_id, field, language)
    if language == "ar":
        return f"بخصوص {title}:\n- {value}"
    return f"About the {title}:\n- {value}"


def ungrounded_answer(entry_id: str, field: str | None, language: str, fee: str) -> str:
    title = _title(entry_id, field, language)
    if language == "ar":
        return f"بخصوص {title}:\n- {fee}"
    return f"About the {title}:\n- {fee}"


def main() -> int:
    catalog = load_catalog()
    rows: list[dict] = []

    def add(answer: str, language: str, score: float, why: str) -> None:
        rows.append(
            {
                "case_id": f"h{len(rows) + 1:03d}",
                "language": language,
                "answer": answer,
                "human_score": score,
                "label_reason": why,
            }
        )

    # 20 grounded answers, both languages — every stated fact is in the catalog.
    for entry_id, field in FACTS:
        for language in ("en", "ar"):
            add(
                grounded_answer(catalog, entry_id, field, language),
                language,
                1.0,
                "every fact stated appears in the catalog",
            )

    # 12 ungrounded answers — a fee that appears nowhere in the catalog.
    for index, (entry_id, field) in enumerate(MONETARY_FACTS):
        for language in ("en", "ar"):
            add(
                ungrounded_answer(entry_id, field, language, INVENTED[language][index]),
                language,
                0.0,
                "states a fee that is not in the catalog",
            )

    # 2 correct refusals — declining to guess is a 1.0, and the judge that scores
    # it 0.0 for "not answering" is the judge this calibration exists to catch.
    for language in ("en", "ar"):
        add(DONT_KNOW[language], language, 1.0, "correctly declines to guess")

    # 6 imprecise answers — catalog-shaped, rounded beyond what it actually says.
    for entry_id, field in FACTS[:3]:
        for language in ("en", "ar"):
            label = _title(entry_id, field, language)
            add(
                IMPRECISE[language].format(label=label),
                language,
                0.5,
                "catalog-adjacent but rounded and over-generalised",
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[float, int] = {}
    for row in rows:
        counts[row["human_score"]] = counts.get(row["human_score"], 0) + 1
    print(f"wrote {len(rows)} labelled answers to {OUT.relative_to(ROOT)}")
    print("  label distribution: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
