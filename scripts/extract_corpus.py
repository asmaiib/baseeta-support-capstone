"""Module 3: schema-pass rate over the 50-case corpus, split by language.

    python scripts/extract_corpus.py                 # the default route
    python scripts/extract_corpus.py --route vllm    # the open-weight model
    python scripts/extract_corpus.py --audit         # the invented-field audit

Six numbers, and the split is the interesting part: Arabic first-try rates lag
English on most models, and the repair loop closes most of that gap. That single
observation justifies the loop better than any slide.
"""

from __future__ import annotations

import argparse
import json

from _common import bootstrap, read_jsonl, rule, write_json

bootstrap()

from retail_support.app import build_client, build_resilient  # noqa: E402
from retail_support.config import get_settings  # noqa: E402
from retail_support.pipeline.extract import CorpusReport, ExtractionFailed, extract_ticket  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default=None, help="pin extraction to one route")
    parser.add_argument("--audit", action="store_true", help="run the invented-field audit")
    parser.add_argument("--corpus", default="customer_messages_50.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    client = build_client(settings, args.route) if args.route else build_resilient(settings)
    rows = read_jsonl(args.corpus)
    report = CorpusReport()
    failures: list[dict] = []
    invented: list[dict] = []

    audit_index = {row["id"]: row for row in read_jsonl("extract_audit_15.jsonl")}

    for row in rows:
        language = row.get("gold", {}).get("language", "en")
        try:
            ticket, outcome = extract_ticket(client, row["text"])
        except ExtractionFailed as exc:
            report.record(language, "escalated")
            failures.append({"id": row["id"], "errors": exc.errors, "raw": (exc.raw or "")[:300]})
            continue
        report.record(language, "first_try" if outcome.first_try else "after_repair")

        if args.audit and row["id"] in audit_index:
            absent = audit_index[row["id"]]["absent_fields"]
            found = []
            if "applicant.national_id" in absent and ticket.applicant.national_id:
                found.append("applicant.national_id")
            if "city" in absent and ticket.city != "unknown":
                found.append("city")
            if found:
                report.invented_fields += len(found)
                invented.append({"id": row["id"], "invented": found})

    rule(f"extract-corpus | route={args.route or settings.primary_route + '+fallback'}")
    print(report.render())
    if args.audit:
        print(
            f"   invented-field audit: {report.invented_fields} invented across "
            f"{len(audit_index)} annotated cases"
        )
        for row in invented:
            print(f"     {row['id']}: {row['invented']}")
    if failures:
        print(f"   escalated to human review: {len(failures)}")
        for failure in failures[:3]:
            print(f"     {failure['id']}: {[e['loc'] for e in failure['errors']]}")

    payload = json.loads(report.model_dump_json())
    payload["route"] = args.route or settings.primary_route
    payload["failures"] = failures
    payload["invented"] = invented
    path = write_json(f"extract_corpus_{args.route or 'default'}.json", payload)
    print(f"\n   written: {path.relative_to(path.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
