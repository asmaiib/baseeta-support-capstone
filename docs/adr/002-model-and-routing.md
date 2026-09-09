# ADR 002 — Model choice is a routing table, not a winner

- **Status:** accepted
- **Date:** 2026-09-09
- **Revisit:** quarterly, or on any provider deprecation notice

## Context

"Which model?" was asked as a single question with a single answer. It is not
one. Baseeta has three kinds of traffic with different constraints:

| Traffic | Volume | Constraint that decides |
|---|---|---|
| FAQ, no personal data | ~70% | unit cost |
| Service transactions, tool calling | ~25% | tool-contract reliability |
| Anything a data classification pins on-premise | varies | residency |

## Decision

Route, do not choose:

```yaml
routing_table:
  faq:      retail_support-small
  service:  retail_support-default
  complex:  retail_support-flagship
  escalate: null                # humans are not a model call
```

Residency-classified traffic goes to the `vllm` route regardless of the table.
That is possible **only** because every model call goes through the
`LLMClient` boundary; without it, "route by data classification" would be a
rewrite rather than a config entry.

A cascade (cheap-first, escalate on `unsupported_amounts` — the same
deterministic check the harness gates on) is implemented in `FAQHandler`
(`cascade_enabled` / `cascade_escalate_alias`) but is not the extension this
submission demonstrates; it is available and tested at the unit level
(`tests/pipeline/test_stages.py`) but not exercised end-to-end here.

## Evidence

**What is measured, on our own golden set, against the zero-key `fake` route:**
routing FAQ traffic to the cheap alias took the full 125-case run from
**28.7 → 5.4 halalas** (81% reduction), gate-verified at 125/125 with safety
at 100% both before and after (`eval/out/eval_primary.json` vs
`eval/out/eval_routing_on.json`).

**What requires live keys and is not yet measured:** the actual quality/cost
comparison across a real commercial backend and a real open-weight endpoint.
Section 6 of this submission (`BENCHMARKS.md`) documents how to run that
comparison (`scripts/bench_providers.py --routes comparison,vllm` once those
routes point at real endpoints via `RETAIL_SUPPORT_COMPARISON_BASE_URL` /
`RETAIL_SUPPORT_VLLM_BASE_URL` and their API keys) rather than presenting
simulator output as if it were evidence about real models. This is recorded
plainly in the Evaluation Report's known-limitations section.

## Self-hosting

`scripts/breakeven.py`, at the illustrative defaults (SAR 12/GPU-hour, 950
tok/s, 1.35× ops overhead — **not** a measured vLLM throughput, since no GPU
was available for this submission): self-hosting crosses over against the
**flagship** tier at roughly 25% sustained utilisation and **never** beats
the cheap hosted tier. Re-run with `--tokens-per-sec <measured>` against a
real vLLM deployment before treating the crossover point as more than
illustrative.

Recommendation, conditional on that re-run confirming the shape: keep any GPU
capacity for residency-bound and flagship-tier traffic; do not migrate FAQ
traffic onto it to raise utilisation — that pays more to look busier.

## Snapshot policy

Model ids are pinned in `configs/retail_support.yaml` and never referenced as
literals in code. A provider deprecation notice is a change ticket: run the
harness against the replacement, compare slices, update the config, update
this ADR.
