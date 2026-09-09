# ADR 001 — Router-first, with one bounded tool loop

- **Status:** accepted
- **Date:** 2026-09-09
- **Deciders:** application engineer of record (capstone submission, Track D — Retail order support)
- **Supersedes:** the `demo_v0.py` single-call prototype

## Context

Baseeta's traffic, from the analysis behind `data/replay_200.jsonl`, is roughly
70% FAQ-shaped (return policy, shipping, warranty), 25% transactional (order
status, filing a return or exchange) and 5% escalation. Three patterns were on
the table: a single agentic loop for everything, a fixed workflow for
everything, or a router dispatching to specialised handlers.

## Decision

**Router first.** A cheap classifier assigns one of three intents and dispatches:

- `faq` → a single call against the product catalog, cacheable, cheap model;
- `service` → a fixed workflow whose middle is a **bounded** tool loop
  (6 iterations, allowed-tool list, authorisation gate on the side-effecting tool);
- `escalate` → a human. Not a model call at all.

The tool loop is the only agentic component in this application, and it is
bounded on every axis that can run away: iterations, tokens, tool list, and a
trace of every call (`session.tool_trace`).

## Consequences

**Good.** Each path is testable in isolation (`tests/pipeline/test_stages.py`).
Cost follows traffic shape rather than worst case: routing FAQ traffic to the
cheap alias measured **28.7 → 5.4 halalas** over the 125-case golden set
(81% reduction) with the eval gate still green at 125/125. A misroute is
*measurable* — it is a stratum in the golden set — where an agent's wrong turn
is a transcript somebody has to read.

**Bad.** Two more moving parts than a single call, and a router that is wrong
sends a customer down the wrong path. We accept this because the misroute rate
is measured and gated (`intent` is one of the four slice keys the harness
reports on), and because the failure is legible: a wrong route produces a
wrong-shaped answer, not a quietly expensive loop.

**Rejected: agent-first.** The decomposition here is known at design time — a
return is check-status → gather-fields → file-request, not an open-ended
plan. An agentic loop would buy flexibility we do not need and pay for it in
unbounded cost and undebuggable traces.

**Rejected: workflow-only.** A fixed chain for FAQ traffic would run the tool
schemas and the workflow prompt for every "what's your return policy?",
paying the transactional price for conversational traffic — measured at
roughly 20x the cost of the cheap-aliased FAQ path.

## Evidence

`BENCHMARKS.md`, cost table: 28.7 → 5.4 halalas across the routing change on
the full 125-case golden set, evaluation suite green (125/125, safety 100%)
at every step, confirmed by `eval/gate.py` against the promoted baseline.
