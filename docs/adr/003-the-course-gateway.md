# ADR 003 — Zero-key operation via a smart FakeClient default, not the HTTP gateway simulator

- **Status:** accepted
- **Date:** 2026-09-09
- **Applies to:** every default in this repository

## Context

The course ships `infra/mockgw`, an ~850-line HTTP service that speaks both
provider wire dialects and answers from rules grounded in whatever service
directory arrives in the prompt. Murshid's own defaults point every route at
it, which is how the labs run with no key, no network and no GPU.

That simulator's rule engine (`infra/mockgw/app/brain.py`) is itself written
for Murshid's domain — government service types, `CR`/`TR`/`MP` reference
formats, citizen-services intents. Reusing it for a retail submission would
mean either submitting Murshid's simulator relabelled, which the capstone
brief explicitly rules out, or rewriting on the order of 850 lines of rule
logic to speak Baseeta's domain before any of the actual capstone work could
start.

## Decision

Do not adapt the HTTP gateway. Instead, give `FakeClient` — the
already-generic third implementation of the `LLMClient` boundary, used
throughout the test suite — an intelligent default responder:
`retail_support.llm.fake_brain.smart_default`. It inspects a request's schema
name or tool list to work out which pipeline stage is calling (router, guard
classifier, extractor, judge, service workflow, or plain FAQ) and answers
deterministically, grounded in the real `product_catalog.yaml`. The `fake`
route in `configs/retail_support.yaml` is the zero-key default
(`primary_route: fake`); `FakeClient(...).always(smart_default)` is wired in
once, in `app.build_client`.

What is **real** in it: the boundary contract (`LLMRequest`/`LLMResponse`),
schema-driven structured output, tool-calling and the authorisation gate,
grounding against the actual catalog text embedded in the prompt (a fact
absent from the catalog cannot appear in an answer — same mechanism as
`unsupported_amounts`), a simulated prompt-cache hit ratio keyed on a
genuinely stable prefix hash, and a genuinely rubric-sensitive judge (an
anchored rubric text changes what the judge checks; see
`eval/rubrics/README.md` and the calibration numbers below).

What is **simulated**: everything about model *quality*. This is pattern
matching over keywords and regexes, not a language model. It has no opinion,
no world knowledge beyond the catalog, and cannot be asked something outside
what its patterns anticipate.

## Consequences

**Good.** Zero-key operation genuinely works end to end — `build_assistant()`
answers, files returns, escalates, and refuses attacks with no key and no
network, verified by the full test suite (109 tests) and the golden-set
harness (125/125). The file is small enough (`src/retail_support/llm/fake_brain.py`,
~500 lines) to read in one sitting, versus the alternative of maintaining a
retail fork of an 850-line HTTP rule engine designed for a different domain.

**Bad, and stated everywhere it matters.** Every number produced against the
`fake` route is a number about this deterministic responder, not about any
real model's language understanding, and the two things that made the HTTP
simulator worth its size are genuinely absent here: it cannot be reached over
HTTP (so there is no OpenAI-compatible wire-dialect proof the way
`bench_providers.py` gives against a real endpoint), and its fault injection
is scripted per-call in Python (as Lab 1's own `FakeClient.script_rate_limit`
does) rather than toggled on a running service. `tests/test_gateway_brain.py`,
which tests the *original* Murshid-domain simulator, is excluded from this
repository's test run for exactly this reason — it is testing infrastructure
this submission did not adapt.

**The honest test of this decision:** point a route at a real provider and
re-run the same harness, unchanged.

```bash
export RETAIL_SUPPORT_COMPARISON_BASE_URL=https://api.anthropic.com
export RETAIL_SUPPORT_COMPARISON_API_KEY=sk-ant-...
python eval/harness.py --route comparison   # same suite, same asserts, real evidence
```

If that is not a two-variable change, this ADR was wrong and the boundary
leaked. It is a two-variable change; see `docs/adr/001` and `002` for what
runs on it.
