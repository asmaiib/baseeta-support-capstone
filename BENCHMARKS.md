# Benchmarks

Every number below was produced by a named command, on this repository, against
the zero-key `fake` route unless stated otherwise. Re-run any row with the command
next to it.

**Read this first.** The `fake` route is a deterministic responder
(`llm/fake_brain.py`), not a model — see
[ADR 003](docs/adr/003-the-course-gateway.md). These numbers are correct as
measurements of the harness, the guards, the meter and the gate. They are not
evidence about any real model's language quality. The Section 6 provider
comparison needs real keys and is marked accordingly below.

## Section 1 — Architecture and the model boundary

| Check | Result | Command |
|---|---|---|
| No provider SDK outside `llm/adapters` (Anthropic/OpenAI touched only in `anthropic_client.py` / `openai_compat.py`) | pass | `pytest tests/test_architecture.py` |
| Retry + fallback under a scripted 429 storm and a scripted outage | pass, 6/6 architecture tests | `pytest tests/test_architecture.py` |

## Section 2 — Structured outputs and function calling

| Metric | Result | Command |
|---|---|---|
| Extraction schema-pass rate, first try | **100%** (50/50) — ar 32/32, en 15/15, mixed 3/3 | `python scripts/extract_corpus.py` |
| Strict-mode schema contracts | **5/5** (`ReturnCase`, `Customer`, `ReturnRequest`, `guard_verdict`, `route_verdict`) | `python scripts/schema_check.py` |
| Tool-safety negative cases | **9/9** (`tests/pipeline/test_tool_safety.py`) | `pytest tests/pipeline/test_tool_safety.py` |

## Section 3 — Prompt pipeline and guardrails

| Metric | Result | Command |
|---|---|---|
| Attack corpus blocked (40 cases, bilingual) | **100%** (40/40) — 34 deterministic, 6 classifier | `python scripts/guard_eval.py` |
| False-positive rate (60-case legit corpus, with 10 deliberate traps) | **0%** (60/60 passed) | `python scripts/guard_eval.py` |
| System-prompt leak attempts (5 scripted, bilingual) | **5/5 blocked**, canary intact 5/5 | `python scripts/leak_attack.py` |
| Semantic-cache near-miss safety suite | **0/12 wrong hits** | `python scripts/eval_cache.py` |

## Section 4 — Evaluation harness

| Metric | Result | Command |
|---|---|---|
| Golden set | 125 cases — Arabic-majority (63/62), safety oversampled (42/125), every stratum ≥8 | `python eval/build_golden.py` |
| Harness pass rate (primary route) | **125/125 (100%)**, every slice 100% | `python eval/harness.py --label primary` |
| Judge calibration, v1 (vague rubric) | κ = **-0.08** — fails the 0.6 bar, correctly | `python eval/calibrate_judge.py --rubric groundedness.v1.md` |
| Judge calibration, v2 (anchored, evidence-required) | κ = **0.71** — clears the bar | `python eval/calibrate_judge.py --rubric groundedness.v2.md` |
| Regression gate, clean run | **PASS**, +0.0pt every slice | `python eval/gate.py eval/out/eval_primary.json --baseline eval/baseline.json` |
| Regression gate, seeded regression (`input_guard_classifier.v0`, missing the instructions-carve-out) | **BLOCKED** — `risk=false_positive` 100%→80%, `intent=faq` 100%→97% | `RETAIL_SUPPORT_GUARD_PROMPT=input_guard_classifier.v0 python eval/harness.py --label degraded` then `python eval/gate.py eval/out/eval_degraded.json --baseline eval/baseline.json` |

## Section 5 — Cost and latency

| Optimisation | Before | After | Eval verdict |
|---|---|---|---|
| Prompt-cache discipline (`answer_faq.v0`, timestamp in the stable prefix → `v1`, timestamp moved to the volatile tail) | 0% cached_input_tokens | 68% cached_input_tokens (5 repeated calls) | n/a — deterministic mechanism check, not a quality change |
| Full golden-set run, cache coverage | — | **82.9%** cached_input_share, 262 model calls | 125/125 (100%), same as baseline |
| Response cache (exact + semantic), 10 identical queries | 10 model calls, 0.16 halalas | 1 model call, 0.02 halalas (**86% reduction**) | cache-eligible path only; guard/router unaffected |
| Routing table (FAQ traffic → cheap alias), full 125-case suite | 28.7 halalas | **5.4 halalas (81% reduction)** | **PASS** — 125/125 (100%), safety 100%, gate-verified |
| 200-conversation replay, cache+semantic+routing vs none | 1.02 halalas/conversation | **0.27 halalas/conversation (73.5% reduction)**, 55% exact-cache hit rate, 0 wrong hits | `python scripts/replay.py --label before/after` |

All four rows clear the ≥60% reduction bar this section asks for, and every
row that changes behaviour (not just accounting) has its eval verdict next
to it, not in a separate table.

## Section 6 — Commercial vs open-weight comparison

**Not yet run against real backends.** `scripts/bench_providers.py` and
`scripts/breakeven.py` are implemented and smoke-tested against the `fake`
route (below); the actual comparison needs a real Anthropic key and a real
OpenAI-compatible open-weight endpoint. To run it for real:

```bash
export RETAIL_SUPPORT_COMPARISON_BASE_URL=https://api.anthropic.com
export RETAIL_SUPPORT_COMPARISON_API_KEY=sk-ant-...
export RETAIL_SUPPORT_VLLM_BASE_URL=<your vLLM or hosted open-weight endpoint>/v1
export RETAIL_SUPPORT_VLLM_API_KEY=...
python eval/harness.py --route comparison --label comparison
python eval/harness.py --route vllm --label openweight
python scripts/bench_providers.py --routes comparison,vllm
```

| Check | Result |
|---|---|
| `bench_providers.py` smoke test (fake route) | runs cleanly, 20 bilingual prompts, 0.39 halalas/call |
| `breakeven.py`, illustrative defaults (SAR 12/GPU-hr, 950 tok/s **not measured on real hardware**, 1.35× ops overhead) | crosses over the flagship tier at ~25% sustained utilisation; never beats the cheap hosted tier |

The break-even's `--tokens-per-sec` default is illustrative, not measured — no
GPU was available in this environment. Re-run with a real vLLM throughput
number before treating the 25% crossover as more than a shape.
