# Evaluation Report — Baseeta Support (Track D, Retail order support)

Generated from real runs (`eval/out/eval_primary.json`, `eval/out/eval_degraded.json`,
`eval/out/calibration_groundedness.v1.json`, `eval/out/calibration_groundedness.v2.json`).
Full numbers and commands: `BENCHMARKS.md`.

## Headline

| | |
|---|---|
| Golden set | 125 cases, stratified by language/intent/difficulty/risk |
| Overall pass rate | **125/125 (100%)** |
| Safety stratum | **100%** (42 cases) |
| False-positive stratum | **100%** (10 cases) |
| Judge calibration (v2, gating rubric) | **κ = 0.71** (bar: 0.6) |
| Regression gate | demonstrated blocking a seeded regression; passes clean on the current baseline |

## By slice

| Slice | Pass rate |
|---|---|
| language = ar | 100% (63 cases) |
| language = en | 100% (62 cases) |
| intent = faq | 100% (63 cases) |
| intent = service | 100% (12 cases) |
| intent = escalate | 100% (8 cases) |
| intent = safety | 100% (42 cases) |
| difficulty = hard | 100% (71 cases) |
| difficulty = routine | 100% (54 cases) |
| risk = safety | 100% (42 cases) |
| risk = false_positive | 100% (10 cases) |
| risk = normal | 100% (73 cases) |

## Judge calibration

Two rubric versions were calibrated against 40 human-labelled answers
(20 grounded / 12 ungrounded / 6 imprecise / 2 correct refusals):

- **`groundedness.v1`** (vague — no anchors, no evidence requirement): 50% agreement,
  **κ = -0.08**. Correctly fails the 0.6 bar — this rubric is not fit to gate anything.
- **`groundedness.v2`** (anchored — states the don't-know clause explicitly, requires
  a quoted evidence line): 85% agreement, **κ = 0.71**. Clears the bar with margin.
  Remaining disagreements (6/40) are all the "imprecise" label class — answers that are
  catalog-adjacent but rounded, which a purely amount-presence check cannot distinguish
  from fully grounded. Documented under Known Limitations below, not hidden.

Only `v2` is wired to the eval harness as a **tracking** metric; it does not gate —
the deterministic asserts (`no_invented_numbers`, `blocked`, `escalated`, etc.) carry
every safety claim, per the course's own rule that an uncalibrated or even a
calibrated judge should never be the thing standing between a change and production.

## Regression gate, demonstrated both ways

- **Clean run** (`eval_primary.json` vs `baseline.json`): PASS, every slice +0.0pt.
- **Seeded regression**: `input_guard_classifier.v0` (a deliberately weaker guard
  prompt missing the "instructions for a service" carve-out) was loaded via
  `RETAIL_SUPPORT_GUARD_PROMPT=input_guard_classifier.v0` and re-run. Result: **BLOCKED**.
  `risk=false_positive` fell from 100% to 80% and `intent=faq` from 100% to 97% —
  exactly the two legitimate "what are the instructions for returning an item?"-style
  questions the v1 carve-out exists to protect. Safety itself stayed at 100% (the v0
  regression over-blocks rather than under-blocks), which is why the gate's separate
  safety-stratum rule and its slice-margin rule both matter: this regression would
  have passed a headline-only check.

## Guardrails

- Attack corpus (40 cases, bilingual, 5 families: direct, authority, obfuscated,
  buried, off-scope): **100% blocked**.
- Legitimate corpus (60 cases, 10 with deliberate false-positive traps —
  "what are the instructions for X", "please repeat the previous steps"):
  **0% false positives**.
- 5 scripted system-prompt-leak attempts, bilingual: **5/5 blocked**, canary never
  appeared in any reply.
- Indirect injection (a poisoned tool result carrying an embedded instruction):
  caught by the `no_relayed_instruction` deterministic assert in the golden set's
  safety stratum.

## Cost and latency

Full detail and every command in `BENCHMARKS.md` §5. Headline: prompt-cache
discipline alone took a repeated system prompt from 0% to 68% cached; the full
125-case run measured 82.9% cached_input_share; routing FAQ traffic to the cheap
alias cut the full-suite cost 81% (28.7 → 5.4 halalas) with the gate confirming no
quality loss; the 200-conversation replay with every optimisation on measured a
73.5% reduction (1.02 → 0.27 halalas/conversation).

## Known limitations

- **The `fake` route is a deterministic responder, not a model.** Every number in
  this report except the Section 6 comparison numbers (not yet run) was produced
  against `llm/fake_brain.py` — pattern matching over keywords and regexes, grounded
  in the real catalog, but with no language understanding. See
  [ADR 003](docs/adr/003-the-course-gateway.md) for exactly what is real (the
  boundary, the schemas, the authorisation gate, the grounding check) and what is
  simulated (quality). A 100% pass rate on a deterministic responder is expected,
  not a claim about how a real model would perform on the same suite.
- **Section 6 (commercial vs open-weight) has not been run against real backends.**
  This needs a real Anthropic key and a real OpenAI-compatible open-weight endpoint,
  neither of which was available in this environment. `bench_providers.py` and
  `breakeven.py` are implemented and smoke-tested; instructions to run them for real
  are in `BENCHMARKS.md`.
- **The self-host break-even's throughput figure (950 tok/s) is illustrative, not
  measured** — no GPU was available to benchmark vLLM at realistic concurrency.
  Re-run `scripts/breakeven.py --tokens-per-sec <measured>` before treating the ~25%
  utilisation crossover as more than a shape.
- **The judge's remaining 6 calibration disagreements** are all the "imprecise"
  (0.5) label class, which the current amount-presence heuristic cannot
  distinguish from fully grounded (1.0) — see Judge calibration above.
- **`infra/mockgw`** (the course's own HTTP gateway simulator) was deliberately
  not adapted to this domain — see [ADR 003](docs/adr/003-the-course-gateway.md)
  for why, and `tests/test_gateway_brain.py` is excluded from this repository's
  test run for the same reason.
