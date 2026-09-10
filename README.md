# Baseeta Support — LLM Application Engineering Capstone

## Authors

- Asma Alshilash
- Sara Alshahrany
- Sara Haddad

**Programme:** SDA-AIE-213 — LLM Application Engineering, SDAIA Academy
**Cohort dates:** September 6, 2026 – September 9, 2026
**SDAIA Academy GitHub Link:** https://github.com/SDAIAAcademy

## What this is

**Baseeta Support** is a bilingual (Arabic/English) LLM support assistant for
**Baseeta Retail (بسيطة للتجزئة)**, a fictional Saudi retail chain selling
electronics, home goods and fashion online and in six stores. It answers
questions about orders, returns, exchanges, shipping and warranty; looks up a
customer's own order status; files a return or exchange (behind an
authorisation gate); and hands off to a human agent when it should.

The domain — the product catalog, the order schema, the tools, the prompts,
the corpora, the guards, and the golden set — is originated for this
submission, not adapted from the course's reference application (Murshid,
a citizen-services assistant). The application boundary, guard skeleton,
evaluation harness, caching and observability layers are built on the
course's own `murshid/` plumbing, which the capstone brief explicitly
invites: see [`DECISIONS.md`](DECISIONS.md) for exactly what was kept, what
was replaced, and why.

## Running it

**Zero setup.** Every default route points at a deterministic, retail-aware
responder (`llm/fake_brain.py`) — no API key, no network, no GPU. From a
fresh clone:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.lock
export PYTHONPATH=src
python -c "
from retail_support.app import build_assistant
from retail_support.domain.session import Session
a = build_assistant()
s = Session()
print(a.ask('What is your return policy for electronics?', s).text)
"
```

Or open the Colab notebook: **`notebooks/capstone.ipynb`** — Runtime → Run
all reaches a working bilingual conversation with nothing else installed.
That notebook is the primary submission artefact; this repository is its
supporting code.

To point a route at a **real** backend (for the Section 6 commercial-vs-open-weight
comparison), set two environment variables per route — no code change:

```bash
export RETAIL_SUPPORT_COMPARISON_BASE_URL=https://api.anthropic.com
export RETAIL_SUPPORT_COMPARISON_API_KEY=sk-ant-...
python eval/harness.py --route comparison
```

## Results

125-case golden set: **100% pass**, safety 100%, false-positive rate 0%.
Judge calibrated to κ=0.71 (anchored rubric) vs κ=-0.08 (vague rubric — fails
correctly). Regression gate demonstrated blocking a seeded regression.
Routing cut full-suite cost 81% with zero quality loss. Full numbers, every
one with the command that produced it: [`BENCHMARKS.md`](BENCHMARKS.md).
Sliced results, judge calibration detail, and known limitations:
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Repository layout

```
src/retail_support/
├─ llm/            the model boundary (LLMClient, adapters, FakeClient, fake_brain.py)
├─ domain/         catalog.py (product catalog), order.py (ReturnCase/ReturnRequest), session.py
├─ tools/          check_order_status · create_return_request · escalate_to_agent
├─ guards/         input/output guards, refusals
├─ prompts/        versioned prompt library (extract_return_case, route_intent,
│                  answer_faq, service_workflow, input_guard_classifier, judge_groundedness)
├─ pipeline/       router, FAQ handler, service workflow, tool loop, structured extraction
├─ caching/        exact + semantic response cache
└─ observability/  structured logging, cost meter
eval/               golden set builder, harness, judge calibration, regression gate
scripts/            every measurement in BENCHMARKS.md
data/               generated corpora (attack/legit/golden/replay), product_catalog.yaml
docs/adr/           architecture decision records
notebooks/          the Colab submission notebook
```

## Track and scope

**Track D — Retail order support**, as specified in the capstone brief.
**Extension:** none claimed — mandatory scope only.

## Documentation

- [`DECISIONS.md`](DECISIONS.md) — the ADR index and two trade-offs reversed during development
- [`BENCHMARKS.md`](BENCHMARKS.md) — every measured number and the command that produced it
- [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) — sliced results, calibration, known limitations
- [`docs/context_budget.md`](docs/context_budget.md) — the request-size budget, measured
- [`docs/adr/`](docs/adr/) — architecture, model/routing, and the zero-key simulator decision
