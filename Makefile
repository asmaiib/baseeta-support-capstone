# RetailSupport — the SDA-AIE-213 golden-thread project.
#
#   make doctor        check your environment before a session starts
#   make help          every target, with one line each
#
# No target here needs an API key. Everything runs against the course gateway in
# infra/mockgw, which `make gateway` starts. Point a route at a real provider with
# two environment variables when you want to (configs/settings.example.env).

PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif
PYTHON := $(BIN)/python
export PYTHONUTF8 = 1
export PYTHONPATH = src

ROUTE ?=
Q ?= How do I renew my commercial licence?
LIMIT ?= 200
LABEL ?= run
ROUTE_FLAG := $(if $(ROUTE),--route $(ROUTE),)

.DEFAULT_GOAL := help
.PHONY: help venv install doctor gateway gateway-stop stop ask chat stream \
        test lint schema-check extract-corpus extract-audit tool-smoke guard-eval \
        leak-attack bench replay replay-before replay-after eval eval-vllm eval-cheap \
        golden calibrate gate baseline eval-cache eval-report token-report breakeven \
        corpora drill-429 drill-outage drill-off stats clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- environment ----------------------------------------------------------
venv:  ## create the virtual environment
	$(PY) -m venv $(VENV)

install: venv  ## install pinned dependencies (no floating installs, ever)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.lock

doctor:  ## check python, config, Arabic rendering and every configured route
	$(PYTHON) -m retail_support.cli doctor

# --- the course gateway ---------------------------------------------------
gateway:  ## run the course gateway in the foreground (Ctrl-C to stop)
	cd infra/mockgw && MOCKGW_SPEED=$${MOCKGW_SPEED:-0.2} ../../$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8080

stats:  ## gateway counters: requests, tokens, cache hits, faults served
	@curl -s http://127.0.0.1:8080/admin/stats

# --- drills ---------------------------------------------------------------
drill-429:  ## Module 2: a 429 storm on the primary model for three minutes
	@curl -s -X POST http://127.0.0.1:8080/admin/fault -H 'content-type: application/json' \
	  -d '{"mode":"rate_limit","seconds":180,"model":"course-flagship","retry_after":2}'

drill-outage:  ## Module 1: the primary model 529s; the fallback hop should serve
	@curl -s -X POST http://127.0.0.1:8080/admin/fault -H 'content-type: application/json' \
	  -d '{"mode":"overload","seconds":300,"model":"course-flagship"}'

drill-off:  ## end any running drill
	@curl -s -X POST http://127.0.0.1:8080/admin/fault -H 'content-type: application/json' -d '{"mode":"off"}'

# --- using the assistant --------------------------------------------------
ask:  ## one question, one answer:  make ask Q="..." [ROUTE=vllm]
	$(PYTHON) -m retail_support.cli $(ROUTE_FLAG) ask "$(Q)"

chat:  ## a windowed conversation
	$(PYTHON) -m retail_support.cli $(ROUTE_FLAG) chat

stream:  ## stream one answer and report TTFT vs total
	$(PYTHON) -m retail_support.cli $(ROUTE_FLAG) stream "$(Q)"

# --- quality gates you run constantly -------------------------------------
test:  ## the unit and contract suites (no network, no keys)
	$(PYTHON) -m pytest

lint:  ## ruff
	$(PYTHON) -m ruff check src tests scripts eval infra

schema-check:  ## every output contract still fits the strict-mode subset
	$(PYTHON) scripts/schema_check.py

# --- Module 2 -------------------------------------------------------------
bench:  ## the same 20 bilingual prompts against every route
	$(PYTHON) scripts/bench_providers.py

token-report:  ## the Arabic token premium, measured per tokenizer
	$(PYTHON) scripts/token_report.py

# --- Module 3 -------------------------------------------------------------
extract-corpus:  ## schema-pass rate over the 50-case corpus [ROUTE=vllm]
	$(PYTHON) scripts/extract_corpus.py $(ROUTE_FLAG)

extract-audit:  ## the same, plus the invented-field audit
	$(PYTHON) scripts/extract_corpus.py $(ROUTE_FLAG) --audit

tool-smoke:  ## the scripted conversation that must trigger exactly one tool call
	$(PYTHON) scripts/tool_smoke.py

# --- Module 4 -------------------------------------------------------------
guard-eval:  ## attack block rate AND legitimate false-positive rate
	$(PYTHON) scripts/guard_eval.py

leak-attack:  ## five system-prompt extraction attempts against the canary
	$(PYTHON) scripts/leak_attack.py

# --- Module 5 -------------------------------------------------------------
golden:  ## rebuild the golden set and print the strata histogram
	$(PYTHON) eval/build_golden.py

eval:  ## the golden set through the real pipeline
	$(PYTHON) eval/harness.py --label $(LABEL) $(ROUTE_FLAG)

eval-vllm:  ## the same suite on the open-weight route
	$(PYTHON) eval/harness.py --label vllm --route vllm

eval-cheap:  ## the same suite on the cheap model
	$(PYTHON) eval/harness.py --label cheap --route cheap

calibrate:  ## judge vs the 40 human labels, both rubrics
	$(PYTHON) eval/build_human_labels.py
	-$(PYTHON) eval/calibrate_judge.py --rubric groundedness.v1.md
	$(PYTHON) eval/calibrate_judge.py --rubric groundedness.v2.md

gate:  ## compare the last run against the baseline
	$(PYTHON) eval/gate.py eval/out/eval_$(LABEL).json --baseline eval/baseline.json

baseline:  ## promote the current run to the baseline (a governed act)
	$(PYTHON) eval/promote_baseline.py $(LABEL)
	@echo "baseline is now eval_$(LABEL).json — say why in the commit message"

eval-report:  ## regenerate EVALUATION_REPORT.md from everything in eval/out
	$(PYTHON) eval/report.py

# --- Module 6 -------------------------------------------------------------
replay:  ## replay conversations and meter them:  make replay LABEL=after CACHE=1
	$(PYTHON) scripts/replay.py --label $(LABEL) --limit $(LIMIT) $(ROUTE_FLAG) \
	  $(if $(CACHE),--cache,) $(if $(SEMANTIC),--semantic,) $(if $(ROUTING),--routing,) \
	  $(if $(CASCADE),--cascade,) $(if $(PROMPT),--prompt $(PROMPT),)

replay-before:  ## the baseline replay, with the cache-killer prompt version
	$(PYTHON) scripts/replay.py --label before --limit $(LIMIT) --prompt answer_faq.v4

replay-after:  ## the optimised replay: prompt cache, response cache, routing + cascade
	$(PYTHON) scripts/replay.py --label after --limit $(LIMIT) --cache --semantic --routing --cascade

eval-cache:  ## the semantic cache's near-miss safety suite
	$(PYTHON) scripts/eval_cache.py

breakeven:  ## self-host vs commercial, against both hosted tiers
	$(PYTHON) scripts/breakeven.py

# --- housekeeping ---------------------------------------------------------
corpora:  ## regenerate every corpus from the curated seeds
	$(PYTHON) scripts/generate_corpora.py

clean:  ## remove run artefacts (never the corpora, never the baseline)
	rm -rf eval/out/*.json logs/*.jsonl .pytest_cache .ruff_cache
