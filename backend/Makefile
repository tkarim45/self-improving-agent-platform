PY := $(HOME)/miniconda3/envs/personal/bin/python
TENANT ?= duckdb

.PHONY: api help install corpus ingest test lint fmt eval eval-full agent-demo agent-baseline agent-dry traces golden golden-live flywheel-traffic flywheel-cycle flywheel-log sim sim-dry curve dev clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install deps into the personal conda env
	$(PY) -m pip install -r requirements.txt

corpus: ## Fetch the DuckDB documentation corpus into data/corpus/
	$(PY) -m src.corpus fetch

ingest: ## Ingest the corpus into the hybrid index (make ingest TENANT=duckdb)
	$(PY) -m src.ingest data/corpus/$(TENANT) --tenant $(TENANT)

test: ## Run the offline test suite (no network, no cloud spend)
	$(PY) -m pytest

lint: ## Lint
	$(PY) -m ruff check src tests

fmt: ## Format + autofix
	$(PY) -m ruff format src tests && $(PY) -m ruff check --fix src tests

agent-demo: ## Run the M2 agent demo on real Bedrock (SPENDS MONEY; source creds first)
	$(PY) -m src.agent demo --spend-limit 0.20 --out eval/agent/m2_demo.json

agent-baseline: ## Always-cheap baseline for the router comparison (spends money)
	$(PY) -m src.agent demo --router cheap --spend-limit 0.20 --out eval/agent/m2_always_cheap.json

agent-dry: ## Exercise the agent plumbing with the fake provider (no spend)
	$(PY) -m src.agent demo --dry-run

traces: ## View persisted request traces (python -m src.ops traces|summary|show <id>)
	$(PY) -m src.ops summary --db data/traces.db
	$(PY) -m src.ops traces --db data/traces.db

golden: ## Golden gate in replay mode (deterministic, free — the CI gate)
	$(PY) -m src.eval golden --replay eval/golden/records.json --threshold 0.75

golden-live: ## Golden gate for real on Bedrock, with the LLM-judge (SPENDS ~$0.30)
	$(PY) -m src.eval golden --live --judge --threshold 0.75 \
		--records-out eval/golden/records.json --out eval/golden/report_good.md

flywheel-traffic: ## Run the traffic batch through the agent, traced (SPENDS ~$1)
	$(PY) -m src.flywheel traffic --live --db data/traces.db --ts $$(date -u +%Y-%m-%dT%H:%M:%S)

flywheel-cycle: ## One flywheel pass: mine -> train -> shadow -> canary -> decide (free)
	$(PY) -m src.flywheel cycle --db data/traces.db --ts $$(date -u +%Y-%m-%dT%H:%M:%S)

flywheel-log: ## Promotion history + active config
	$(PY) -m src.flywheel log

sim: ## 6 simulated weeks of traffic driving the flywheel unattended (SPENDS ~$2)
	$(PY) -m src.sim run --weeks 6 --k 12 --live --fresh --state-dir data/sim

sim-dry: ## Simulator plumbing check with the fake provider (free)
	$(PY) -m src.sim run --weeks 3 --k 6 --dry-run --fresh --state-dir /tmp/simdry

curve: ## Render the headline improvement curve from the last simulation
	$(PY) -c "from pathlib import Path; from src.sim.chart import render; \
	print(render(Path('data/sim/weekly.json'), Path('eval/sim/curve.png')))"

eval: ## Run the labeled retrieval eval (fast arms only)
	$(PY) -m src.eval retrieval --tenant $(TENANT) --out eval/retrieval/report.md

eval-full: ## Retrieval eval including the cross-encoder arms (slow: ~7s/query)
	$(PY) -m src.eval retrieval --tenant $(TENANT) \
		--reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --out eval/retrieval/report.md

api: ## Run the M7 FastAPI backend on :8000 (dry-only; SIAP_ALLOW_LIVE=1 to allow spend)
	$(PY) -m src.api --reload

dev: ## Run API + frontend hint
	@echo "backend:  make api            (http://127.0.0.1:8000)"
	@echo "frontend: cd ../frontend && npm run dev   (http://localhost:3000)"

clean:
	rm -rf data/index .pytest_cache **/__pycache__
