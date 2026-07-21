PY := $(HOME)/miniconda3/envs/personal/bin/python
TENANT ?= duckdb

.PHONY: help install corpus ingest test lint fmt eval eval-full agent-demo agent-baseline agent-dry dev clean

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

eval: ## Run the labeled retrieval eval (fast arms only)
	$(PY) -m src.eval retrieval --tenant $(TENANT) --out eval/retrieval/report.md

eval-full: ## Retrieval eval including the cross-encoder arms (slow: ~7s/query)
	$(PY) -m src.eval retrieval --tenant $(TENANT) \
		--reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --out eval/retrieval/report.md

dev: ## Run API + console (Milestone 3+)
	@echo "not implemented until Milestone 3"

clean:
	rm -rf data/index .pytest_cache **/__pycache__
