.PHONY: help setup setup-dl setup-geo dev-api dev-ui build test test-all lint clean docker demo

PY := python3
BIN := .venv/bin

help:
	@echo "setup      backend core deps (CPU, no CUDA/GDAL)"
	@echo "setup-dl   + PyTorch (read requirements-dl.txt header first)"
	@echo "setup-geo  + rasterio/geopandas/osmnx"
	@echo "dev-api    run FastAPI on :8000"
	@echo "dev-ui     run Vite on :5173"
	@echo "test-all   every suite: core, api, contract, metric, deployment, build"
	@echo "train-kaggle    training workflow  (DATA=... OUT=... EPOCHS=...)"
	@echo "eval-checkpoint val+test+occlusion+baseline  (CKPT=path/to/best.pt)"
	@echo "audit      ML pipeline audit (data / model / backend)"
	@echo "eval-occlusion  occlusion-stratified eval  (CKPT=path/to/best.pt)"
	@echo "docker     docker compose up --build"

.venv:
	$(PY) -m venv .venv && $(BIN)/pip install -U pip wheel

setup: .venv
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install fastapi "uvicorn[standard]" pydantic python-multipart
	cd frontend && npm install

setup-dl: .venv
	$(BIN)/pip install -r requirements-dl.txt

setup-geo: .venv
	$(BIN)/pip install -r requirements-geo.txt

dev-api:
	$(BIN)/uvicorn backend.app.main:app --reload --port 8000

dev-ui:
	cd frontend && npm run dev

build:
	cd frontend && npm run build

test:
	$(PY) tests/run_tests_nopytest.py

train-kaggle:
	$(PY) scripts/kaggle_train.py --data $(or $(DATA),data/processed) \
	  --out $(or $(OUT),checkpoints/unet_r34) --epochs $(or $(EPOCHS),40) --seed 42

eval-checkpoint:
	$(PY) scripts/evaluate_checkpoint.py --checkpoint $(CKPT) \
	  --data $(or $(DATA),data/processed) --out outputs/reports

audit:
	$(PY) -W ignore scripts/audit_ml_pipeline.py --stage data
	$(PY) -W ignore scripts/audit_ml_pipeline.py --stage model
	$(PY) -W ignore scripts/audit_ml_pipeline.py --stage backend

eval-occlusion:
	$(PY) scripts/evaluate_occlusion.py --checkpoint $(CKPT) --data data/processed/test

test-all:
	@echo "== scientific core ==" && $(PY) -W ignore tests/run_tests_nopytest.py | tail -2
	@echo "== resilience metric ==" && $(PY) -W ignore backend/tests/test_resilience_metric.py 2>/dev/null | tail -1
	@echo "== deployment ==" && $(PY) -W ignore backend/tests/test_deployment.py 2>/dev/null | tail -1
	@echo "== trained pipeline ==" && $(PY) -W ignore backend/tests/test_trained_pipeline.py 2>/dev/null | tail -1
	@echo "== api e2e ==" && $(PY) -W ignore backend/tests/test_api_e2e.py 2>/dev/null | tail -2
	@echo "== contract ==" && $(PY) -W ignore backend/tests/test_contract.py 2>/dev/null | tail -2
	@echo "== typecheck ==" && cd frontend && npx tsc --noEmit && echo "  clean"
	@echo "== build ==" && cd frontend && npm run build 2>&1 | tail -1

demo:
	$(PY) scripts/run_demo.py --out outputs/demo

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache frontend/dist
