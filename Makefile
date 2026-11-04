PY ?= .venv/bin/python
SEEDS ?= 0 1 2 3 4
ALGOS = configs/dqn.yaml configs/ddpg.yaml configs/per_ddpg.yaml configs/ppo.yaml configs/alphazero.yaml

setup:           ## create venv and install pinned deps (uv or pip)
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .

test:            ## determinism, resume, oracle-bound tests (~10 s)
	$(PY) -m pytest -q tests

reproduce:       ## ~1 min: tiny end-to-end run of all 3 algos, 1 seed
	$(PY) scripts/sweep.py --configs $(ALGOS) --seeds 0 --out results/quick --jobs 3 \
	  --set train.total_steps=6000 eval.every_steps=3000 eval.test_days=10 algo.sims=16

sweep:           ## full experiment: 3 algos x 5 seeds -> results/summary.md + learning_curves.png
	$(PY) scripts/sweep.py --configs $(ALGOS) --seeds $(SEEDS) --jobs 5

verify:         ## re-run a recorded run and check it reproduces: make verify RUN=results/dqn/seed0
	$(PY) scripts/verify.py --run $(RUN)

ui:              ## browse runs, plots and artifacts in MLflow
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

bench:           ## CPU vs GPU update throughput (+ profiler trace with PROFILE=1)
	$(PY) scripts/bench.py $(if $(PROFILE),--profile)
