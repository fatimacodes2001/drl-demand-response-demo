PY ?= .venv/bin/python
SEEDS ?= 0 1 2 3 4
ALGOS = configs/dqn.yaml configs/ddpg.yaml configs/per_ddpg.yaml configs/ppo.yaml configs/alphazero.yaml
C4 = configs/connect4_alphazero.yaml

setup:           ## create venv, install pinned deps and the dr package
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .

test:            ## determinism, resume, fingerprint, replay and oracle tests (~30 s)
	$(PY) -m pytest -q tests

reproduce:       ## ~1 min smoke run of every algorithm, 1 seed, short budget
	$(PY) scripts/sweep.py --configs $(ALGOS) --seeds 0 --out results/quick --jobs 3 \
	  --set train.total_steps=6000 eval.every_steps=3000 eval.test_days=10 algo.sims=16
	$(PY) scripts/sweep.py --configs $(C4) --seeds 0 --out results/quick/connect4 --jobs 1 \
	  --set train.total_steps=3000 eval.every_steps=1500 eval.val_games=10 eval.test_games=10 algo.sims=16

sweep:           ## full experiment: every algorithm x 5 seeds -> results/summary.md + learning_curves.png
	$(PY) scripts/sweep.py --configs $(ALGOS) --seeds $(SEEDS) --jobs 5

sweep-c4:        ## Connect Four self-play x 5 seeds -> results/connect4/summary.md
	$(PY) scripts/sweep.py --configs $(C4) --seeds $(SEEDS) --jobs 5 --out results/connect4

verify:          ## re-run a recorded run and check it reproduces: make verify RUN=results/dqn/seed0
	$(PY) scripts/verify.py --run $(RUN)

ui:              ## browse runs, plots and artifacts in MLflow
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

bench:           ## CPU vs GPU update throughput (+ profiler trace with PROFILE=1)
	$(PY) scripts/bench.py $(if $(PROFILE),--profile)
