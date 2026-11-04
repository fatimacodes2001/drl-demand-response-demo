# drl-demand-response-demo

Reproducible deep reinforcement learning experiments on a **synthetic home-battery demand-response task**.
Five agents (DQN, PPO, DDPG, PER-DDPG and an AlphaZero-style search agent) share one config-driven harness
that trains from random weights, measures improvement against a known optimum, tracks every run in MLflow
and can re-run any recorded run to check that it reproduces.

All data is generated from integer day seeds. Nothing here comes from any employer, client or research
partner. The task is not a board game, but the loop is the same as game-playing RL: sequential decisions,
delayed reward, search versus a learned policy, and a measurable optimum.

## Contents
1. [The task](#1-the-task)
2. [Algorithms](#2-algorithms)
3. [Install](#3-install)
4. [Run experiments](#4-run-experiments)
5. [View runs and plots](#5-view-runs-and-plots)
6. [Configs](#6-configs)
7. [Reproducibility](#7-reproducibility)
8. [Results](#8-results)
9. [GPU](#9-gpu)
10. [Layout](#10-layout)
11. [Known limitations](#11-known-limitations)

## 1. The task
One episode is one day of 48 half-hour steps. A 12.5 kWh home battery earns money by charging when
electricity is cheap and discharging when it is expensive or during an FCAS-style frequency event.

| item | definition |
|---|---|
| Observation (13 values) | state of charge, time of day (sin/cos), next 6 prices, current load and PV, event now, event soon |
| Action | 5 discrete levels from full charge to full discharge (DQN, PPO, AlphaZero), or one float in [-1, 1] (DDPG) |
| Reward | -(energy bill) + FCAS bonus for discharging in an event - battery degradation; end-of-day state of charge is valued at the day's mean price |
| Data | Load, PV, price spikes and events are generated per day from a seed. 200 train days, disjoint validation (5000+) and test (20000+) days |

Because prices, load and events are known day-ahead, the dynamics are deterministic given the day. That gives
two things: an exact simulator for tree search, and a **dynamic-programming oracle** that computes the optimal
return for any day. Every score below is reported as **% of the oracle's saving over doing nothing**, so
numbers are comparable across algorithms and easy to sanity-check (0 = no control, 100 = optimal).

## 2. Algorithms
| name (config) | file | what it is | action space |
|---|---|---|---|
| `dqn` | `agents/dqn.py` | Double DQN with a target network and uniform replay | discrete |
| `ppo` | `agents/ppo.py` | Clipped-surrogate PPO with GAE and 8 parallel env copies | discrete |
| `ddpg` | `agents/ddpg.py` | DDPG: deterministic actor, Q critic, soft target updates, Gaussian exploration noise | continuous |
| `per_ddpg` | `agents/ddpg.py` | DDPG with proportional prioritised replay and annealed importance weights (`algo.per: true`) | continuous |
| `alphazero` | `agents/alphazero.py` | PUCT tree search guided by a policy/value network, trained on its own search visit counts and returns | discrete |

Notes on `alphazero`: the loop is the AlphaZero one (self-play with search, train on the search results,
repeat). The "opponent" is the stochastic day rather than a second player, and the known env acts as the
model, so there is no learned dynamics as in MuZero. Search runs on a numpy copy of the weights.

Reference policies for comparison: `no_control` (idle battery), `random`, `rule_based` (charge in the cheapest
30% of the day, discharge in the priciest 30% or during events) and the DP `oracle`.

## 3. Install
Requires Python 3.10+ (developed on 3.12). Linux, macOS or Docker.

```bash
git clone <this repo> && cd drl-demand-response-demo
make setup                      # venv + pinned deps + editable install of the `dr` package
source .venv/bin/activate       # or call .venv/bin/python directly (the Makefile does)
make test                       # about 15 s, should print "10 passed"
```
Without make: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`

For a CUDA machine install the matching torch build first (for example
`pip install torch --index-url https://download.pytorch.org/whl/cu124`), then the rest of `requirements.txt`.

Docker (CUDA base image): `docker build -t drl-dr . && docker run --gpus all drl-dr` runs `make reproduce`.

## 4. Run experiments
| command | what it does | time (1 CPU core per run) |
|---|---|---|
| `make test` | determinism, resume, fingerprint and oracle-bound tests | ~15 s |
| `make reproduce` | all 5 algorithms, 1 seed, short budget, into `results/quick` | under a minute; a smoke run, so scores are low and not comparable to the table below |
| `make sweep` | all 5 algorithms x 5 seeds, builds the table and plots, logs a `summary` run | ~6 min with 6 parallel jobs |
| `make verify RUN=results/dqn/seed0` | re-run a recorded run and check that it reproduces | ~30 s for DQN |
| `make bench` | CPU vs GPU update throughput (`PROFILE=1` adds a `torch.profiler` trace) | ~1 min |
| `make ui` | open the MLflow UI | |

A single run, with any config value overridden from the command line:
```bash
python scripts/train.py --config configs/dqn.yaml --seed 0
python scripts/train.py --config configs/ppo.yaml --seed 3 --device cuda --set algo.lr=1e-3 train.total_steps=500000
python scripts/train.py --config configs/dqn.yaml --seed 0 --resume      # continue from results/dqn/seed0/ckpt.pt
```
A custom sweep: `python scripts/sweep.py --configs configs/dqn.yaml configs/ddpg.yaml --seeds 0 1 2 --jobs 4`.
Finished runs with an identical fingerprint are skipped, so interrupted sweeps can simply be started again.
Use `--force` to re-run regardless.

Each run prints its learning curve as it goes. The first line is the untrained "before" policy:
```
[dqn s0] steps=      0 train=    nan val_saving= -0.36 ( -4.3% of oracle) 0s
[dqn s0] steps=  60000 train=  -0.87 val_saving=  7.05 ( 84.8% of oracle) 26s
```

## 5. View runs and plots
**MLflow** (primary): `make ui`, then open http://127.0.0.1:5000. Experiment `drl-demand-response` holds:
- one run per (config, seed), named `<variant>-seed<k>-<fingerprint prefix>`. It has all config values as params,
  the fingerprint components and git state as tags, validation metrics per evaluation step (so you get live
  curves), `test_*` final metrics, and artifacts: `config.yaml`, `meta.json`, `metrics.jsonl`, `final.json`,
  `plots/training_curve.png` and `plots/dispatch_day<N>.png` (battery state of charge against the oracle).
- a `summary` run (tag `kind=summary`) with the 5-seed comparison, `plots/learning_curves.png`, `summary.md`
  and `baselines.json`.
- a `bench` run (tag `kind=bench`) with the throughput plot and raw numbers.

Useful UI steps: filter `tags.variant = "dqn"`, select several runs and use Compare, or group by `tags.variant`.
Storage is `./mlflow.db` (SQLite). Point at a shared server with `MLFLOW_TRACKING_URI=http://host:5000`.
Disable tracking with `--set tracker=none`.

**On disk**, per run, `results/<variant>/seed<k>/`:
```
config.yaml    resolved config (after extends and overrides)
meta.json      fingerprint parts, git SHA and dirty flag, python, torch, device, GPU, host, start time
metrics.jsonl  one row per evaluation, including the untrained row at step 0
final.json     held-out test metrics plus the fingerprint
plots/         training_curve.png, dispatch_day<N>.png
ckpt.pt, final.pt   checkpoints (git-ignored; include replay buffers)
```
Sweep-level files: `results/summary.md`, `results/learning_curves.png`, `results/baselines.json`.

Plot helpers live in `src/dr/plotting.py` (`run_curves`, `compare_curves`, `dispatch`, `bench`). They return
matplotlib figures, and `Tracker.log_figure` saves them under the run directory and attaches them to MLflow.
Re-plot any finished run: `python scripts/evaluate.py --run results/dqn/seed0 --day 20003`.

## 6. Configs
An experiment is one YAML file that `extends` `configs/base.yaml`; later keys override earlier ones.
```yaml
# configs/per_ddpg.yaml
extends: ddpg.yaml
name: per_ddpg
algo: {per: true, per_alpha: 0.6, per_beta_start: 0.4, train_total: 60000}
```
`base.yaml` sets the shared pieces: `seed`, `device` (`auto` = CUDA if available), `tracker` (`mlflow` or
`none`), `env` parameters (FCAS price, degradation, feed-in), `train` (`total_steps`, `n_train_days`) and
`eval` (`every_steps`, `val_days`, `test_days`). Anything can be overridden with `--set a.b=value`.
`name` decides the output folder and MLflow variant label.

## 7. Reproducibility
Each run has a **fingerprint**: a 12-character hash of

| component | what goes in |
|---|---|
| `config_hash` | the fully resolved config including the seed (logging-only keys `tracker` and `experiment` are excluded) |
| `code_hash` | the contents of every Python file in `src/dr` |
| `data_hash` | every generated day the run can touch (train, validation, test) plus the env parameters |
| `libs` and `device` | torch and numpy versions, and the resolved device (`cpu` or `cuda`) |

It is stored in `meta.json`, `final.json` and the MLflow tags. Two runs with the same fingerprint should give the
same metrics. A change to the config, any source file, the data generator or library versions changes it, so a
number can always be traced to exactly what produced it. With real data, the dataset checksum (for example a DVC
hash) would go into `data_hash`.

How runs are made repeatable:
- Python, numpy and torch are seeded, each agent owns its own numpy `Generator`, torch runs one CPU thread and
  deterministic algorithms are requested. CPU runs are bitwise repeatable.
- Train, validation and test days are disjoint. Hyperparameters were tuned on validation days (and test days
  10000-10019 with one seed); the reported test days 20000+ were never used for a decision.
- Results come from several seeds with mean and standard deviation, not one curve.
- Dependencies are pinned in `requirements.txt` and the Docker image sets `CUBLAS_WORKSPACE_CONFIG`.

How to check it yourself:
```bash
make verify RUN=results/ddpg/seed1
# fingerprint MATCH | learning curve MATCH | test return 1.153986 vs 1.153986 MATCH
```
`verify` re-runs the recorded `config.yaml` in a temp folder and compares the fingerprint, every learning-curve row
and the test return. It exits non-zero on a mismatch and says which part of the fingerprint changed. Across
different GPUs use `scripts/verify.py --run ... --tol 1e-3`, because float kernels can differ between devices.
`make test` also asserts that the same seed gives identical metrics for all five algorithms, that a different seed
differs, that finished runs are skipped, that `--resume` continues from the checkpoint, and that the fingerprint
reacts to config, data and device changes.

Honest limits: `--resume` continues training from weights, optimiser, replay and RNG state, but is not claimed to
be bit-identical to an uninterrupted run. CUDA runs are deterministic on one machine but may differ across GPU
models. The tuning history above is part of the record.

## 8. Results
5 seeds, 50 held-out test days, 1 CPU core per run on an Apple M4 Pro. Mean ± std over seeds.
Re-running `make sweep` gave identical numbers three times, including after code refactors.

| method | test saving $/day | % of oracle saving | env steps | train wall (s) |
|---|---|---|---|---|
| no_control | 0.00 | 0.0 | - | - |
| random | -5.72 | -65.2 | - | - |
| rule_based | 0.34 | 3.9 | - | - |
| alphazero | 7.57 ± 0.29 | 86.3 ± 3.3 | 30k | 83 |
| ddpg | 7.92 ± 0.12 | 90.2 ± 1.4 | 60k | 58 |
| dqn | 7.74 ± 0.11 | 88.2 ± 1.3 | 60k | 30 |
| per_ddpg | 7.78 ± 0.18 | 88.6 ± 2.1 | 60k | 84 |
| ppo | 6.61 ± 1.20 | 75.3 ± 13.6 | 1M | 49 |

What the numbers say:
- DDPG is best at its budget; DQN is the cheapest to train. PER gave no measurable gain over uniform replay
  here (88.6 ± 2.1 against 90.2 ± 1.4) and costs about 1.5x the time. DDPG's continuous action can also place
  the battery off the 1.25 kWh grid the oracle optimises over, so a continuous policy could in principle
  slightly exceed 100%.
- PPO needs about 17x more samples than DQN and varies a lot across seeds.
- AlphaZero starts near 24% with an untrained network because search over the exact model already plans well,
  and training lifts it to about 86%.
- The algorithms use different step budgets and evaluation cadences (set in their configs), so the learning
  curves are not matched-compute comparisons.

`results/learning_curves.png` shows validation curves; `results/<variant>/seed0/plots/dispatch_day20000.png`
shows a trained policy's battery schedule next to the oracle's.

## 9. GPU
The networks are small, and env stepping and tree search are CPU Python, so at the default sizes a GPU does not
speed up training. `scripts/bench.py` measures update throughput across network sizes and batch sizes and logs a
plot. Speedup of the accelerator over one CPU thread (Apple M4 Pro, MPS; CUDA not yet measured):

| network | batch 128 | batch 1024 | batch 8192 |
|---|---|---|---|
| 128 wide x 2 layers | 0.6x | 1.6x | 5.4x |
| 512 wide x 4 layers | 3.8x | 4.0x | 2.5x |
| 2048 wide x 4 layers | 2.4x | 2.6x | 2.0x |

Run `make bench` (add `PROFILE=1` for a `torch.profiler` trace in `results/bench_trace.json`) on the target
server for CUDA numbers. Train on GPU with `--device cuda`; replay, networks and updates then run on the device,
while environments and tree search stay on the CPU.

## 10. Layout
```
configs/        base.yaml and one file per experiment (dqn, ppo, ddpg, per_ddpg, alphazero)
src/dr/
  env.py          environment, scenario generator, DP oracle and oracle policy
  agents/         dqn.py ppo.py ddpg.py alphazero.py common.py (MLP, replay buffers, seeding)
  runner.py       one experiment: run directory, evaluation loop, checkpoints, test, plots
  evaluate.py     splits, baselines, headline metrics
  fingerprint.py  config + code + data + library hash
  tracking.py     MLflow wrapper (no-op when disabled)
  plotting.py     figure builders
  config.py       YAML loading, extends, overrides, hash
scripts/        train.py sweep.py report.py evaluate.py verify.py bench.py
tests/          determinism, resume, fingerprint, oracle bound
Makefile  Dockerfile  pyproject.toml  requirements.txt
```

## 11. Known limitations
- Synthetic single-household task; it is not a substitute for real grid data.
- AlphaZero here is single-agent planning with a known model, not two-player self-play and not MuZero.
- No CUDA measurements have been taken yet (see GPU section).
- Hyperparameters were lightly tuned on a single seed; the table is not a tuned-to-the-limit comparison.
