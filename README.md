# drl-demand-response-demo

A small, fully reproducible reinforcement-learning harness with two tasks:

- **Demand response** (single agent): a home battery arbitrages a time-of-use tariff and FCAS-style events.
  Five agents (DQN, PPO, DDPG, PER-DDPG and an AlphaZero-style search agent) are scored against an exact
  dynamic-programming optimum.
- **Connect Four** (two-player self-play): the same AlphaZero-style agent plays itself from random weights and
  is scored against fixed opponents and against its own previous checkpoint.

Every run has a fingerprint (config + code + data + libraries + platform), is tracked in MLflow with its plots,
can be re-run and checked bit-for-bit with one command, and resumes from a checkpoint without changing the
result. All data is synthetic or generated; nothing comes from any employer, client or research partner.

**What this is and is not.** It is a demonstration of how I set up training, measurement and reproducibility,
built in a few days for this purpose. The Connect Four agent is a real AlphaZero-style self-play loop (search,
self-play, train on search targets, repeat) with measured improvement from random initialisation, but with a
small MLP, 50 simulations and about 8000 games it is far from strong play. The demand-response task is a
simplified stand-in for the energy-domain research in my publications, not that research itself.

## Contents
1. [Tasks](#1-tasks)
2. [Algorithms](#2-algorithms)
3. [Install](#3-install)
4. [Run](#4-run)
5. [View runs and plots](#5-view-runs-and-plots)
6. [Configs](#6-configs)
7. [Reproducibility](#7-reproducibility)
8. [Results: demand response](#8-results-demand-response)
9. [Results: Connect Four](#9-results-connect-four)
10. [GPU](#10-gpu)
11. [Layout](#11-layout)
12. [Limitations](#12-limitations)

## 1. Tasks

### Demand response (`task: dr`)
One episode is one day of 48 half-hour steps. A 12.5 kWh battery earns money by charging when electricity is
cheap and discharging when it is expensive or during an FCAS-style frequency event.

| item | definition |
|---|---|
| Observation (147 values) | state of charge, time of day (sin/cos), and the remaining day's prices, event flags and net load (load minus PV), zero-padded |
| Action | 5 discrete levels from full charge to full discharge (DQN, PPO, AlphaZero) or one float in [-1, 1] (DDPG) |
| Reward | -(energy bill) + FCAS bonus for discharging in an event - degradation; the end-of-day charge is valued at the day's mean price |
| Data | load, PV, price spikes and events generated per day from a seed: 200 train days, disjoint validation (5000+) and test (20000+) days |

The whole remaining day is observed, so every agent plans from the same information as the tree search and the
**dynamic-programming oracle is a reachable ceiling**, not a clairvoyant one. Scores are reported as **% of the
oracle's saving over doing nothing** (0 = idle battery, 100 = optimal). The oracle runs on a grid 10x finer
than the discrete actions, so it also bounds the continuous DDPG policy; `oracle_discrete` is the exact ceiling
for the 5-action agents (91.2% of the fine oracle's saving on the test days).

### Connect Four (`task: connect4`)
Standard 6x7 board. The observation is two planes (own pieces, opponent pieces) relative to the player to move,
so one network plays both colours. Evaluation plays a fixed set of random two-ply openings, each once as first
and once as second player, against:
- a **random** player,
- a **win/block** heuristic (take an immediate win, otherwise block an immediate threat, otherwise random),
- the agent's **previous checkpoint** (during training), and
- additionally reports the **raw network with no search** so the network's own learning is separated from
  what 50 simulations of tree search provide by themselves.

## 2. Algorithms
| name (config) | file | what it is | task |
|---|---|---|---|
| `dqn` | `agents/dqn.py` | Double DQN with a target network and uniform replay | dr |
| `ppo` | `agents/ppo.py` | Clipped-surrogate PPO with GAE and 8 parallel env copies | dr |
| `ddpg` | `agents/ddpg.py` | DDPG: deterministic actor, Q critic, soft targets, Gaussian exploration noise | dr |
| `per_ddpg` | `agents/ddpg.py` | DDPG with proportional prioritised replay (sum tree, annealed importance weights) | dr |
| `alphazero` | `agents/alphazero.py` | PUCT search guided by a policy/value network, trained on its own search visit counts and returns | dr |
| `c4_alphazero` | `agents/alphazero.py` | same agent, two-player: values flip sign each ply, moves are sampled for the first 8 plies then greedy, batches are mirror-augmented | connect4 |

On demand response the "opponent" of the search agent is the stochastic day and the known env is the model, so
it is single-agent planning, not self-play, and not MuZero (no learned dynamics).

## 3. Install
Python 3.12 (3.10+ should work; developed and recorded on 3.12). Linux, macOS or Docker.
```bash
make setup                 # venv, pinned deps, editable install of the `dr` package
make test                  # ~40 s on a laptop: determinism, resume, fingerprint, replay, rules, golden values
```
Without make: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`

Docker: `docker build -t drl-dr . && docker run --rm drl-dr` runs `make reproduce`. The image is
`python:3.12-slim` plus the pinned requirements; on an x86_64 Linux host the PyPI torch wheel bundles CUDA, so
`docker run --gpus all drl-dr python scripts/train.py --config configs/dqn.yaml --device cuda` trains on the GPU.
On an arm64 Mac the same build gives CPU-only torch and the test suite passes inside it.

## 4. Run
| command | what it does | time on an M4 Pro laptop |
|---|---|---|
| `make test` | the test suite | ~40 s |
| `make reproduce` | smoke run of every algorithm and both tasks, 1 seed, tiny budgets, into `results/quick/` | ~2 min |
| `make sweep` | demand response, 5 algorithms x 5 seeds, builds `results/summary.md` and plots | ~6 min (6 parallel jobs) |
| `make sweep-c4` | Connect Four, 5 seeds x 200k plies, into `results/connect4/` | ~10 min (5 parallel jobs) |
| `make verify RUN=results/dqn/seed1` | re-run a recorded run and check it reproduces | ~1 min for DQN |
| `make bench` | CPU vs GPU update throughput (`PROFILE=1` adds a `torch.profiler` trace) | ~1 min |
| `make golden` | refresh `tests/golden.json` after an intended change to results | ~20 s |
| `make ui` | MLflow UI at http://127.0.0.1:5000 | |

Single runs, overrides, interruption and resume:
```bash
python scripts/train.py --config configs/dqn.yaml --seed 0
python scripts/train.py --config configs/ppo.yaml --seed 3 --device cuda --set algo.lr=1e-3 train.total_steps=500000
python scripts/train.py --config configs/connect4_alphazero.yaml --seed 0 --stop-at 100000   # checkpoint and exit
python scripts/train.py --config configs/connect4_alphazero.yaml --seed 0 --resume           # continue, same result
```
Finished runs with an identical fingerprint are skipped, so an interrupted sweep can simply be started again.
Each run prints its learning curve as it goes; the first line is the untrained policy:
```
[dqn s0] steps=      0 train=    nan val % of oracle saving=  -8.0 0s
[dqn s0] steps=  60000 train=   0.41 val % of oracle saving=  88.3 43s
```

## 5. View runs and plots
**MLflow** (`make ui`, experiment `drl-demand-response`): one run per (config, seed), named
`<variant>-seed<k>-<fingerprint prefix>`, with every config value as a param, the fingerprint parts and git
state as tags, validation metrics per evaluation step, `test_*` final metrics, and artifacts: `config.yaml`,
`meta.json`, `metrics.jsonl`, `final.json`, `plots/training_curve.png` and a task plot (`dispatch_day<N>.png`
for demand response, `sample_game.png` for Connect Four). Each sweep also logs a `summary:<dir>` run with the
comparison table and `learning_curves.png`, and `make bench` logs a `bench` run. Filter by `tags.variant` or
`tags.fingerprint`; the store keeps every run ever logged, including superseded ones. Storage is `./mlflow.db`
(SQLite); set `MLFLOW_TRACKING_URI` for a shared server, or `--set tracker=none` to disable tracking.

**On disk**, per run, `results/<variant>/seed<k>/`:
```
config.yaml     resolved config          metrics.jsonl  one row per evaluation, including the untrained row
meta.json       fingerprint parts, git SHA and dirty flag, python, host, GPU, start time
final.json      held-out test metrics plus the fingerprint
final.pt        network weights only (weights_only=True loadable)      plots/  figures
ckpt.pt         full training state, git-ignored                      mlflow_run_id
```
Sweep-level: `results/summary.md`, `results/learning_curves.png`, `results/baselines.json`; same under
`results/connect4/`. Re-plot a finished run with `python scripts/evaluate.py --run results/dqn/seed0`.
Figure builders live in `src/dr/plotting.py`.

## 6. Configs
An experiment is one YAML file; demand-response configs `extends: base.yaml`, later keys override earlier ones,
and `--set a.b=value` overrides anything from the command line (`1e-3` parses as a float).
```yaml
extends: ddpg.yaml           # configs/per_ddpg.yaml
name: per_ddpg
algo: {per: true, per_alpha: 0.6, per_beta_start: 0.4}
```
Shared keys: `task`, `seed`, `device` (`auto` = CUDA if available), `tracker` (`mlflow` | `none`), `env`, `train`
(`total_steps`, `n_train_days`), `eval` (`every_steps`, `val_days`/`test_days` or `val_games`/`test_games`).
`name` is the output folder and MLflow variant label; names in one sweep must be unique and variants in one
results directory must share task, env and eval settings, or the report refuses to build a table.

## 7. Reproducibility
**Fingerprint.** Each run's identity is a 12-character hash of:

| part | contents |
|---|---|
| `config_hash` | the resolved config including the seed (logging keys `tracker`, `experiment` excluded) |
| `code_hash` | every Python file under `src/dr` (scripts only launch and report, so they are excluded) |
| `data_hash` | every generated day a demand-response run can touch, or the game plus opening seeds for Connect Four |
| `libs`, `platform`, `device` | torch, numpy and Python versions; OS and CPU architecture; cpu or cuda |

It is in `meta.json`, `final.json` and the MLflow tags. Any change to config, source, data generator, library
versions or machine changes it, so a number can always be traced to exactly what produced it, and a finished run
is only skipped when its fingerprint still matches. With real data the dataset checksum would go in `data_hash`.

**Determinism.** Python, numpy and torch are seeded, each agent owns its numpy generator and PPO its torch
generator, torch runs one CPU thread with deterministic algorithms requested. CPU runs are bitwise repeatable.
Checkpoints hold every RNG stream (numpy, torch, CUDA, PPO's generator), replay buffers (the prioritised
buffer's sum tree is stored as built so float sums repeat), live env states and the previous evaluation's
weights, so **interrupting a run and resuming it gives bit-identical metrics to an uninterrupted run**. The test
suite asserts this for all six configurations, plus: same seed gives identical metrics, different seeds differ,
finished runs are skipped, the fingerprint reacts to config, data and device changes, replay sampling follows
priorities, the Connect Four rules, and a golden-value comparison (below).

**Check it yourself.**
```bash
make verify RUN=results/dqn/seed1
# fingerprint MATCH | learning curve MATCH | test return 1.779184 vs 1.779184 MATCH
```
`verify` re-runs the recorded `config.yaml` in a temporary folder and compares the fingerprint, every row of the
learning curve and the test result, exiting non-zero on a mismatch and saying which fingerprint part changed.
Across machines use `scripts/verify.py --run ... --tol 1e-3`: the fingerprint will differ (platform, library
versions) and float kernels may round differently; the metrics should agree closely. `tests/golden.json` holds
tiny-run results recorded on the development machine (macOS arm64); `make test` compares against them to 1e-4
relative as a hard test and bitwise as an expected-to-maybe-fail test, and GitHub Actions runs the suite on
Linux x86_64 on every push.

**Splits and tuning.** Train, validation and test days/openings are disjoint. Hyperparameters were tuned with one
seed on validation days and on test days 10000-10019; the reported test days (20000+) were never used for any
decision. For Connect Four, the temperature schedule, augmentation and budget were chosen from single-seed runs
scored on the validation openings; the test openings were only used for the final table.

**Recorded, not claimed.** CUDA runs use deterministic-algorithm mode but have not been run here; CUDA numbers
and cross-GPU determinism are to be collected on a GPU machine with `make bench` and `make verify --tol`.

## 8. Results: demand response
5 seeds, 50 held-out test days, one CPU core per run (Apple M4 Pro), mean ± sample std. Produced by `make sweep`
at this commit; `make verify` on a recorded run reproduces it exactly.

| method | test saving $/day | % of oracle saving | env steps | train wall (s) |
|---|---|---|---|---|
| no_control | 0.00 | 0.0 | - | - |
| random | -5.72 | -59.5 | - | - |
| rule_based | 0.34 | 3.6 | - | - |
| oracle_discrete (5-action ceiling) | 8.78 | 91.2 | - | - |
| alphazero | 7.91 ± 0.13 | 82.2 ± 1.3 | 30k | 102 |
| ddpg | 9.08 ± 0.04 | 94.3 ± 0.4 | 60k | 93 |
| dqn | 8.49 ± 0.08 | 88.2 ± 0.8 | 60k | 52 |
| per_ddpg | 8.87 ± 0.18 | 92.2 ± 1.9 | 60k | 120 |
| ppo | 8.37 ± 0.08 | 87.0 ± 0.8 | 1M | 68 |

Reading it: DQN reaches 88.2 of the 91.2 points available to a 5-action policy. DDPG's continuous actions let it
pass that ceiling (94.3 of 100). PER gave no gain over uniform replay here and costs more time. PPO needs about
17x more samples than DQN for a similar result. The search agent starts well above zero with an untrained network
because planning with the exact model already works, and training lifts it to 82%; its budget counts env steps
only, not the 64 simulations per step. Budgets and evaluation cadences differ per algorithm (set in the configs),
so the curves in `results/learning_curves.png` are not matched-compute comparisons.

## 9. Results: Connect Four
5 seeds, 100 held-out openings played as both colours (200 games per opponent), 200k self-play plies per seed
(about 8000 games, 560 s each on one core). Produced by `make sweep-c4` at this commit.

| method | win % vs random | win % vs win/block | loss % vs win/block | net only (no search), win % vs win/block |
|---|---|---|---|---|
| random | 48.5 | 3.5 | 96.5 | - |
| winblock (vs itself) | 95.0 | 47.0 | 50.0 | - |
| untrained search, 50 sims | 95.0 | 49.5 | 48.0 | 0.0 |
| **c4_alphazero** (5 seeds) | 96.3 ± 0.8 | **68.5 ± 4.3** | **26.0 ± 3.7** | **45.8 ± 4.6** |

Reading it: at equal search budget, training moved the agent from parity with the heuristic (49.5% wins, 48%
losses) to 68.5% wins and 26% losses, and the raw network alone went from losing every game to winning 46%, so
the network learned real play rather than relying on search. During training the agent beat its previous
checkpoint in most evaluations (per-seed curves in `results/connect4/*/seed*/metrics.jsonl`,
`val_winrate_prev`). Validation win rate against the heuristic rises from about 58% to 72% over the run
(`results/connect4/learning_curves.png`); the first 20k plies show no gain, then learning takes off.
Untrained search with 200 simulations already wins 79% (loss 16%), so search depth remains the cheaper route to
strength at this scale; this is a demonstration of a working, measured self-play loop, not of a strong player.

## 10. GPU
The networks here are small, and env stepping and tree search are CPU Python, so at these sizes a GPU does not
speed training up. `make bench` measures learner-update throughput across network and batch sizes and logs a
plot. Speedup of the accelerator over one CPU thread on this laptop (Apple M4 Pro, MPS; CUDA not yet measured):

| network | batch 128 | batch 1024 | batch 8192 |
|---|---|---|---|
| 128 wide x 2 layers | 0.7x | 1.6x | 6.0x |
| 512 wide x 4 layers | 4.4x | 4.2x | 2.6x |
| 2048 wide x 4 layers | 2.3x | 2.6x | 2.0x |

On a CUDA machine run `make bench` (add `PROFILE=1` for a `torch.profiler` trace) and train with
`--device cuda`: replay sampling, networks and updates move to the device, environments and search stay on CPU.

## 11. Layout
```
configs/            base.yaml, dqn, ppo, ddpg, per_ddpg, alphazero (demand response); connect4_alphazero
src/dr/
  env.py              demand-response env, scenario generator, vectorised DP oracle and oracle policy
  games/connect4.py   Connect Four env, scripted opponents, fixed-opening games
  tasks/              dr.py, connect4.py: env factory, evaluation protocol, baselines, data hash, plots
  agents/             common.py (BaseAgent, MLP, replay buffers, sum tree), dqn, ppo, ddpg, alphazero
  runner.py           one experiment: run directory, eval loop, checkpoints, stop/resume, test, plots
  fingerprint.py      config + code + data + libs + platform hash
  tracking.py         MLflow wrapper (no-op when disabled)       plotting.py   figure builders
  config.py           YAML loading, extends, overrides, hash      evaluate.py   demand-response splits and baselines
scripts/            train, sweep, report, evaluate, verify, bench, golden
tests/              determinism, resume, fingerprint, replay, config, Connect Four rules, golden values
.github/workflows/  CI on Linux        Makefile  Dockerfile  pyproject.toml  requirements.txt
```

## 12. Limitations
- Demand response is a synthetic single-household task with perfect day-ahead information; the real problem has
  forecast error, which neither the oracle nor the agents face here.
- The Connect Four agent is weak in absolute terms (MLP, 50 simulations, ~8000 games); the evaluation set is
  small enough that single-seed win rates move by ±8 points, which is why 5 seeds and sample std are reported.
- `code_hash` covers all of `src/dr`, including plotting code, so a plot-only change marks recorded runs as stale
  even though it cannot alter results. This is deliberate caution, not precision.
- No CUDA measurements yet; cross-architecture bitwise agreement is tested but not promised.
- The MLflow store keeps superseded runs from development; filter by `tags.fingerprint` for current ones.
