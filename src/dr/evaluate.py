"""Policy evaluation on fixed day sets, reference baselines and headline metrics."""
import numpy as np

from .env import DREnv, Scenario, oracle, N_ACTIONS

# Train days are 0..n_train_days-1. Hyperparameters were tuned on the val days and on test days
# 10000-10019, so the reported test set starts at 20000 and was never used for any decision.
VAL_BASE, TEST_BASE = 5000, 20000


def days_for(split, n):
    """Return the `n` fixed day seeds of a split ('val' or 'test')."""
    base = {"val": VAL_BASE, "test": TEST_BASE}[split]
    return list(range(base, base + n))


def rollout(policy, day, env_cfg):
    """Total return of `policy` on one day."""
    env = DREnv(env_cfg)
    env.reset(day)
    total, done = 0.0, False
    while not done:
        _, r, done, _ = env.step(policy(env))
        total += r
    return total


def evaluate(policy, days, env_cfg):
    """Array of per-day returns."""
    return np.array([rollout(policy, d, env_cfg) for d in days])


def no_control(env):
    """Leave the battery idle."""
    return 2


def rule_based(env):
    """Charge in the cheapest 30% of the day, discharge in the priciest 30% or during events."""
    s, t = env.sc, env.t
    if s.event[t]:
        return 4
    srt = sorted(s.price)
    lo, hi = srt[int(0.3 * len(srt))], srt[int(0.7 * len(srt))]
    return 0 if s.price[t] <= lo else 4 if s.price[t] >= hi else 2


def random_policy(seed):
    """Uniformly random discrete actions."""
    rng = np.random.default_rng(seed)
    return lambda env: int(rng.integers(N_ACTIONS))


def oracle_returns(days, env_cfg, sub=10):
    """Per-day optimal returns from dynamic programming on a grid `sub` times finer than the actions."""
    p = DREnv(env_cfg).p
    return np.array([oracle(Scenario(d, p), sub) for d in days])


def summarise(returns, days, env_cfg):
    """Mean return, saving versus no control, and the share of the oracle's saving achieved."""
    nc = evaluate(no_control, days, env_cfg)
    opt = oracle_returns(days, env_cfg)
    saving = returns - nc
    return dict(
        return_mean=float(returns.mean()), return_std=float(returns.std()),
        saving_per_day=float(saving.mean()),
        pct_of_oracle_saving=float(100 * saving.mean() / (opt - nc).mean()),
    )


def baselines(days, env_cfg):
    """Metrics for the no-control, rule-based and random references plus both oracles.

    `oracle_discrete` is the exact optimum on the agents' 5-action grid, so it is the ceiling for the
    discrete agents; `oracle` uses a 10x finer grid and is the reference every score is divided by.
    """
    pols = {"no_control": no_control, "rule_based": rule_based, "random": random_policy(0)}
    out = {k: summarise(evaluate(f, days, env_cfg), days, env_cfg) for k, f in pols.items()}
    out["oracle_discrete"] = summarise(oracle_returns(days, env_cfg, sub=1), days, env_cfg)
    out["oracle"] = summarise(oracle_returns(days, env_cfg), days, env_cfg)
    return out
