"""Plot builders for runs, comparisons and benchmarks; return figures for saving or MLflow logging."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import DREnv, Scenario, STEPS, oracle_policy


def run_curves(rows, ref=None):
    """Training curves for one run: validation % of oracle saving and train return against env steps."""
    x = [r["env_steps"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(x, [r["val_pct_of_oracle_saving"] for r in rows], marker="o")
    for name, v in (ref or {}).items():
        ax[0].axhline(v, color="gray", ls="--", lw=1, label=name)
    ax[0].axhline(100, color="k", lw=0.8, label="oracle")
    ax[0].set(xlabel="environment steps", ylabel="% of oracle saving (val days)", title="Validation performance")
    ax[0].legend(fontsize=8)
    tr = [(r["env_steps"], r["train_return"]) for r in rows if r.get("train_return", float("nan")) == r.get("train_return")]
    if tr:
        ax[1].plot(*zip(*tr))
    ax[1].set(xlabel="environment steps", ylabel="return ($/day)", title="Training return (recent mean)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def compare_curves(curves, ref):
    """Mean +/- std validation curves per method over seeds, with reference lines.

    `curves` maps a method name to a list (one per seed) of metric-row lists.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, runs in curves.items():
        n = min(len(r) for r in runs)
        x = np.maximum([r["env_steps"] for r in runs[0][:n]], 300)
        y = np.array([[r["val_pct_of_oracle_saving"] for r in run[:n]] for run in runs])
        ax.plot(x, y.mean(0), label=f"{name} (n={len(runs)})")
        ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.2)
    for name, ls in (("rule_based", ":"), ("no_control", "--")):
        ax.axhline(ref[name], color="gray", ls=ls, lw=1, label=name)
    ax.axhline(100, color="k", lw=0.8, label="oracle (DP)")
    ax.set_xscale("log")
    ax.set(xlabel="environment steps (untrained model drawn at 300)", ylabel="% of oracle saving (val days)",
           title="Learning curves, mean ± std over seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def dispatch(env_cfg, policies, day):
    """Price, load/PV and battery state of charge for one day under each named policy plus the oracle."""
    sc = Scenario(day, DREnv(env_cfg).p)
    h = np.arange(STEPS) / 2
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].step(h, sc.price, where="post", label="price $/kWh")
    events = [i / 2 for i, e in enumerate(sc.event) if e]
    ax[0].scatter(events, [max(sc.price)] * len(events), marker="v", c="r", label="FCAS event")
    ax[0].legend(loc="upper left")
    ax[1].step(h, sc.load, where="post", label="load kW")
    ax[1].step(h, sc.pv, where="post", label="PV kW")
    ax[1].legend()
    for name, pol in {**policies, "oracle": oracle_policy(sc)}.items():
        env = DREnv(env_cfg)
        env.reset(day)
        soc = [env.soc]
        for _ in range(STEPS):
            env.step(pol(env))
            soc.append(env.soc)
        ax[2].plot(np.arange(STEPS + 1) / 2, soc, label=f"{name} SoC kWh")
    ax[2].legend()
    ax[2].set_xlabel("hour of day")
    fig.suptitle(f"Dispatch on day {day}")
    fig.tight_layout()
    return fig


def bench(rows):
    """Updates per second against batch size for each device and network size."""
    sizes = sorted({(r["hidden"], r["layers"]) for r in rows})
    fig, ax = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 3.6), sharey=True)
    for a, (h, ly) in zip(np.atleast_1d(ax), sizes):
        for dev in sorted({r["device"] for r in rows}):
            pts = sorted((r["batch"], r["updates_per_s"]) for r in rows
                         if r["device"] == dev and r["hidden"] == h and r["layers"] == ly)
            a.plot(*zip(*pts), marker="o", label=dev)
        a.set(xscale="log", yscale="log", xlabel="batch size", title=f"hidden {h} x {ly} layers")
        a.grid(alpha=0.3)
    np.atleast_1d(ax)[0].set_ylabel("updates / s")
    np.atleast_1d(ax)[0].legend()
    fig.tight_layout()
    return fig
