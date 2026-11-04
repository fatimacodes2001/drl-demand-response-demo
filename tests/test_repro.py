"""Reproducibility, resume, fingerprint and oracle-bound tests."""
import json
from pathlib import Path

import pytest

from dr import config, evaluate as ev, runner
from dr.env import DREnv, Scenario, oracle
from dr.fingerprint import fingerprint

ROOT = Path(__file__).resolve().parents[1]

SMALL = ["train.total_steps=1500", "eval.every_steps=750", "eval.val_days=3", "eval.test_days=3",
         "device=cpu", "tracker=none", "algo.sims=8", "algo.train_steps=5", "algo.learning_starts=200"]


def cfg_for(name, seed=0):
    """A tiny, CPU-only, untracked version of a shipped config."""
    c = config.load(ROOT / f"configs/{name}.yaml", SMALL)
    c["seed"] = seed
    return c


def test_oracle_upper_bounds_every_policy():
    """Oracle upper bounds every policy."""
    days = ev.days_for("test", 10)
    opt = ev.oracle_returns(days, {})
    for pol in (ev.no_control, ev.rule_based, ev.random_policy(1)):
        assert (ev.evaluate(pol, days, {}) <= opt + 1e-9).all()


def test_env_is_deterministic_and_clone_independent():
    """Env is deterministic and clone independent."""
    e = DREnv(); e.reset(7)
    c = e.clone(); c.step(0)
    assert e.t == 0 and c.t == 1
    assert oracle(Scenario(7, e.p)) == oracle(Scenario(7, e.p))


@pytest.mark.parametrize("algo", ["dqn", "ppo", "alphazero", "ddpg", "per_ddpg"])
def test_same_seed_same_metrics(algo, tmp_path):
    """Same seed same metrics."""
    r1 = runner.run(cfg_for(algo), tmp_path / "a")
    r2 = runner.run(cfg_for(algo), tmp_path / "b")
    m = lambda d: [{k: v for k, v in json.loads(l).items() if k != "wall_s"} for l in (d / "metrics.jsonl").read_text().splitlines()]
    assert m(tmp_path / "a") == m(tmp_path / "b")
    assert r1["return_mean"] == r2["return_mean"]


def test_different_seed_differs(tmp_path):
    """Different seed differs."""
    a = runner.run(cfg_for("dqn", 0), tmp_path / "a")
    b = runner.run(cfg_for("dqn", 1), tmp_path / "b")
    assert a["return_mean"] != b["return_mean"]


def test_finished_run_is_skipped_and_resume_continues(tmp_path):
    """Finished run is skipped and resume continues."""
    c = cfg_for("dqn")
    runner.run(c, tmp_path / "r")
    t = (tmp_path / "r" / "final.json").stat().st_mtime_ns
    # identical fingerprint: skipped
    runner.run(c, tmp_path / "r")
    assert (tmp_path / "r" / "final.json").stat().st_mtime_ns == t
    c2 = cfg_for("dqn"); c2["train"]["total_steps"] = 2500
    # larger budget plus resume: continues from the checkpoint
    runner.run(c2, tmp_path / "r", resume=True)
    rows = [json.loads(l) for l in (tmp_path / "r" / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["env_steps"] >= 2500 and rows[0]["env_steps"] == 0 and len(rows) > 3


def test_fingerprint_tracks_config_code_and_data():
    """Fingerprint tracks config code and data."""
    base = fingerprint(cfg_for("dqn"), "cpu")
    assert base == fingerprint(cfg_for("dqn"), "cpu")
    assert base["fingerprint"] != fingerprint(cfg_for("dqn", 1), "cpu")["fingerprint"]
    c = cfg_for("dqn"); c["env"]["fcas_price"] = 1.5
    other = fingerprint(c, "cpu")
    assert other["data_hash"] != base["data_hash"] and other["fingerprint"] != base["fingerprint"]
    assert fingerprint(cfg_for("dqn"), "cuda")["fingerprint"] != base["fingerprint"]
