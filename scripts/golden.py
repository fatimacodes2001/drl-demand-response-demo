"""Record tiny-run results as golden values for tests/test_golden.py.

    python scripts/golden.py

Run this after any change under src/dr that is meant to alter results. The test suite skips the golden
comparison when the recorded code hash no longer matches, so forgetting is loud rather than silent.
"""
import json
import tempfile
from pathlib import Path

from dr import config, runner
from dr.fingerprint import fingerprint

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["train.total_steps=1500", "eval.every_steps=750", "eval.val_days=3", "eval.test_days=3",
         "eval.val_games=2", "eval.test_games=2", "device=cpu", "tracker=none", "algo.sims=8",
         "algo.train_steps=5", "algo.learning_starts=200"]
ALGOS = ["dqn", "ppo", "ddpg", "per_ddpg", "alphazero", "connect4_alphazero"]


def tiny_config(name):
    """The shipped config shrunk to a few seconds of CPU training."""
    return config.load(ROOT / f"configs/{name}.yaml", SMALL)


def main():
    fp = fingerprint(tiny_config("dqn"), "cpu")
    golden = {"_recorded_on": {"code_hash": fp["code_hash"], "libs": fp["libs"], "platform": fp["platform"]}}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ALGOS:
            res = runner.run(tiny_config(name), Path(tmp) / name)
            golden[name] = {k: v for k, v in res.items() if isinstance(v, float) and k != "train_wall_s"}
    (ROOT / "tests/golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote tests/golden.json for {golden['_recorded_on']}")


if __name__ == "__main__":
    main()
