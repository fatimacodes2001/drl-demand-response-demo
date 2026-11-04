"""Run configs x seeds in parallel processes, then build the report.

    python scripts/sweep.py --configs configs/dqn.yaml configs/ppo.yaml --seeds 0 1 2 3 4 --jobs 4

Finished runs with an identical fingerprint are skipped, so re-running is cheap and safe.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dr import config, tracking

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    names = [config.load(c)["name"] for c in a.configs]
    if len(set(names)) != len(names):
        sys.exit(f"config names must be unique, got {names}; runs would overwrite each other")

    # create the MLflow database once so parallel runs do not race to initialise it
    tracking.init_store()

    def job(cs):
        cfg, seed = cs
        cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--config", cfg, "--seed", str(seed), "--out", a.out]
        cmd += ["--device", a.device] if a.device else []
        cmd += ["--set", *a.set] if a.set else []
        cmd += ["--force"] if a.force else []
        return subprocess.run(cmd).returncode

    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(job, [(c, s) for c in a.configs for s in a.seeds]))
    if any(codes):
        sys.exit(f"{sum(map(bool, codes))} run(s) failed")
    subprocess.run([sys.executable, str(ROOT / "scripts/report.py"), "--out", a.out], check=True)


if __name__ == "__main__":
    main()
