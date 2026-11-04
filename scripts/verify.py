"""Re-run a recorded run from its saved config and check that it reproduces.

    python scripts/verify.py --run results/dqn/seed0 [--tol 0]

Exit code 0 means the learning-curve metrics and test return match: exactly on CPU, and within `--tol`
(relative) when comparing across different GPUs.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

from dr import runner


def _rows(d):
    """Metric rows of a run without the wall-clock field."""
    return [{k: v for k, v in json.loads(line).items() if k != "wall_s"}
            for line in (d / "metrics.jsonl").read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--tol", type=float, default=0.0)
    a = p.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    cfg["tracker"] = "none"
    want = json.loads((run / "final.json").read_text())
    close = lambda x, y: x == y if a.tol == 0 else abs(x - y) <= a.tol * max(1.0, abs(y))

    with tempfile.TemporaryDirectory() as tmp:
        got = runner.run(cfg, Path(tmp) / "rerun")
        same_fp = got["fingerprint"] == want["fingerprint"]
        if not same_fp:
            print(f"fingerprint differs: recorded {want['fingerprint']}, now {got['fingerprint']} "
                  "(config, code, data, libraries or device changed since the original run)")
        old, new = _rows(run), _rows(Path(tmp) / "rerun")
        same_curve = len(old) == len(new) and all(
            close(x[k], y[k]) for x, y in zip(old, new) for k in x if isinstance(x[k], (int, float)))
        same_test = close(got["return_mean"], want["return_mean"])
    print(f"fingerprint {'MATCH' if same_fp else 'DIFF'} | learning curve {'MATCH' if same_curve else 'DIFF'} | "
          f"test return {got['return_mean']:.6f} vs {want['return_mean']:.6f} {'MATCH' if same_test else 'DIFF'}")
    sys.exit(0 if same_curve and same_test else 1)


if __name__ == "__main__":
    main()
