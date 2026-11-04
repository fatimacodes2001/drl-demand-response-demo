"""Run one experiment: python scripts/train.py --config configs/dqn.yaml --seed 0 [--set algo.lr=1e-3]."""
import argparse
from pathlib import Path

from dr import config, runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | cuda | auto")
    p.add_argument("--set", nargs="*", default=[], help="overrides such as algo.lr=1e-3")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if finished with an identical fingerprint")
    a = p.parse_args()

    cfg = config.load(a.config, a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.device:
        cfg["device"] = a.device
    runner.run(cfg, Path(a.out) / cfg["name"] / f"seed{cfg['seed']}", resume=a.resume, force=a.force)


if __name__ == "__main__":
    main()
