"""Re-evaluate a finished run on the test days and plot a day's dispatch against the oracle.

    python scripts/evaluate.py --run results/dqn/seed0 --day 20003
"""
import argparse
from pathlib import Path

import torch
import yaml

from dr import evaluate as ev
from dr import plotting
from dr.agents import AGENTS
from dr.agents.common import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--day", type=int, default=ev.TEST_BASE)
    a = p.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    seed_everything(cfg["seed"])
    agent = AGENTS[cfg["algo"]["name"]](cfg, "cpu", cfg["seed"])
    agent.load_policy_state_dict(torch.load(run / "final.pt", weights_only=True))
    pol = agent.policy()

    days = ev.days_for("test", cfg["eval"]["test_days"])
    print(ev.summarise(ev.evaluate(pol, days, cfg["env"]), days, cfg["env"]))
    path = run / "plots" / f"dispatch_day{a.day}.png"
    path.parent.mkdir(exist_ok=True)
    plotting.dispatch(cfg["env"], {cfg["name"]: pol}, a.day).savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
