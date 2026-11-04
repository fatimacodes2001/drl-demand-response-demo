"""Re-evaluate a finished run on the test split and regenerate its task plots.

    python scripts/evaluate.py --run results/dqn/seed0
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from dr.agents import AGENTS
from dr.agents.common import seed_everything
from dr.tasks import make_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    seed_everything(cfg["seed"])
    task = make_task(cfg)
    agent = AGENTS[cfg["algo"]["name"]](cfg, "cpu", cfg["seed"], task)
    agent.load_policy_state_dict(torch.load(run / "final.pt", weights_only=True))

    print(json.dumps(task.evaluate(agent.policy(), "test"), indent=2))

    def save(fig, name):
        path = run / "plots" / name
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)

    task.plots(agent, save)


if __name__ == "__main__":
    main()
