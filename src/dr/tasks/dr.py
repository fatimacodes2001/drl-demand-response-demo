"""Demand-response task: fixed day splits, DP oracle reference and dispatch plots."""
import hashlib
import json

import numpy as np

from .. import evaluate as ev
from .. import plotting
from ..env import DREnv, Scenario, OBS_DIM, N_ACTIONS


class DemandResponseTask:
    """Single-agent battery dispatch scored as % of the dynamic-programming oracle's saving."""

    name = "dr"
    obs_dim, n_actions = OBS_DIM, N_ACTIONS
    key_metric, key_label = "pct_of_oracle_saving", "% of oracle saving"
    table_metrics = (("saving_per_day", "test saving $/day", ".2f"), ("pct_of_oracle_saving", "% of oracle saving", ".1f"))
    baseline_order = ("no_control", "random", "rule_based", "oracle_discrete")
    algorithms = ("dqn", "ppo", "ddpg", "alphazero")

    def __init__(self, cfg):
        self.env_cfg, self.ecfg = cfg["env"], cfg["eval"]

    def make_env(self):
        return DREnv(self.env_cfg)

    def days(self, split):
        return ev.days_for(split, self.ecfg[f"{split}_days"])

    def data_hash(self, cfg):
        """Hash of every generated day a run can touch plus the env parameters.

        With real data this is where a dataset checksum (for example a DVC hash) would go.
        """
        p = DREnv(self.env_cfg).p
        days = [*range(cfg["train"]["n_train_days"]), *self.days("val"), *self.days("test")]
        h = hashlib.sha256(json.dumps(p, sort_keys=True).encode())
        for d in days:
            s = Scenario(d, p)
            for arr in (s.load, s.pv, s.price, s.event):
                h.update(np.asarray(arr, np.float64).tobytes())
        return h.hexdigest()[:12]

    def evaluate(self, policy, split, prev_policy=None):
        """Headline metrics of `policy` on the fixed days of `split`."""
        days = self.days(split)
        return ev.summarise(ev.evaluate(policy, days, self.env_cfg), days, self.env_cfg)

    def reference_lines(self):
        """Key-metric values of simple references on the validation days, for learning-curve plots."""
        return {"rule_based": self.evaluate(ev.rule_based, "val")[self.key_metric], "no_control": 0.0}

    def baselines(self):
        """Reference rows for the sweep report, computed on the test days."""
        return ev.baselines(self.days("test"), self.env_cfg)

    def plots(self, agent, log_figure):
        """Battery schedule of the trained policy next to the oracle on the first test day."""
        day = self.days("test")[0]
        log_figure(plotting.dispatch(self.env_cfg, {agent.label: agent.policy()}, day), f"dispatch_day{day}.png")
