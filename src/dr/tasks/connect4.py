"""Connect Four self-play task: win rates against fixed opponents and against the previous checkpoint."""
import hashlib

import numpy as np
import torch

from .. import plotting
from ..agents.alphazero import AZNet, NumpyNet, search
from ..games.connect4 import Connect4Env, OBS_DIM, N_ACTIONS, play, random_player, winblock_player

OPENING_BASE = {"val": 5000, "test": 20000}


class Connect4Task:
    """Two-player zero-sum game; only the search-based agent applies.

    Evaluation plays every fixed opening twice (agent first, agent second) against a random player, a
    win-or-block heuristic and, from the second evaluation on, the policy from the previous evaluation.
    Win rates are percentages of games. The agent's `train_return` is the fraction of decisive self-play
    games, since the only reward is +1 to the winner.
    """

    name = "connect4"
    obs_dim, n_actions = OBS_DIM, N_ACTIONS
    key_metric, key_label = "winrate_winblock", "win % vs win/block heuristic"
    table_metrics = (("winrate_random", "win % vs random", ".1f"), ("winrate_winblock", "win % vs win/block", ".1f"),
                     ("loss_winblock", "loss % vs win/block", ".1f"))
    baseline_order = ("random", "winblock")
    algorithms = ("alphazero",)

    def __init__(self, cfg):
        self.ecfg, self.algo = cfg["eval"], cfg["algo"]

    def make_env(self):
        return Connect4Env()

    def openings(self, split):
        base = OPENING_BASE[split]
        return range(base, base + self.ecfg[f"{split}_games"])

    def data_hash(self, cfg):
        """No dataset: the identity is the game plus the fixed opening seeds."""
        spec = f"connect4|val={list(self.openings('val'))}|test={list(self.openings('test'))}"
        return hashlib.sha256(spec.encode()).hexdigest()[:12]

    def policy_from_weights(self, weights):
        """Deterministic search policy built from saved network weights."""
        net = AZNet(OBS_DIM, self.algo["hidden"], N_ACTIONS)
        net.load_state_dict(weights["net"])
        npnet = NumpyNet(net)
        return lambda env: int(search(env, npnet, self.algo, None, noise=False).argmax())

    def match(self, policy, opponent, split):
        """Win and loss percentages of `policy` over both colours of every opening in `split`."""
        results = []
        for o in self.openings(split):
            results.append(play(policy, opponent, o))
            results.append(-play(opponent, policy, o))
        results = np.array(results)
        return 100 * float((results > 0).mean()), 100 * float((results < 0).mean())

    def evaluate(self, policy, split, prev_policy=None):
        """Win/loss rates against the scripted opponents and, if given, the previous checkpoint's weights."""
        out = {}
        for name, make in (("random", random_player), ("winblock", winblock_player)):
            seed = int(hashlib.sha256(f"{split}-{name}".encode()).hexdigest()[:8], 16)
            out[f"winrate_{name}"], out[f"loss_{name}"] = self.match(policy, make(np.random.default_rng(seed)), split)
        if prev_policy is not None:
            out["winrate_prev"], out["loss_prev"] = self.match(policy, self.policy_from_weights(prev_policy), split)
        return out

    def reference_lines(self):
        """Key-metric values of the scripted players on the validation openings."""
        rng = np.random.default_rng(0)
        return {"random player": self.evaluate(random_player(rng), "val")[self.key_metric], "even (50%)": 50.0}

    def baselines(self):
        """Reference rows for the sweep report: the scripted players scored like an agent on the test split."""
        rng = np.random.default_rng(0)
        return {"random": self.evaluate(random_player(rng), "test"),
                "winblock": self.evaluate(winblock_player(rng), "test")}

    def plots(self, agent, log_figure):
        """Final position of one test game, agent (yellow, moves first) against the win/block heuristic."""
        env, policy = Connect4Env(), agent.policy()
        opponent = winblock_player(np.random.default_rng(0))
        rng = np.random.default_rng(self.openings("test")[0])
        order, n, result = np.zeros_like(env.board, dtype=np.int16), 0, "draw"
        while not env.done:
            if n < 2:
                c = int(rng.choice(np.flatnonzero(env.legal())))
            else:
                c = policy(env) if n % 2 == 0 else opponent(env)
            row = int(env.heights[c])
            _, r, done, _ = env.step(c)
            n += 1
            order[row, c] = n if env.board[row, c] > 0 else -n
            if done and r > 0:
                result = "agent wins" if n % 2 == 1 else "heuristic wins"
        log_figure(plotting.board(order, f"{agent.label} (yellow) vs win/block (red): {result}"), "sample_game.png")
