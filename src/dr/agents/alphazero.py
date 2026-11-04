"""AlphaZero-style agent: PUCT tree search guided by a policy/value network.

The loop is the AlphaZero one: (1) self-play episodes where the search picks actions,
(2) train the network on (state -> visit-count policy, return-to-go value), (3) repeat, so
the improved network makes the next search stronger. Here the "opponent" is the stochastic
day (prices, load, events), and the known env dynamics act as the model, so there is no
MuZero-style learned dynamics. Search runs on a numpy copy of the weights (cheap, CPU).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import DREnv, OBS_DIM, N_ACTIONS


class AZNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi, self.v = nn.Linear(hidden, N_ACTIONS), nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


class NumpyNet:
    """Weights snapshot used during search; avoids per-node torch overhead."""

    def __init__(self, net):
        g = lambda m: (m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy())
        self.l1, self.l2, self.pi, self.v = g(net.trunk[0]), g(net.trunk[2]), g(net.pi), g(net.v)

    def __call__(self, obs):
        h = np.maximum(obs @ self.l1[0] + self.l1[1], 0)
        h = np.maximum(h @ self.l2[0] + self.l2[1], 0)
        return h @ self.pi[0] + self.pi[1], float((h @ self.v[0] + self.v[1])[0])


class Node:
    """Search-tree node holding an env state and per-action visit statistics."""

    __slots__ = ("env", "done", "P", "N", "W", "R", "children", "v", "legal")

    def __init__(self, env, done=False):
        self.env, self.done = env, done
        self.N = np.zeros(N_ACTIONS); self.W = np.zeros(N_ACTIONS); self.R = np.zeros(N_ACTIONS)
        self.children = [None] * N_ACTIONS


def expand(node, net, vscale):
    """Evaluate a node with the network, set masked priors and return its value in dollars."""
    logits, v = net(node.env.obs())
    node.legal = np.array(node.env.legal())
    z = np.where(node.legal, logits, -np.inf)
    p = np.exp(z - z.max())
    node.P = p / p.sum()
    node.v = v * vscale
    return node.v


def search(env, net, a, rng, noise):
    """Run PUCT simulations from `env` and return the root visit counts."""
    vs, g, c = a["value_scale"], a["gamma"], a["c_puct"]
    root = Node(env.clone())
    expand(root, net, vs)
    if noise:
        d = np.zeros(N_ACTIONS)
        d[root.legal] = rng.dirichlet([a["dirichlet_alpha"]] * int(root.legal.sum()))
        root.P = (1 - a["dirichlet_frac"]) * root.P + a["dirichlet_frac"] * d
    for _ in range(a["sims"]):
        node, path = root, []
        while True:
            tot = node.N.sum()
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), node.v)
            score = q / vs + c * node.P * np.sqrt(tot + 1) / (1 + node.N)
            act = int(np.where(node.legal, score, -np.inf).argmax())
            path.append((node, act))
            child = node.children[act]
            if child is None:
                cenv = node.env.clone()
                _, r, done, _ = cenv.step(act)
                node.R[act] = r
                child = node.children[act] = Node(cenv, done)
                G = 0.0 if done else expand(child, net, vs)
                break
            if child.done:
                G = 0.0
                break
            node = child
        for nd, ac in reversed(path):
            G = nd.R[ac] + g * G
            nd.N[ac] += 1
            nd.W[ac] += G
    return root.N


class AlphaZeroAgent:
    """Self-play search plus network training loop."""

    def __init__(self, cfg, device, seed):
        self.a, self.dev, self.env_cfg = cfg["algo"], device, cfg["env"]
        self.n_days = cfg["train"]["n_train_days"]
        self.rng = np.random.default_rng(seed)
        self.net = AZNet(self.a["hidden"]).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.a["lr"])
        cap = self.a["buffer"]
        self.B = dict(obs=np.zeros((cap, OBS_DIM), np.float32), pi=np.zeros((cap, N_ACTIONS), np.float32),
                      z=np.zeros(cap, np.float32))
        self.n, self.i, self.env_steps, self.iters, self.sims_total, self.recent = 0, 0, 0, 0, 0, []

    def policy(self):
        """Search-based policy `act(env) -> int` using the current network and no exploration noise."""
        net = NumpyNet(self.net)
        return lambda env: int(search(env, net, self.a, self.rng, noise=False).argmax())

    def self_play(self, net):
        """Play one training day choosing actions from search visit counts; store (state, pi, return)."""
        env = DREnv(self.env_cfg)
        env.reset(int(self.rng.integers(self.n_days)))
        obs_l, pi_l, rew_l, done = [], [], [], False
        while not done:
            visits = search(env, net, self.a, self.rng, noise=True)
            pi = visits / visits.sum()
            w = visits ** (1.0 / self.a["temperature"])
            act = int(self.rng.choice(N_ACTIONS, p=w / w.sum()))
            obs_l.append(env.obs()); pi_l.append(pi)
            _, r, done, _ = env.step(act)
            rew_l.append(r)
            self.sims_total += self.a["sims"]
        G, z = 0.0, []
        for r in reversed(rew_l):
            G = r + self.a["gamma"] * G
            z.append(G / self.a["value_scale"])
        for o, p, v in zip(obs_l, pi_l, reversed(z)):
            self.B["obs"][self.i], self.B["pi"][self.i], self.B["z"][self.i] = o, p, v
            self.i, self.n = (self.i + 1) % len(self.B["z"]), min(self.n + 1, len(self.B["z"]))
        self.env_steps += len(rew_l)
        return sum(rew_l)

    def train_chunk(self):
        """One AlphaZero iteration: self-play episodes, then gradient steps on the replay."""
        net = NumpyNet(self.net)
        rets = [self.self_play(net) for _ in range(self.a["episodes_per_iter"])]
        self.recent = (self.recent + rets)[-20:]
        for _ in range(self.a["train_steps"]):
            idx = self.rng.integers(0, self.n, self.a["batch"])
            o, p, z = (torch.as_tensor(self.B[k][idx], device=self.dev) for k in ("obs", "pi", "z"))
            logits, v = self.net(o)
            pl = -(p * F.log_softmax(logits, -1)).sum(-1).mean()
            vl = F.mse_loss(v, z)
            loss = pl + self.a["value_coef"] * vl
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.iters += 1
        return {"train_return": float(np.mean(self.recent)), "policy_loss": pl.item(),
                "value_loss": vl.item(), "sims_total": self.sims_total}

    def state_dict(self):
        return dict(net=self.net.state_dict(), opt=self.opt.state_dict(), B=self.B, n=self.n, i=self.i,
                    env_steps=self.env_steps, iters=self.iters, sims_total=self.sims_total,
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, s):
        self.net.load_state_dict(s["net"]); self.opt.load_state_dict(s["opt"])
        self.B, self.n, self.i, self.env_steps = s["B"], s["n"], s["i"], s["env_steps"]
        self.iters, self.sims_total = s["iters"], s["sims_total"]
        self.rng.bit_generator.state = s["rng"]
