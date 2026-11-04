"""AlphaZero-style agent: PUCT tree search guided by a policy/value network.

The loop is the AlphaZero one: (1) self-play episodes where the search picks actions,
(2) train the network on (state -> visit-count policy, return-to-go value), (3) repeat, so
the improved network makes the next search stronger. The env is the model (no MuZero-style
learned dynamics) and search runs on a numpy copy of the weights.

Envs with `two_player = True` (Connect Four) report rewards for the player who just moved, and
values flip sign at every ply during backup and when computing targets. Single-player envs
(demand response) accumulate discounted rewards as usual.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseAgent


class AZNet(nn.Module):
    """Two-layer trunk with policy-logit and scalar value heads."""

    def __init__(self, obs_dim, hidden, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi, self.v = nn.Linear(hidden, n_actions), nn.Linear(hidden, 1)

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
        k = len(env.legal())
        self.N, self.W, self.R = np.zeros(k), np.zeros(k), np.zeros(k)
        self.children = [None] * k


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
    # in a two-player game the child's value is the opponent's, so it enters the parent negated
    sign = -1.0 if getattr(env, "two_player", False) else 1.0
    root = Node(env.clone())
    expand(root, net, vs)
    if noise:
        d = np.zeros(len(root.P))
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
            G = nd.R[ac] + g * sign * G
            nd.N[ac] += 1
            nd.W[ac] += G
    return root.N


class AlphaZeroAgent(BaseAgent):
    """Self-play search plus network training loop."""

    nets = ("net",)
    opts = ("opt",)

    def __init__(self, cfg, device, seed, task):
        super().__init__(cfg, device, seed, task)
        self.net = AZNet(task.obs_dim, self.a["hidden"], task.n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.a["lr"])
        cap = self.a["buffer"]
        self.B = dict(obs=np.zeros((cap, task.obs_dim), np.float32), pi=np.zeros((cap, task.n_actions), np.float32),
                      z=np.zeros(cap, np.float32))
        self.n, self.i, self.iters, self.sims_total = 0, 0, 0, 0

    def policy(self):
        """Search-based policy `act(env) -> int` using the current network and no exploration noise."""
        net = NumpyNet(self.net)
        return lambda env: int(search(env, net, self.a, self.rng, noise=False).argmax())

    def self_play(self, net):
        """Play one training day choosing actions from search visit counts; store (state, pi, return)."""
        env = self.task.make_env()
        env.reset(self.sample_day())
        sign = -1.0 if getattr(env, "two_player", False) else 1.0
        obs_l, pi_l, rew_l, done = [], [], [], False
        while not done:
            visits = search(env, net, self.a, self.rng, noise=True)
            pi = visits / visits.sum()
            w = visits ** (1.0 / self.a["temperature"])
            act = int(self.rng.choice(len(w), p=w / w.sum()))
            obs_l.append(env.obs())
            pi_l.append(pi)
            _, r, done, _ = env.step(act)
            rew_l.append(r)
            self.sims_total += self.a["sims"]
        G, z = 0.0, []
        for r in reversed(rew_l):
            G = r + self.a["gamma"] * sign * G
            z.append(G / self.a["value_scale"])
        for o, p, v in zip(obs_l, pi_l, reversed(z)):
            self.B["obs"][self.i], self.B["pi"][self.i], self.B["z"][self.i] = o, p, v
            self.i, self.n = (self.i + 1) % len(self.B["z"]), min(self.n + 1, len(self.B["z"]))
        self.env_steps += len(rew_l)
        return sum(rew_l)

    def train_chunk(self):
        """One AlphaZero iteration: self-play episodes, then gradient steps on the replay."""
        net = NumpyNet(self.net)
        for _ in range(self.a["episodes_per_iter"]):
            self.record(self.self_play(net))
        for _ in range(self.a["train_steps"]):
            idx = self.rng.integers(0, self.n, self.a["batch"])
            o, p, z = (torch.as_tensor(self.B[k][idx], device=self.dev) for k in ("obs", "pi", "z"))
            logits, v = self.net(o)
            pl = -(p * F.log_softmax(logits, -1)).sum(-1).mean()
            vl = F.mse_loss(v, z)
            loss = pl + self.a["value_coef"] * vl
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.iters += 1
        return {"train_return": self.train_return(), "policy_loss": pl.item(), "value_loss": vl.item(),
                "sims_total": self.sims_total}

    def extra_state(self):
        return {"B": {k: v[:self.n].copy() for k, v in self.B.items()}, "n": self.n, "i": self.i,
                "iters": self.iters, "sims_total": self.sims_total}

    def load_extra_state(self, s):
        for k, v in s["B"].items():
            self.B[k][:len(v)] = v
        self.n, self.i, self.iters, self.sims_total = s["n"], s["i"], s["iters"], s["sims_total"]
