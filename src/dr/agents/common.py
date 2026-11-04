"""Shared helpers: seeding, MLPs and replay buffers."""
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed):
    """Seed python, numpy and torch, pin one CPU thread and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # a single thread gives a fixed reduction order, so CPU runs are bitwise repeatable
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name):
    """Resolve 'auto' to cuda when available, otherwise return `name` unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def mlp(i, o, hidden, layers=2):
    """ReLU MLP with `layers` hidden layers of width `hidden`."""
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, o))


class Replay:
    """Fixed-size ring buffer of transitions stored as numpy arrays."""

    def __init__(self, cap, obs_dim, extra=()):
        self.cap, self.n, self.i = cap, 0, 0
        self.data = {"obs": np.zeros((cap, obs_dim), np.float32)}
        for name, shape in extra:
            self.data[name] = np.zeros((cap, *shape), np.float32)

    def __len__(self):
        return self.n

    def add(self, **kw):
        """Insert one transition given as keyword fields."""
        for k, v in kw.items():
            self.data[k][self.i] = v
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, rng, batch, device):
        """Uniformly sample a batch as tensors on `device`."""
        idx = rng.integers(0, self.n, batch)
        return {k: torch.as_tensor(v[idx], device=device) for k, v in self.data.items()}

    def state(self):
        """Serialisable snapshot for checkpoints."""
        return {"data": {k: v[:self.n].copy() for k, v in self.data.items()}, "n": self.n, "i": self.i}

    def load(self, s):
        """Restore a snapshot made by `state`."""
        for k, v in s["data"].items():
            self.data[k][:len(v)] = v
        self.n, self.i = s["n"], s["i"]


class SumTree:
    """Binary tree of partial sums giving O(log n) priority updates and prefix-sum lookups."""

    def __init__(self, cap):
        self.size = 1 << max(cap - 1, 1).bit_length()
        self.tree = np.zeros(2 * self.size)

    @property
    def total(self):
        return self.tree[1]

    def set(self, idx, values):
        """Set leaf values at `idx` and refresh their ancestors."""
        pos = np.asarray(idx) + self.size
        self.tree[pos] = values
        pos = np.unique(pos >> 1)
        while pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            if pos[0] == 1:
                break
            pos = np.unique(pos >> 1)

    def find(self, prefix):
        """Leaf index whose cumulative-sum interval contains each value in `prefix`."""
        v, node = np.asarray(prefix, np.float64).copy(), np.ones(len(prefix), np.int64)
        while node[0] < self.size:
            left = 2 * node
            go_right = v > self.tree[left]
            v -= np.where(go_right, self.tree[left], 0.0)
            node = left + go_right
        return node - self.size

    def leaf(self, idx):
        return self.tree[np.asarray(idx) + self.size]


class PrioritizedReplay(Replay):
    """Proportional prioritised replay (Schaul et al. 2016) with stratified sampling and IS weights."""

    def __init__(self, cap, obs_dim, extra=(), alpha=0.6):
        super().__init__(cap, obs_dim, extra)
        self.alpha, self.tree, self.max_pri = alpha, SumTree(cap), 1.0
        self.pri = np.zeros(cap, np.float64)

    def _set(self, idx, pri):
        self.pri[idx] = pri
        self.tree.set(idx, np.asarray(pri) ** self.alpha)

    def add(self, **kw):
        # new transitions get the current maximum priority so each is replayed at least once
        self._set(np.array([self.i]), self.max_pri)
        super().add(**kw)

    def sample(self, rng, batch, device, beta=0.4):
        """Stratified sample of `batch` transitions plus importance weights and indices."""
        seg = self.tree.total / batch
        idx = np.minimum(self.tree.find((np.arange(batch) + rng.random(batch)) * seg), self.n - 1)
        p = self.tree.leaf(idx) / self.tree.total
        w = (self.n * p) ** -beta
        out = {k: torch.as_tensor(v[idx], device=device) for k, v in self.data.items()}
        out["w"] = torch.as_tensor(w / w.max(), dtype=torch.float32, device=device)
        out["idx"] = idx
        return out

    def update(self, idx, td_abs):
        """Set the priorities of sampled transitions to their absolute TD error."""
        self._set(idx, td_abs + 1e-6)
        self.max_pri = max(self.max_pri, float(td_abs.max()))

    def state(self):
        return {**super().state(), "pri": self.pri[:self.n].copy(), "max_pri": self.max_pri}

    def load(self, s):
        super().load(s)
        self.max_pri = s["max_pri"]
        self._set(np.arange(len(s["pri"])), s["pri"])
