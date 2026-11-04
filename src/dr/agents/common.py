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


class PrioritizedReplay(Replay):
    """Proportional prioritised replay (Schaul et al. 2016) with importance-sampling weights."""

    def __init__(self, cap, obs_dim, extra=(), alpha=0.6):
        super().__init__(cap, obs_dim, extra)
        self.alpha, self.pri, self.max_pri = alpha, np.zeros(cap, np.float64), 1.0

    def add(self, **kw):
        # new transitions get the current maximum priority so each is replayed at least once
        self.pri[self.i] = self.max_pri
        super().add(**kw)

    def sample(self, rng, batch, device, beta=0.4):
        p = self.pri[:self.n] ** self.alpha
        p /= p.sum()
        idx = rng.choice(self.n, batch, p=p)
        w = (self.n * p[idx]) ** -beta
        out = {k: torch.as_tensor(v[idx], device=device) for k, v in self.data.items()}
        out["w"] = torch.as_tensor(w / w.max(), dtype=torch.float32, device=device)
        out["idx"] = idx
        return out

    def update(self, idx, td_abs):
        """Set the priorities of sampled transitions to their absolute TD error."""
        self.pri[idx] = td_abs + 1e-6
        self.max_pri = max(self.max_pri, float(td_abs.max()))

    def state(self):
        return {**super().state(), "pri": self.pri[:self.n].copy(), "max_pri": self.max_pri}

    def load(self, s):
        super().load(s)
        self.pri[:len(s["pri"])], self.max_pri = s["pri"], s["max_pri"]
