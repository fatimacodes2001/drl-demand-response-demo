"""DDPG (continuous action in [-1, 1]) with optional prioritised replay (PER-DDPG)."""
import copy

import numpy as np
import torch
import torch.nn as nn

from ..env import DREnv, OBS_DIM
from .common import mlp, Replay, PrioritizedReplay


class Actor(nn.Module):
    """Deterministic policy with a tanh output in [-1, 1]."""

    def __init__(self, hidden):
        super().__init__()
        self.net = mlp(OBS_DIM, 1, hidden)

    def forward(self, o):
        return torch.tanh(self.net(o))


class DDPGAgent:
    """DDPG, or PER-DDPG when `algo.per` is true."""

    def __init__(self, cfg, device, seed):
        self.a, self.dev, self.env_cfg = cfg["algo"], device, cfg["env"]
        a = self.a
        self.n_days = cfg["train"]["n_train_days"]
        self.total_steps = cfg["train"]["total_steps"]
        self.rng = np.random.default_rng(seed)
        self.actor, self.critic = Actor(a["hidden"]).to(device), mlp(OBS_DIM + 1, 1, a["hidden"]).to(device)
        self.t_actor, self.t_critic = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=a["actor_lr"])
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=a["critic_lr"])
        extra = [("next_obs", (OBS_DIM,)), ("act", (1,)), ("rew", ()), ("done", ())]
        self.per = a.get("per", False)
        self.buf = (PrioritizedReplay(a["buffer"], OBS_DIM, extra, a["per_alpha"]) if self.per
                    else Replay(a["buffer"], OBS_DIM, extra))
        self.env_steps, self.recent = 0, []

    def sigma(self):
        """Current std of the Gaussian exploration noise."""
        a = self.a
        frac = min(self.env_steps / a["noise_decay_steps"], 1.0)
        return a["noise_start"] + frac * (a["noise_end"] - a["noise_start"])

    def act(self, obs):
        """Deterministic continuous action for one observation."""
        with torch.no_grad():
            return float(self.actor(torch.as_tensor(obs, device=self.dev)))

    def policy(self):
        """Deterministic policy `act(env) -> float`."""
        return lambda env: self.act(env.obs())

    def update(self):
        """One critic and actor step (importance-weighted when PER is on), then soft target updates."""
        a = self.a
        if self.per:
            beta = a["per_beta_start"] + min(self.env_steps / self.total_steps, 1.0) * (1 - a["per_beta_start"])
            b = self.buf.sample(self.rng, a["batch"], self.dev, beta)
        else:
            b = self.buf.sample(self.rng, a["batch"], self.dev)
        w = b.get("w", 1.0)
        with torch.no_grad():
            nq = self.t_critic(torch.cat([b["next_obs"], self.t_actor(b["next_obs"])], 1)).squeeze(1)
            y = b["rew"] * a["reward_scale"] + a["gamma"] * (1 - b["done"]) * nq
        q = self.critic(torch.cat([b["obs"], b["act"]], 1)).squeeze(1)
        td = y - q
        closs = (w * td ** 2).mean()
        self.copt.zero_grad(); closs.backward(); self.copt.step()
        aloss = -self.critic(torch.cat([b["obs"], self.actor(b["obs"])], 1)).mean()
        self.aopt.zero_grad(); aloss.backward(); self.aopt.step()
        with torch.no_grad():
            for t, s in ((self.t_actor, self.actor), (self.t_critic, self.critic)):
                for pt, ps in zip(t.parameters(), s.parameters()):
                    pt.mul_(1 - a["tau"]).add_(a["tau"] * ps)
        if self.per:
            self.buf.update(b["idx"], td.detach().abs().cpu().numpy())
        return closs.item()

    def train_chunk(self):
        """One episode of noisy interaction with an update per env step."""
        env = DREnv(self.env_cfg)
        obs, done, ret, loss = env.reset(int(self.rng.integers(self.n_days))), False, 0.0, 0.0
        while not done:
            if len(self.buf) < self.a["learning_starts"]:
                act = float(self.rng.uniform(-1, 1))
            else:
                act = float(np.clip(self.act(obs) + self.rng.normal(0, self.sigma()), -1, 1))
            nobs, r, done, _ = env.step(act)
            self.buf.add(obs=obs, next_obs=nobs, act=[act], rew=r, done=float(done))
            obs, ret = nobs, ret + r
            self.env_steps += 1
            if len(self.buf) >= self.a["learning_starts"]:
                loss = self.update()
        self.recent = (self.recent + [ret])[-20:]
        return {"train_return": float(np.mean(self.recent)), "sigma": self.sigma(), "critic_loss": loss}

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    t_actor=self.t_actor.state_dict(), t_critic=self.t_critic.state_dict(),
                    aopt=self.aopt.state_dict(), copt=self.copt.state_dict(), buf=self.buf.state(),
                    env_steps=self.env_steps, rng=self.rng.bit_generator.state)

    def load_state_dict(self, s):
        for k in ("actor", "critic", "t_actor", "t_critic"):
            getattr(self, k).load_state_dict(s[k])
        self.aopt.load_state_dict(s["aopt"]); self.copt.load_state_dict(s["copt"])
        self.buf.load(s["buf"]); self.env_steps = s["env_steps"]
        self.rng.bit_generator.state = s["rng"]
