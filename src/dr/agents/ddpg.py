"""DDPG (continuous action in [-1, 1]) with optional prioritised replay (PER-DDPG)."""
import copy

import numpy as np
import torch
import torch.nn as nn

from .common import BaseAgent, mlp, Replay, PrioritizedReplay


class Actor(nn.Module):
    """Deterministic policy with a tanh output in [-1, 1]."""

    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden)

    def forward(self, o):
        return torch.tanh(self.net(o))


class DDPGAgent(BaseAgent):
    """DDPG, or PER-DDPG when `algo.per` is true."""

    nets = ("actor", "critic", "t_actor", "t_critic")
    opts = ("aopt", "copt")

    def __init__(self, cfg, device, seed, task):
        super().__init__(cfg, device, seed, task)
        a, obs_dim = self.a, task.obs_dim
        self.actor, self.critic = Actor(obs_dim, a["hidden"]).to(device), mlp(obs_dim + 1, 1, a["hidden"]).to(device)
        self.t_actor, self.t_critic = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=a["actor_lr"])
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=a["critic_lr"])
        extra = [("next_obs", (obs_dim,)), ("act", (1,)), ("rew", ()), ("done", ())]
        self.per = a.get("per", False)
        self.buf = (PrioritizedReplay(a["buffer"], obs_dim, extra, a["per_alpha"]) if self.per
                    else Replay(a["buffer"], obs_dim, extra))

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
        self.copt.zero_grad()
        closs.backward()
        self.copt.step()
        aloss = -self.critic(torch.cat([b["obs"], self.actor(b["obs"])], 1)).mean()
        self.aopt.zero_grad()
        aloss.backward()
        self.aopt.step()
        with torch.no_grad():
            for t, s in ((self.t_actor, self.actor), (self.t_critic, self.critic)):
                for pt, ps in zip(t.parameters(), s.parameters()):
                    pt.mul_(1 - a["tau"]).add_(a["tau"] * ps)
        if self.per:
            self.buf.update(b["idx"], td.detach().abs().cpu().numpy())
        return closs.item()

    def train_chunk(self):
        """One episode of noisy interaction with an update per env step."""
        env = self.task.make_env()
        obs, done, ret, loss = env.reset(self.sample_day()), False, 0.0, 0.0
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
        self.record(ret)
        return {"train_return": self.train_return(), "sigma": self.sigma(), "critic_loss": loss}

    def extra_state(self):
        return {"buf": self.buf.state()}

    def load_extra_state(self, s):
        self.buf.load(s["buf"])
