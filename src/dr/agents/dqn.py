"""Double DQN with a target network and uniform replay (value-based baseline)."""
import copy

import torch
import torch.nn.functional as F

from ..env import DREnv, OBS_DIM, N_ACTIONS
from .common import BaseAgent, mlp, Replay


class DQNAgent(BaseAgent):
    """Double DQN over the five discrete battery actions."""

    nets = ("q", "tq")
    opts = ("opt",)

    def __init__(self, cfg, device, seed):
        super().__init__(cfg, device, seed)
        self.q = mlp(OBS_DIM, N_ACTIONS, self.a["hidden"]).to(device)
        self.tq = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.a["lr"])
        self.buf = Replay(self.a["buffer"], OBS_DIM,
                          [("next_obs", (OBS_DIM,)), ("act", ()), ("rew", ()), ("done", ())])
        self.updates = 0

    def eps(self):
        """Current epsilon of the linear exploration schedule."""
        a = self.a
        frac = min(self.env_steps / a["eps_decay_steps"], 1.0)
        return a["eps_start"] + frac * (a["eps_end"] - a["eps_start"])

    def greedy(self, obs):
        """Argmax-Q action for one observation."""
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs, device=self.dev)).argmax())

    def policy(self):
        """Greedy policy `act(env) -> int`."""
        return lambda env: self.greedy(env.obs())

    def update(self):
        """One double-DQN gradient step on a replay batch; returns the loss."""
        a = self.a
        b = self.buf.sample(self.rng, a["batch"], self.dev)
        with torch.no_grad():
            # double DQN: the online net picks the action, the target net scores it
            best = self.q(b["next_obs"]).argmax(1, keepdim=True)
            nxt = self.tq(b["next_obs"]).gather(1, best).squeeze(1)
            y = b["rew"] * a["reward_scale"] + a["gamma"] * (1 - b["done"]) * nxt
        pred = self.q(b["obs"]).gather(1, b["act"].long().unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % a["target_update"] == 0:
            self.tq.load_state_dict(self.q.state_dict())
        return float(loss.detach())

    def train_chunk(self):
        """One episode of epsilon-greedy interaction with an update per env step."""
        env = DREnv(self.env_cfg)
        obs, done, ret, loss = env.reset(self.sample_day()), False, 0.0, 0.0
        while not done:
            act = int(self.rng.integers(N_ACTIONS)) if self.rng.random() < self.eps() else self.greedy(obs)
            nobs, r, done, _ = env.step(act)
            self.buf.add(obs=obs, next_obs=nobs, act=act, rew=r, done=float(done))
            obs, ret = nobs, ret + r
            self.env_steps += 1
            if len(self.buf) >= self.a["learning_starts"]:
                loss = self.update()
        self.record(ret)
        return {"train_return": self.train_return(), "eps": self.eps(), "loss": loss}

    def extra_state(self):
        return {"buf": self.buf.state(), "updates": self.updates}

    def load_extra_state(self, s):
        self.buf.load(s["buf"])
        self.updates = s["updates"]
