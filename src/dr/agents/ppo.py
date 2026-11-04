"""PPO (clipped surrogate, GAE) with parallel env copies (policy-gradient baseline)."""
import numpy as np
import torch

from ..env import DREnv, OBS_DIM, N_ACTIONS
from .common import mlp


class PPOAgent:
    """Actor-critic PPO over the five discrete battery actions."""

    def __init__(self, cfg, device, seed):
        self.a, self.dev, self.env_cfg = cfg["algo"], device, cfg["env"]
        self.n_days = cfg["train"]["n_train_days"]
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        h = self.a["hidden"]
        self.pi = mlp(OBS_DIM, N_ACTIONS, h).to(device)
        self.vf = mlp(OBS_DIM, 1, h).to(device)
        self.opt = torch.optim.Adam([*self.pi.parameters(), *self.vf.parameters()], lr=self.a["lr"])
        self.envs = [DREnv(self.env_cfg) for _ in range(self.a["n_envs"])]
        self.obs = np.stack([e.reset(int(self.rng.integers(self.n_days))) for e in self.envs])
        self.ep_ret = np.zeros(len(self.envs))
        self.env_steps, self.recent = 0, []

    def policy(self):
        """Greedy policy `act(env) -> int`."""
        def act(env):
            with torch.no_grad():
                return int(self.pi(torch.as_tensor(env.obs(), device=self.dev)).argmax())
        return act

    def train_chunk(self):
        """Collect one rollout from all envs, then run several clipped-surrogate epochs on it."""
        a, dev, n = self.a, self.dev, len(self.envs)
        T = a["rollout_len"]
        O = torch.zeros(T, n, OBS_DIM, device=dev); A = torch.zeros(T, n, dtype=torch.long, device=dev)
        LP = torch.zeros(T, n, device=dev); R = torch.zeros(T, n, device=dev)
        D = torch.zeros(T, n, device=dev); V = torch.zeros(T + 1, n, device=dev)
        for t in range(T):
            o = torch.as_tensor(self.obs, device=dev)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=self.pi(o))
                act = dist.sample()
                O[t], A[t], LP[t], V[t] = o, act, dist.log_prob(act), self.vf(o).squeeze(1)
            rew, done = np.zeros(n), np.zeros(n)
            for i, e in enumerate(self.envs):
                ob, r, d, _ = e.step(int(act[i]))
                rew[i], done[i] = r, d
                self.ep_ret[i] += r
                if d:
                    self.recent = (self.recent + [self.ep_ret[i]])[-20:]
                    self.ep_ret[i] = 0.0
                    ob = e.reset(int(self.rng.integers(self.n_days)))
                self.obs[i] = ob
            R[t] = torch.as_tensor(rew * a["reward_scale"], dtype=torch.float32, device=dev)
            D[t] = torch.as_tensor(done, dtype=torch.float32, device=dev)
            self.env_steps += n
        with torch.no_grad():
            V[T] = self.vf(torch.as_tensor(self.obs, device=dev)).squeeze(1)
        adv, last = torch.zeros(T, n, device=dev), 0.0
        # generalised advantage estimation
        for t in reversed(range(T)):
            nd = 1 - D[t]
            delta = R[t] + a["gamma"] * V[t + 1] * nd - V[t]
            last = delta + a["gamma"] * a["gae_lambda"] * nd * last
            adv[t] = last
        ret = adv + V[:T]
        flat = lambda x: x.reshape(T * n, *x.shape[2:])
        O, A, LP, adv, ret = map(flat, (O, A, LP, adv, ret))
        for _ in range(a["epochs"]):
            perm = torch.randperm(T * n, device=dev, generator=self.gen)
            for s in range(0, T * n, a["minibatch"]):
                i = perm[s:s + a["minibatch"]]
                dist = torch.distributions.Categorical(logits=self.pi(O[i]))
                ratio = (dist.log_prob(A[i]) - LP[i]).exp()
                ad = (adv[i] - adv[i].mean()) / (adv[i].std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - a["clip"], 1 + a["clip"]) * ad).mean()
                vl = ((self.vf(O[i]).squeeze(1) - ret[i]) ** 2).mean()
                loss = pl + a["vf_coef"] * vl - a["ent_coef"] * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*self.pi.parameters(), *self.vf.parameters()], 0.5)
                self.opt.step()
        return {"train_return": float(np.mean(self.recent)) if self.recent else float("nan"),
                "policy_loss": pl.item(), "value_loss": vl.item()}

    def state_dict(self):
        return dict(pi=self.pi.state_dict(), vf=self.vf.state_dict(), opt=self.opt.state_dict(),
                    env_steps=self.env_steps, rng=self.rng.bit_generator.state)

    def load_state_dict(self, s):
        self.pi.load_state_dict(s["pi"]); self.vf.load_state_dict(s["vf"]); self.opt.load_state_dict(s["opt"])
        self.env_steps = s["env_steps"]
        self.rng.bit_generator.state = s["rng"]
