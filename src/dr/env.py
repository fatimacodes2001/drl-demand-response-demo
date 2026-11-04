"""Single-household demand-response environment (synthetic and public-safe).

One episode is one day of 48 half-hour steps. A home battery arbitrages a time-of-use tariff and earns
a bonus for discharging during FCAS-style frequency events. Prices, load, PV and event times are known
day-ahead, so the dynamics are deterministic given the day seed. The env is therefore an exact simulator
for tree search, and `oracle` gives the optimum on the discrete action grid by dynamic programming.
"""
import numpy as np

STEPS = 48
DT = 0.5
UNIT = 1.25
N_LEVELS = 10
CAP = N_LEVELS * UNIT
SOC0_LEVEL = N_LEVELS // 2
N_ACTIONS = 5
LOOKAHEAD = 6
OBS_DIM = 1 + 2 + LOOKAHEAD + 2 + 2

DEFAULTS = dict(fcas_price=0.8, degradation=0.03, feedin=0.06)


class Scenario:
    """One day of exogenous signals, generated deterministically from the integer `day`."""

    def __init__(self, day, p):
        r = np.random.default_rng(day)
        h = np.arange(STEPS) * DT
        g = lambda mu, sd: np.exp(-(((h - mu) / sd) ** 2))
        season = r.uniform(0.6, 1.4)
        load = (0.35 + 0.5 * g(7.5, 1.5) + 1.1 * season * g(19, 2.2)
                + r.normal(0, 0.05, STEPS)).clip(0.1)
        cloud = r.uniform(0.3, 1.0)
        pv = (np.sin(np.pi * (h - 6) / 12).clip(0) * 3.0 * (2.0 - season) * cloud
              * r.uniform(0.85, 1.0, STEPS))
        price = np.where((h >= 16) & (h < 21), 0.55, np.where((h >= 7) & (h < 16), 0.30, 0.18))
        price = price * r.uniform(0.8, 1.3)
        for _ in range(r.integers(0, 3)):
            i = r.integers(0, STEPS - 2)
            price[i:i + r.integers(1, 3)] *= r.uniform(2.0, 4.0)
        event = np.zeros(STEPS)
        for _ in range(r.integers(0, 4)):
            i = r.integers(0, STEPS - 3)
            event[i:i + r.integers(1, 4)] = 1.0
        pad = np.pad(price, (0, LOOKAHEAD), mode="edge")
        self.day = day
        self.load, self.pv, self.price, self.event = (x.tolist() for x in (load, pv, price, event))
        self.look = [(pad[t:t + LOOKAHEAD] / 0.5).tolist() for t in range(STEPS)]
        self.feedin, self.fcas, self.degr = p["feedin"], p["fcas_price"], p["degradation"]
        self.term_price = float(price.mean())

    def reward(self, t, discharge):
        """Reward in dollars for discharging `discharge` kWh (negative means charging) at step t."""
        grid = self.load[t] - self.pv[t] - discharge
        cost = self.price[t] * grid if grid > 0 else self.feedin * grid
        bonus = self.fcas * max(discharge, 0.0) if self.event[t] else 0.0
        return -cost + bonus - self.degr * abs(discharge)

    def terminal(self, soc):
        """Value of the end-of-day state of charge relative to the starting level."""
        return (soc - SOC0_LEVEL * UNIT) * self.term_price


class DREnv:
    """Gym-style environment with cheap `clone()` so tree search can branch from any state."""

    __slots__ = ("p", "sc", "t", "soc")

    def __init__(self, cfg=None):
        self.p = {**DEFAULTS, **(cfg or {})}
        self.sc = None

    def reset(self, day):
        """Start the day generated from seed `day` at half full and return the first observation."""
        self.sc = Scenario(day, self.p)
        self.t, self.soc = 0, SOC0_LEVEL * UNIT
        return self.obs()

    def clone(self):
        c = DREnv.__new__(DREnv)
        c.p, c.sc, c.t, c.soc = self.p, self.sc, self.t, self.soc
        return c

    def legal(self):
        """Boolean mask of discrete actions that keep the battery inside its limits."""
        return [-1e-9 <= self.soc - (a - 2) * UNIT <= CAP + 1e-9 for a in range(N_ACTIONS)]

    def obs(self):
        s, t = self.sc, min(self.t, STEPS - 1)
        ang = 2 * np.pi * t / STEPS
        nxt = max(s.event[t + 1:t + 3], default=0.0)
        return np.array([self.soc / CAP, np.sin(ang), np.cos(ang), *s.look[t],
                         s.load[t] / 2.0, s.pv[t] / 3.0, s.event[t], nxt], dtype=np.float32)

    def step(self, a):
        """Advance one step. `a` is an int in 0..4 (discrete) or a float in [-1, 1] (+1 = full discharge)."""
        if isinstance(a, (int, np.integer)):
            d = (a - 2) * UNIT
        else:
            d = float(min(max(a, -1.0), 1.0)) * 2 * UNIT
        d = min(max(d, self.soc - CAP), self.soc)
        r = self.sc.reward(self.t, d)
        self.soc -= d
        self.t += 1
        done = self.t >= STEPS
        if done:
            r += self.sc.terminal(self.soc)
        return self.obs(), r, done, {}


def _value_table(sc):
    """Backward dynamic programming over the 11 state-of-charge levels; returns V[t][level]."""
    V = [None] * (STEPS + 1)
    V[STEPS] = np.array([sc.terminal(i * UNIT) for i in range(N_LEVELS + 1)])
    for t in reversed(range(STEPS)):
        V[t] = np.array([
            max((sc.reward(t, (a - 2) * UNIT) + V[t + 1][i - (a - 2)]
                 for a in range(N_ACTIONS) if 0 <= i - (a - 2) <= N_LEVELS), default=-np.inf)
            for i in range(N_LEVELS + 1)])
    return V


def oracle(sc):
    """Exact optimal return for a day on the discrete action grid."""
    return float(_value_table(sc)[0][SOC0_LEVEL])


def oracle_policy(sc):
    """Return a policy `act(env) -> int` that follows the dynamic-programming optimum for day `sc`."""
    V = _value_table(sc)

    def act(env):
        i, t = int(round(env.soc / UNIT)), env.t
        return max((a for a in range(N_ACTIONS) if 0 <= i - (a - 2) <= N_LEVELS),
                   key=lambda a: sc.reward(t, (a - 2) * UNIT) + V[t + 1][i - (a - 2)])
    return act
