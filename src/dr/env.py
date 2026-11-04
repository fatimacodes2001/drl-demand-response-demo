"""Single-household demand-response environment (synthetic and public-safe).

One episode is one day of 48 half-hour steps. A home battery arbitrages a time-of-use tariff and earns
a bonus for discharging during FCAS-style frequency events. Prices, load, PV and event times are known
day-ahead and the whole remaining day is part of the observation, so every agent (model-free or
search-based) plans from the same information and the dynamic-programming `oracle` is a reachable
ceiling rather than a clairvoyant one.
"""
import numpy as np

STEPS = 48
DT = 0.5
UNIT = 1.25
N_LEVELS = 10
CAP = N_LEVELS * UNIT
SOC0_LEVEL = N_LEVELS // 2
N_ACTIONS = 5
MAX_KWH = 2 * UNIT
OBS_DIM = 3 + 3 * STEPS
PRICE_SCALE, LOAD_SCALE = 0.5, 2.0

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
        self.day = day
        self.load, self.pv, self.price, self.event = (x.tolist() for x in (load, pv, price, event))
        self.feedin, self.fcas, self.degr = p["feedin"], p["fcas_price"], p["degradation"]
        self.term_price = float(price.mean())
        # remaining-day signals, zero-padded so obs(t) can take a fixed-length slice
        pad = lambda x: np.pad(np.asarray(x, np.float32), (0, STEPS))
        self.future = np.stack([pad(price / PRICE_SCALE), pad(event), pad((load - pv) / LOAD_SCALE)])

    def reward(self, t, discharge):
        """Reward in dollars for discharging `discharge` kWh (negative means charging) at step t."""
        grid = self.load[t] - self.pv[t] - discharge
        cost = self.price[t] * grid if grid > 0 else self.feedin * grid
        bonus = self.fcas * max(discharge, 0.0) if self.event[t] else 0.0
        return -cost + bonus - self.degr * abs(discharge)

    def reward_vec(self, t, discharge):
        """Vectorised `reward` over an array of discharge amounts."""
        grid = self.load[t] - self.pv[t] - discharge
        cost = np.where(grid > 0, self.price[t] * grid, self.feedin * grid)
        bonus = self.fcas * np.maximum(discharge, 0.0) * self.event[t]
        return -cost + bonus - self.degr * np.abs(discharge)

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

    def get_state(self):
        """Serialisable state for checkpoints."""
        return (self.sc.day, self.t, self.soc)

    def set_state(self, s):
        day, self.t, self.soc = s
        if self.sc is None or self.sc.day != day:
            self.sc = Scenario(day, self.p)

    def legal(self):
        """Boolean mask of discrete actions that keep the battery inside its limits."""
        return [-1e-9 <= self.soc - (a - 2) * UNIT <= CAP + 1e-9 for a in range(N_ACTIONS)]

    def obs(self):
        t = min(self.t, STEPS - 1)
        ang = 2 * np.pi * t / STEPS
        return np.concatenate([np.array([self.soc / CAP, np.sin(ang), np.cos(ang)], np.float32),
                               self.sc.future[:, t:t + STEPS].ravel()])

    def step(self, a):
        """Advance one step. `a` is an int in 0..4 (discrete) or a float in [-1, 1] (+1 = full discharge)."""
        if isinstance(a, (int, np.integer)):
            d = (a - 2) * UNIT
        else:
            d = float(min(max(a, -1.0), 1.0)) * MAX_KWH
        d = min(max(d, self.soc - CAP), self.soc)
        r = self.sc.reward(self.t, d)
        self.soc -= d
        self.t += 1
        done = self.t >= STEPS
        if done:
            r += self.sc.terminal(self.soc)
        return self.obs(), r, done, {}


def _value_table(sc, sub):
    """Backward dynamic programming on a grid of UNIT/sub kWh; returns (V[t][level], unit)."""
    unit, levels, amax = UNIT / sub, N_LEVELS * sub, 2 * sub
    V = [None] * (STEPS + 1)
    V[STEPS] = np.array([sc.terminal(i * unit) for i in range(levels + 1)])
    acts = np.arange(-amax, amax + 1)
    for t in reversed(range(STEPS)):
        r = sc.reward_vec(t, acts * unit)
        best = np.full(levels + 1, -np.inf)
        for a, ra in zip(acts, r):
            # discharging a units moves level i to i - a; only in-range moves are candidates
            lo, hi = max(0, a), min(levels, levels + a)
            np.maximum(best[lo:hi + 1], ra + V[t + 1][lo - a:hi - a + 1], out=best[lo:hi + 1])
        V[t] = best
    return V, unit


def oracle(sc, sub=10):
    """Optimal return for a day on a grid `sub` times finer than the agents' discrete actions.

    `sub=1` is the exact optimum on the discrete grid. The default is a tight bound for continuous
    policies: the cost is piecewise linear in the discharge, so refining the grid changes little.
    """
    V, _ = _value_table(sc, sub)
    return float(V[0][SOC0_LEVEL * sub])


def oracle_policy(sc, sub=10):
    """Policy `act(env) -> float` following the dynamic-programming optimum on the `sub` grid."""
    V, unit = _value_table(sc, sub)
    levels, amax = N_LEVELS * sub, 2 * sub

    def act(env):
        i, t = int(round(env.soc / unit)), env.t
        acts = [a for a in range(-amax, amax + 1) if 0 <= i - a <= levels]
        a = max(acts, key=lambda a: sc.reward(t, a * unit) + V[t + 1][i - a])
        return a * unit / MAX_KWH
    return act
