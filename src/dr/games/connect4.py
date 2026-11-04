"""Connect Four as a two-player environment for self-play search, plus scripted opponents.

Rewards are from the perspective of the player who just moved: +1 for a winning move, 0 otherwise.
The observation is two planes (own pieces, opponent pieces) relative to the player to move, so one
network plays both colours.
"""
import numpy as np

ROWS, COLS = 6, 7
N_ACTIONS = COLS
OBS_DIM = 2 * ROWS * COLS
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Connect4Env:
    """Standard 6x7 board; row 0 is the bottom."""

    two_player = True
    __slots__ = ("board", "heights", "player", "t", "done")

    def __init__(self, cfg=None):
        self.reset()

    def reset(self, seed=None):
        """Empty board, player 1 to move. `seed` is accepted for interface parity and ignored."""
        self.board = np.zeros((ROWS, COLS), np.int8)
        self.heights = np.zeros(COLS, np.int8)
        self.player, self.t, self.done = 1, 0, False
        return self.obs()

    def clone(self):
        c = Connect4Env.__new__(Connect4Env)
        c.board, c.heights = self.board.copy(), self.heights.copy()
        c.player, c.t, c.done = self.player, self.t, self.done
        return c

    def get_state(self):
        """Serialisable state for checkpoints."""
        return (self.board.copy(), self.heights.copy(), self.player, self.t, self.done)

    def set_state(self, s):
        self.board, self.heights, self.player, self.t, self.done = s[0].copy(), s[1].copy(), *s[2:]

    def legal(self):
        """Boolean mask of columns that are not full."""
        return [int(h) < ROWS for h in self.heights]

    def obs(self):
        own, opp = self.board == self.player, self.board == -self.player
        return np.concatenate([own.ravel(), opp.ravel()]).astype(np.float32)

    def step(self, c):
        """Drop the mover's piece in column `c`; reward +1 to the mover on a win, 0 otherwise."""
        r = int(self.heights[c])
        if self.done or r >= ROWS:
            raise ValueError(f"illegal move {c}")
        self.board[r, c] = self.player
        self.heights[c] += 1
        self.t += 1
        win = self._wins(r, c)
        self.done = win or self.t == ROWS * COLS
        self.player = -self.player
        return self.obs(), 1.0 if win else 0.0, self.done, {}

    def _wins(self, r, c):
        p = self.board[r, c]
        for dr, dc in DIRS:
            n = 1
            for s in (1, -1):
                rr, cc = r + s * dr, c + s * dc
                while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == p:
                    n += 1
                    rr, cc = rr + s * dr, cc + s * dc
            if n >= 4:
                return True
        return False

    def render(self):
        """Text board with X for player 1 and O for player -1, top row first."""
        sym = {1: "X", -1: "O", 0: "."}
        return "\n".join(" ".join(sym[int(v)] for v in row) for row in self.board[::-1])


def random_player(rng):
    """Uniformly random legal move."""
    return lambda env: int(rng.choice(np.flatnonzero(env.legal())))


def winblock_player(rng):
    """Take an immediate win, otherwise block the opponent's immediate win, otherwise play randomly."""

    def immediate_win(env, player):
        for c in np.flatnonzero(env.legal()):
            e = env.clone()
            e.player = player
            if e.step(int(c))[1] > 0:
                return int(c)
        return None

    def act(env):
        c = immediate_win(env, env.player)
        if c is None:
            c = immediate_win(env, -env.player)
        return c if c is not None else int(rng.choice(np.flatnonzero(env.legal())))
    return act


def play(first, second, opening_seed=None, opening_plies=2):
    """One game; returns +1 if `first` wins, -1 if `second` wins, 0 for a draw.

    With `opening_seed`, the first `opening_plies` moves are random from that seed so that a fixed set
    of seeds gives a fixed, varied set of starting positions for deterministic players.
    """
    env = Connect4Env()
    if opening_seed is not None:
        rng = np.random.default_rng(opening_seed)
        for _ in range(opening_plies):
            env.step(int(rng.choice(np.flatnonzero(env.legal()))))
    players = (first, second)
    mover = 0
    while not env.done:
        _, r, done, _ = env.step(players[mover](env))
        if done:
            return 0 if r == 0 else (1 if mover == 0 else -1)
        mover ^= 1
    return 0
