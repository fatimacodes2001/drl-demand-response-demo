"""Connect Four rules and scripted-opponent tests."""
import numpy as np
import pytest

from dr.games.connect4 import Connect4Env, OBS_DIM, ROWS, COLS, play, random_player, winblock_player


def moves(env, cols):
    """Play a sequence of columns and return the last (reward, done)."""
    out = None
    for c in cols:
        _, r, done, _ = env.step(c)
        out = (r, done)
    return out


def test_vertical_horizontal_and_diagonal_wins():
    """Each line direction ends the game with reward 1 to the mover."""
    assert moves(Connect4Env(), [0, 1, 0, 1, 0, 1, 0]) == (1.0, True)
    assert moves(Connect4Env(), [0, 0, 1, 1, 2, 2, 3]) == (1.0, True)
    assert moves(Connect4Env(), [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]) == (1.0, True)


def test_non_winning_moves_give_zero_and_illegal_moves_raise():
    """Ordinary moves return 0 and a full column or finished game rejects moves."""
    e = Connect4Env()
    assert moves(e, [3, 3, 3, 3, 3, 3]) == (0.0, False)
    assert e.legal() == [True] * 3 + [False] + [True] * 3
    with pytest.raises(ValueError):
        e.step(3)


def test_draw_fills_the_board():
    """A full board with no line is a draw."""
    e = Connect4Env()
    order = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
             0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
    last = None
    for c in order:
        if e.done:
            break
        last = e.step(c)
    assert e.t == ROWS * COLS and e.done and last[1] == 0.0


def test_obs_is_relative_to_mover_and_clone_is_independent():
    """Pieces swap planes when the player changes; clones do not share arrays."""
    e = Connect4Env()
    e.step(3)
    o = e.obs()
    assert o.shape == (OBS_DIM,) and o[:ROWS * COLS].sum() == 0 and o[ROWS * COLS:].sum() == 1
    c = e.clone()
    c.step(3)
    assert e.t == 1 and c.t == 2 and e.board[1, 3] == 0


def test_winblock_wins_when_possible_and_blocks_otherwise():
    """The scripted opponent takes a win in one and blocks a threat in one."""
    e = Connect4Env()
    moves(e, [0, 6, 0, 6, 0])
    assert winblock_player(np.random.default_rng(0))(e) == 0
    e = Connect4Env()
    moves(e, [0, 6, 0, 6, 0, 5])
    assert winblock_player(np.random.default_rng(0))(e) == 0


def test_winblock_beats_random_most_of_the_time():
    """Sanity check on opponent strength ordering."""
    rng = np.random.default_rng(0)
    results = [play(winblock_player(rng), random_player(rng), opening_seed=s) for s in range(60)]
    results += [-play(random_player(rng), winblock_player(rng), opening_seed=s) for s in range(60)]
    assert np.mean(np.array(results) > 0) > 0.75
