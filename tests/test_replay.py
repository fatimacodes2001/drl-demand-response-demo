"""Prioritised replay sampling tests."""
import numpy as np

from dr.agents.common import PrioritizedReplay, SumTree


def test_sum_tree_matches_cumulative_sums():
    """Prefix lookups on the tree agree with a plain cumulative sum."""
    rng = np.random.default_rng(0)
    vals = rng.random(37)
    t = SumTree(37)
    t.set(np.arange(37), vals)
    assert abs(t.total - vals.sum()) < 1e-9
    q = rng.random(200) * vals.sum()
    assert (t.find(q) == np.minimum(np.searchsorted(np.cumsum(vals), q), 36)).all()


def test_sampling_frequency_follows_priority():
    """A transition with 8x the priority is drawn about 8x as often (alpha = 1)."""
    buf = PrioritizedReplay(8, 2, [("act", ())], alpha=1.0)
    for i in range(8):
        buf.add(obs=[i, i], act=i)
    buf._set(np.arange(8), np.array([1.0] * 7 + [8.0]))
    rng = np.random.default_rng(1)
    counts = np.zeros(8)
    for _ in range(400):
        counts += np.bincount(buf.sample(rng, 32, "cpu")["idx"], minlength=8)
    assert abs(counts[7] / counts[:7].mean() - 8) < 1.0


def test_weights_are_normalised_and_checkpoint_roundtrips():
    """Importance weights are at most 1, and a saved buffer restores the same sampling distribution."""
    buf = PrioritizedReplay(16, 2, [("act", ())])
    for i in range(10):
        buf.add(obs=[i, i], act=i)
    buf.update(np.arange(10), np.linspace(0.1, 2, 10))
    assert buf.sample(np.random.default_rng(0), 8, "cpu")["w"].max() <= 1 + 1e-6
    other = PrioritizedReplay(16, 2, [("act", ())])
    other.load(buf.state())
    assert np.allclose(other.tree.tree, buf.tree.tree)
