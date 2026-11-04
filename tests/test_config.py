"""Config loading tests."""
from dr import config


def test_scientific_notation_without_dot_is_a_float(tmp_path):
    """`1e-3` in a YAML file or an override parses as a number, not a string."""
    f = tmp_path / "c.yaml"
    f.write_text("a: {lr: 1e-3, n: 5, name: foo, big: 2E5}\n")
    cfg = config.load(f, ["a.lr=5e-4", "a.extra=1e2"])
    assert cfg["a"] == {"lr": 5e-4, "n": 5, "name": "foo", "big": 2e5, "extra": 100.0}
    assert all(isinstance(cfg["a"][k], float) for k in ("lr", "big", "extra"))


def test_extends_merges_nested_keys(tmp_path):
    """A child config overrides only the keys it sets."""
    (tmp_path / "base.yaml").write_text("x: {a: 1, b: 2}\ny: 3\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nx: {b: 9}\n")
    assert config.load(tmp_path / "child.yaml") == {"x": {"a": 1, "b": 9}, "y": 3}


def test_config_hash_ignores_logging_keys():
    """Tracker settings do not change a run's identity."""
    assert config.config_hash({"a": 1, "tracker": "mlflow"}) == config.config_hash({"a": 1, "tracker": "none"})
    assert config.config_hash({"a": 1}) != config.config_hash({"a": 2})
