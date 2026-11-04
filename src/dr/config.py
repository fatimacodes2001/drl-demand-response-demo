"""YAML experiment configs with `extends`, CLI overrides and a stable hash."""
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

NON_FUNCTIONAL = ("tracker", "experiment")


def _merge(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class _Loader(yaml.SafeLoader):
    """SafeLoader that reads `1e-3` as a float (YAML 1.1 requires `1.0e-3`)."""


_Loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(
    r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?$|^[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?$"),
    list("-+0123456789."))


def _parse(val):
    return yaml.load(val, Loader=_Loader)


def load(path, overrides=()):
    """Load a config, resolve `extends`, then apply `a.b=value` overrides (values parsed as YAML)."""
    path = Path(path)
    cfg = yaml.load(path.read_text(), Loader=_Loader)
    if "extends" in cfg:
        cfg = _merge(load(path.parent / cfg.pop("extends")), cfg)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = _parse(val)
    return cfg


def config_hash(cfg):
    """Short hash of everything that can change results; logging-only keys are excluded."""
    cfg = {k: v for k, v in cfg.items() if k not in NON_FUNCTIONAL}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
