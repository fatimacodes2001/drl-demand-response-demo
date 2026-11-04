"""Run identity: a hash of config, code, generated data and library versions.

Two runs with the same fingerprint should produce the same metrics (bitwise on CPU). Any change to the
config, to a source file under src/dr, to the generated day set, to library or Python versions, to the
OS/CPU architecture or to the resolved device changes it. scripts/ are deliberately excluded: they only
launch and report, so editing them must not invalidate recorded runs.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from . import evaluate as ev
from .config import config_hash
from .env import DREnv, Scenario

SRC = Path(__file__).parent


def code_hash():
    """Hash of every Python source file in the package."""
    h = hashlib.sha256()
    for f in sorted(SRC.rglob("*.py")):
        h.update(f.relative_to(SRC).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def data_hash(cfg):
    """Hash of every generated day a run can touch plus the env parameters.

    With real data this is where a dataset checksum (for example a DVC hash) would go.
    """
    p = DREnv(cfg["env"]).p
    days = [*range(cfg["train"]["n_train_days"]), *ev.days_for("val", cfg["eval"]["val_days"]),
            *ev.days_for("test", cfg["eval"]["test_days"])]
    h = hashlib.sha256(json.dumps(p, sort_keys=True).encode())
    for d in days:
        s = Scenario(d, p)
        for arr in (s.load, s.pv, s.price, s.event):
            h.update(np.asarray(arr, np.float64).tobytes())
    return h.hexdigest()[:12]


def fingerprint(cfg, device):
    """Return the hash components and the combined `fingerprint`."""
    parts = dict(config_hash=config_hash(cfg), code_hash=code_hash(), data_hash=data_hash(cfg),
                 libs=f"torch{torch.__version__}+numpy{np.__version__}+py{platform.python_version()}",
                 platform=f"{platform.system()}-{platform.machine()}", device=device)
    parts["fingerprint"] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]
    return parts
