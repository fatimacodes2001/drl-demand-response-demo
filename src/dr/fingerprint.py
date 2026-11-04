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

from .config import config_hash
from .tasks import make_task

SRC = Path(__file__).parent


def code_hash():
    """Hash of every Python source file in the package."""
    h = hashlib.sha256()
    for f in sorted(SRC.rglob("*.py")):
        h.update(f.relative_to(SRC).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def fingerprint(cfg, device):
    """Return the hash components and the combined `fingerprint`."""
    parts = dict(config_hash=config_hash(cfg), code_hash=code_hash(), data_hash=make_task(cfg).data_hash(cfg),
                 libs=f"torch{torch.__version__}+numpy{np.__version__}+py{platform.python_version()}",
                 platform=f"{platform.system()}-{platform.machine()}", device=device)
    parts["fingerprint"] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]
    return parts
