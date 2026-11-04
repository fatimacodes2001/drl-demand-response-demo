"""Compare tiny runs against values recorded on another machine (see scripts/golden.py).

The close comparison is the cross-platform reproducibility claim; the bitwise one is allowed to fail
because CPU architectures and BLAS builds may round differently.
"""
import json
import sys
from pathlib import Path

import pytest

from dr import runner
from dr.fingerprint import code_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from golden import ALGOS, tiny_config  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Run every tiny config once for the module."""
    if code_hash() != GOLDEN["dqn"]["code_hash"]:
        pytest.skip(f"golden values were recorded for code hash {GOLDEN['dqn']['code_hash']}; "
                    "run `python scripts/golden.py` to refresh them")
    out = tmp_path_factory.mktemp("golden")
    return {name: runner.run(tiny_config(name), out / name) for name in ALGOS}


@pytest.mark.parametrize("name", ALGOS)
def test_matches_golden_closely(results, name):
    """Results agree with the recorded values to 1e-4 relative."""
    for k, want in GOLDEN[name].items():
        if k != "code_hash":
            assert results[name][k] == pytest.approx(want, rel=1e-4), k


@pytest.mark.xfail(strict=False, reason="bitwise agreement across CPU architectures is not guaranteed")
@pytest.mark.parametrize("name", ALGOS)
def test_matches_golden_bitwise(results, name):
    """Results are bit-identical to the recorded values."""
    for k, want in GOLDEN[name].items():
        if k != "code_hash":
            assert results[name][k] == want, k
