"""Compare tiny runs against values recorded by scripts/golden.py.

On the platform and library versions the values were recorded with, bitwise agreement is a hard test (it is
the same-machine determinism claim). Anywhere else the comparisons are informational: they run and report,
but a mismatch is an expected possibility, because BLAS builds and CPU architectures round differently and
1500 training steps amplify the last bit into visibly different numbers.
"""
import json
import sys
from pathlib import Path

import pytest

from dr import runner
from dr.fingerprint import fingerprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from golden import ALGOS, tiny_config  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())
RECORDED = GOLDEN["_recorded_on"]
HERE = fingerprint(tiny_config("dqn"), "cpu")
SAME_PLATFORM = (HERE["libs"], HERE["platform"]) == (RECORDED["libs"], RECORDED["platform"])


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Run every tiny config once for the module."""
    if HERE["code_hash"] != RECORDED["code_hash"]:
        pytest.skip(f"golden values were recorded for code hash {RECORDED['code_hash']}; "
                    "run `make golden` to refresh them")
    out = tmp_path_factory.mktemp("golden")
    return {name: runner.run(tiny_config(name), out / name) for name in ALGOS}


def _compare(results, name, rel):
    for k, want in GOLDEN[name].items():
        got = results[name][k]
        assert got == pytest.approx(want, rel=rel), f"{k}: {got!r} vs recorded {want!r}"


@pytest.mark.skipif(not SAME_PLATFORM, reason="recorded on a different platform or library versions")
@pytest.mark.parametrize("name", ALGOS)
def test_bitwise_on_recording_platform(results, name):
    """Same platform and libraries: results are bit-identical to the recorded values."""
    _compare(results, name, rel=0.0)


@pytest.mark.skipif(SAME_PLATFORM, reason="same platform: covered by the bitwise test")
@pytest.mark.xfail(strict=False, reason="cross-platform agreement is reported, not promised")
@pytest.mark.parametrize("name", ALGOS)
def test_close_on_other_platforms(results, name):
    """Other platforms: within 1e-4 relative of the recorded values."""
    _compare(results, name, rel=1e-4)
