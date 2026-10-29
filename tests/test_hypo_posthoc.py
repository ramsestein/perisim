"""The helpers of scripts/posthoc_hypo.py (exploratory follow-ups of `hypo`): the weight swap and the MAP block."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_hypo as P  # noqa: E402
from hypo.config import load  # noqa: E402
from hypo.data import concat  # noqa: E402
from hypo.data import load as load_lm  # noqa: E402
from hypo.drift import lookup  # noqa: E402
from hypo.frozen import logodds_curve  # noqa: E402
from hypo.metric import fit_metric  # noqa: E402
from test_hypo_external import _xframe  # noqa: E402
from test_hypo_model import SPACE  # noqa: E402

CFG = load()
TCFG = CFG["external"]["threshold"]


def test_swapped_weights_rescale_to_the_transported_total():
    m = SimpleNamespace(w=np.array([2.0, 1.0, 1.0]), enc=SimpleNamespace(feature_names=["a", "b", "c"]))
    loc = {"a": 0.1, "b": 0.6}                                  # "c" missing from the local metric
    assert np.allclose(P.swapped_weights(m, loc), [0.4, 2.4, 0.0])
    assert np.allclose(P.swapped_weights(m, loc, ["b"]), [2.0, 2.4, 1.0])


def test_block_stats_on_a_straight_line():
    lo = -0.1 * (P.GRID - 85.0)                                  # log-odds rise 0.1 per mmHg below 85
    s = P.block_stats(lo, TCFG)
    assert s["threshold"] == pytest.approx(85 - 10 * float(TCFG["rise"]), abs=1e-6)
    assert s["or_70"] == pytest.approx(np.exp(1.5)) and s["or_65"] == pytest.approx(np.exp(2.0))


@pytest.fixture(scope="module")
def metric():
    m = copy.deepcopy(CFG["metric"])
    m["shapes"].update(n_models=4, max_train=2000, max_val=1000, n_explain=600, subsample=300, n_reference_pairs=2000)
    m["nca"].update(epochs=4, n_candidates=100)
    m.update(alpha_queries=500)
    dev = concat([load_lm("mimic", CFG, _xframe("mimic", 300, 5, 1)), load_lm("eicu", CFG, _xframe("eicu", 300, 5, 2))])
    return fit_metric(dev, m, SPACE, seed=0, threads=1)


def test_swapping_in_the_same_weights_changes_nothing(metric):
    same = P.swapped_weights(metric, metric.weights_table())
    assert np.allclose(same, metric.w)
    a, b = lookup(metric.enc, metric.w), lookup(metric.enc, same)
    assert all(np.allclose(x["points"], y["points"]) for x, y in zip(a, b))


def test_map_block_is_the_sum_of_the_three_curves(metric):
    enc, fi = metric.enc, metric.enc.feature_index()
    if any(enc.blocks[fi[v]].shape is None for v in P.MAPV if v in fi):
        pytest.skip("a MAP summary is not shaped in the synthetic metric")
    g = {v: enc.blocks[fi[v]].shape for v in P.MAPV}
    want = sum(np.interp(P.GRID, *logodds_curve(enc, v, g[v])) for v in P.MAPV)
    assert np.allclose(P.block_curve(enc, g), want)
    res = P.map_block(enc, metric.info["curves"], TCFG)
    assert res["aligned"] and res["models"]["threshold"]["n"] == len(metric.info["curves"]["map_last"])
