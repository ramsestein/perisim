"""scripts/posthoc_alignment_robustness.py on synthetic caches."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_alignment_robustness as R  # noqa: E402


def test_overlap_and_knn_self():
    A = np.array([[1, 2, 3], [4, 5, -1]])
    B = np.array([[3, 2, 9], [7, 8, 9]])
    assert R.overlap(A, B).tolist() == [2 / 3, 0.0]
    E = np.array([[0.0], [1.0], [10.0], [11.0]])
    assert sorted(R.knn_self(E, 1)[:, 0].tolist()) == [0, 1, 2, 3]


def _cache(tmp: Path, X: str, rng):
    nq, nb, d = 1200, 5000, 6
    base = rng.normal(size=(nb, d))
    Q = rng.normal(size=(nq, d))
    names = ["transported", "local", "zscore+uniform"] + [f"{s}_{x}" for s in R.SIZES for x in "ab"]
    for i, m in enumerate(names):
        w = 1 + rng.random(d) * (0.2 if m != "zscore+uniform" else 2)
        D = ((Q[:, None, :] * w - base[None, :400, :] * w) ** 2).sum(2)
        I = np.argsort(D, 1)[:, :50]
        p = 1 / (1 + np.exp(-(Q[:, 0] * w[0] + rng.normal(0, 0.1, nq))))
        np.savez(tmp / f"hypo_{X}_{m}.npz", I=I, valid=np.ones_like(I, bool), p_q=p, Ewq=(Q * w).astype(np.float32),
                 alpha=np.array(1.0))


def test_end_to_end(tmp_path):
    cache = tmp_path / "experiments" / "posthoc_alignment" / "cache" / "units"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    _cache(cache, "mimic", rng)
    R.N_BOOT = 100
    assert R.main(["--workdir", str(tmp_path), "--sites", "mimic"]) == 0
    out = json.loads((cache.parents[1] / "robustness.json").read_text())
    s = out["sites"]["mimic"]
    assert set(s["case_base"]) == {f"k{k}" for k in R.KS_CB}
    assert "reading" in s["R2"] and "section_15_3_robust" in out["rule"]
    assert "Rule (pre-specified)" in (cache.parents[1] / "robustness.md").read_text()
