"""Internal validation of `hypo` (Phase 2) on synthetic landmarks: the weighted AUROC of the bootstrap, patient-level
folds, the local comparators and one whole rotation fold, resumable, into the aggregate report."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hypo import compare
from hypo.config import load
from hypo.cv import outer_folds, settings
from hypo.data import load as load_lm
from hypo.report import auc_prep, auc_weighted, site_metrics
from test_hypo_model import _frame

CFG = load()


def test_weighted_auroc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.random(3000) < 0.3
    s = np.round(rng.random(3000) + 0.5 * y, 1)                 # many ties
    assert auc_weighted(y, np.ones(3000), auc_prep(s)) == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 4, 3000).astype(float)
    rep = np.repeat(np.arange(3000), w.astype(int))
    assert auc_weighted(y, w, auc_prep(s)) == pytest.approx(roc_auc_score(y[rep], s[rep]))


def test_outer_folds_are_patient_level_and_cover_every_landmark():
    lm = load_lm("mimic", CFG, _frame("mimic", 200, 5, 3))
    f = outer_folds(lm, 5, seed=1)
    assert (f >= 0).all() and set(f) == set(range(5))
    for s in np.unique(lm.ft.subjects):
        assert len(set(f[lm.ft.subjects == s])) == 1


def test_paired_difference_with_itself_is_zero():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.3).astype(int)
    p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.1, 2000), 0.01, 0.99)
    subj = np.repeat(np.arange(400), 5).astype(object)
    r = site_metrics({"PERISIM": p, "copy": p.copy()}, y, subj, 50, 0, ["PERISIM"])
    d = r["paired"]["PERISIM − copy"]
    assert d["logloss"] == 0 and d["logloss_ci"] == [0.0, 0.0] and d["auroc_ci"] == [0.0, 0.0]
    m = r["methods"]["PERISIM"]
    assert m["auroc_ci"][0] <= m["auroc"] <= m["auroc_ci"][1]


@pytest.mark.parametrize("name", ["prevalence", "last_map", "logreg", "lightgbm", "euclidean"])
def test_local_comparators_fit_on_training_and_predict_probabilities(name):
    lm = load_lm("mimic", CFG, _frame("mimic", 300, 5, 4))
    f = outer_folds(lm, 5, seed=0)
    tr, te = lm.take(np.where(f != 0)[0]), lm.take(np.where(f == 0)[0])
    base = settings(False)[2]["baselines"]
    base["lgbm_grid"] = {**base["lgbm_grid"], "num_leaves": [7], "learning_rate": [0.1]}
    pred, info = compare.fit(name, tr, CFG["metric"], base, n_inner=3, seed=0, threads=1)
    p = pred(te)
    assert p.shape == (len(te),) and ((p > 0) & (p < 1)).all()
    if name not in ("prevalence",):
        assert roc_auc_score(te.y, p) > 0.75                  # the synthetic outcome depends on the last MAP


def test_one_rotation_fold_runs_resumes_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    d = tmp_path / "experiments" / "hypo" / "landmarks"
    d.mkdir(parents=True)
    for i, c in enumerate(CFG["cohorts"]):
        _frame(c, 150, 4, 10 + i).write_parquet(d / f"{c}.parquet")
    from hypo.cv import run
    from hypo.report import report

    out = run(["eicu"], ["perisim", "prevalence", "last_map"], [0], quick=True, threads=1)
    files = sorted(p.name for p in out.glob("*.parquet"))
    assert "perisim_ext-eicu_f0.parquet" in files and "comp_mimic_f0_last_map.parquet" in files
    stamp = (out / "perisim_ext-eicu_f0.parquet").stat().st_mtime_ns
    run(["eicu"], ["perisim"], [0], quick=True, threads=1)          # finished units are kept
    assert (out / "perisim_ext-eicu_f0.parquet").stat().st_mtime_ns == stamp
    res = report(quick=True)
    assert set(res["sites"]) == {"inspire", "mimic"}
    m = res["sites"]["mimic"]
    assert {"PERISIM (with inspire)", "prevalence", "last_map"} == set(m["methods"])
    assert "PERISIM (with inspire) − last_map" in m["paired"] and m["folds"] == [0]
    assert (tmp_path / "experiments" / "hypo" / "reports" / "cv_quick.md").exists()
