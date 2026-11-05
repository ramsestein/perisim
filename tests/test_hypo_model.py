"""Metric and case base of `hypo` on synthetic landmarks: exclusions, within-site candidates, site-constant inputs,
the learning sample and a metric that finds the informative input."""

import copy

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from hypo.casebase import CaseBase
from hypo.config import load
from hypo.data import concat, learning_sample
from hypo.data import load as load_lm
from hypo.metric import SiteNCA, fit_encoder, fit_metric

CFG = load()
SPACE = {"families": {"logreg": {"weight": 1, "params": {"C": 1.0, "max_iter": 500}},
                      "lightgbm": {"weight": 1, "params": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 8,
                                                           "min_child_samples": 20, "subsample": 1.0,
                                                           "colsample_bytree": 1.0, "reg_lambda": 1.0}}}}


def _frame(cohort: str, n_pat: int, per: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pat * per
    inp = CFG["model"]["inputs"]
    pid = np.repeat(np.arange(n_pat), per)
    cols = {v: rng.normal(0, 1, n) for v in inp["continuous"]}
    base = rng.normal(80, 8, n_pat)[pid]                    # a patient's MAP level: its landmarks are near-copies
    cols["map_last"] = base + rng.normal(0, 2, n)
    cols["L"] = np.tile(np.arange(2, 2 + 2 * per, 2), n_pat).astype(float)
    cols.update({v: (rng.random(n) < 0.3).astype(float) for v in inp["binary"]})
    y = rng.random(n) < 1 / (1 + np.exp((cols["map_last"] - 75) / 3))
    df = pl.DataFrame({**cols, "surgical_group": rng.choice(["a", "b", "c"], n),
                       "stay_id": [f"s{i}" for i in pid], "subject_id": [f"p{i}" for i in pid],
                       "y": y, "band": np.where(y, 0, 2).astype(np.int8), "normotensive": True, "known": True})
    if cohort == "inspire":
        df = df.with_columns([pl.lit(None, pl.Float64).alias(v) for v in CFG["model"]["source_missing"]["inspire"]])
    return df.with_columns(pl.col("L").cast(pl.Int32))


@pytest.fixture(scope="module")
def lm():
    return concat([load_lm("inspire", CFG, _frame("inspire", 300, 5, 1)), load_lm("mimic", CFG, _frame("mimic", 400, 5, 2))])


@pytest.fixture(scope="module")
def mcfg():
    m = copy.deepcopy(CFG["metric"])
    m["shapes"].update(n_models=4, max_train=2000, max_val=1000, n_explain=600, subsample=300, n_reference_pairs=2000)
    m["nca"].update(epochs=4, n_candidates=100)
    m.update(alpha_queries=500)
    return m


def test_learning_sample_caps_patients_and_balances_sites(lm):
    rows = learning_sample(lm, per_patient=2, per_site=10_000, seed=0)
    s = lm.ft.subjects[rows]
    assert max(np.unique(s, return_counts=True)[1]) <= 2
    per_site = [int((lm.site[rows] == c).sum()) for c in ("inspire", "mimic")]
    assert per_site == [600, 600]                            # the smallest site offers 300 patients × 2
    assert np.array_equal(rows, learning_sample(lm, 2, 10_000, seed=0))


def test_site_constant_inputs_leave_the_metric(lm, mcfg):
    enc, dropped = fit_encoder(lm, mcfg["shapes"])
    assert "vaso_on__missing" in dropped and "vaso_hours__missing" in dropped   # = "this is INSPIRE"
    assert "vaso_on" in enc.feature_names                     # informative within the other site


def test_nca_candidates_stay_in_the_anchor_site_and_skip_its_patient(lm, mcfg):
    enc, _ = fit_encoder(lm, mcfg["shapes"])
    E = enc.transform(lm.ft)
    nca = SiteNCA(enc.dim_feature, enc.n_features, 2, n_candidates=50, sites=lm.site)
    nca.refresh_candidates(E, lm.ft.subjects)
    cand = np.where(nca.valid, nca.cand, -1)
    rows = np.repeat(np.arange(len(lm)), cand.shape[1])[cand.ravel() >= 0]
    c = cand.ravel()[cand.ravel() >= 0]
    assert (lm.site[c] == lm.site[rows]).all()
    assert (lm.ft.subjects[c] != lm.ft.subjects[rows]).all()


@pytest.fixture(scope="module")
def fitted(lm, mcfg):
    tr = np.isin(lm.ft.subjects, np.unique(lm.ft.subjects)[::2]) | (lm.site == "inspire")   # half of mimic held out
    rows = np.where(tr)[0][learning_sample(lm.take(np.where(tr)[0]), 3, 10_000, seed=0)]
    return tr, fit_metric(lm.take(rows), mcfg, SPACE, seed=3, threads=2)


def test_metric_weights_the_informative_input(fitted):
    _, m = fitted
    w = m.weights_table()
    assert w["map_last"] == max(w.values())
    assert np.isfinite(m.w).all() and m.info["n_accepted"] == 4


def test_case_base_excludes_own_patient_and_repeats_no_patient(lm, fitted, mcfg):
    tr, m = fitted
    mimic = np.where(lm.site == "mimic")[0]
    cb = CaseBase(m, lm.take(mimic[tr[mimic]]), mcfg)
    info = cb.fit_alpha(seed=0)
    assert info["alpha"] in mcfg["alpha_grid"] and info["short_neighbourhoods"] < 0.05
    q = lm.take(mimic)                                       # held-out patients and the case base's own
    out = cb.predict(q)
    nb = np.where(out["valid"], cb.subj_s[out["I"]], "")
    assert not (nb == np.asarray(q.ft.subjects, dtype=str)[:, None]).any()
    for r in nb[:200]:
        r = r[r != ""]
        assert len(set(r)) == len(r)
    test = ~tr[mimic]
    assert roc_auc_score(q.y[test], out["p"][test]) > 0.75
    assert np.allclose(out["p_band"].sum(1), 1.0) and out["p_band"].shape[1] == 3


def test_metric_is_reproducible(lm, mcfg):
    rows = learning_sample(lm, 2, 400, seed=1)
    a = fit_metric(lm.take(rows), mcfg, SPACE, seed=5, threads=1)
    b = fit_metric(lm.take(rows), mcfg, SPACE, seed=5, threads=1)
    assert np.allclose(a.w, b.w)
