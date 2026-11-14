"""scripts/review_analyses.py (DECISIONS 2026-11-15): its statistics, the leakage guard of the external stage (block F),
and the whole script on a synthetic workdir."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_analyses as RA  # noqa: E402

from hypo.config import load  # noqa: E402

CFG = load()


# --------------------------------------------------------------------------- statistics
def test_recalibration_methods():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 4000)
    y = (rng.random(4000) < np.clip(p * 0.5, 0, 1)).astype(float)          # the model over-predicts twice
    for m in ("intercept", "logistic", "isotonic"):
        q = RA.recal(m, p, y, p)
        assert abs(q.mean() - y.mean()) < 0.01, m
    assert np.array_equal(RA.recal("logistic", p[:5], np.ones(5), p), p)  # one class: identity
    assert np.all(np.diff(RA.recal("isotonic", p, y, np.sort(p))) >= -1e-12)


def test_rank_nodes_approx_is_exact_with_few_values_and_close_otherwise():
    rng = np.random.default_rng(1)
    srt = np.sort(rng.integers(0, 30, 2000).astype(float))
    exact = lambda x: (np.searchsorted(srt, x, "left") + np.searchsorted(srt, x, "right")) / (2.0 * srt.size)
    x = np.r_[srt[::7], np.nan]
    med = float(np.median(srt))
    a = RA.rank_nodes_approx(x, srt, med, nodes=101)
    assert np.allclose(a[:-1], exact(x[:-1])) and a[-1] == exact(np.array([med]))[0]
    srt2 = np.sort(rng.normal(size=20000))
    x2 = rng.normal(size=500)
    e2 = (np.searchsorted(srt2, x2, "left") + np.searchsorted(srt2, x2, "right")) / (2.0 * srt2.size)
    assert np.max(np.abs(RA.rank_nodes_approx(x2, srt2, 0.0) - e2)) < 0.01


def test_jaccard_rows():
    A = np.array([[1, 2, 3, -1], [5, 6, 7, 8]])
    B = np.array([[1, 2, 4, -1], [5, 6, 7, 8]])
    assert np.allclose(RA.jaccard_rows(A, B), [2 / 4, 1.0])


def test_sustained_and_onsets_and_episodes():
    stay_lm = np.array(["a", "a", "b", "c"])
    L = np.array([2, 4, 2, 2])
    died = np.array([False, False, False, True])
    sh = np.array(["a", "a", "a", "b"])
    hh = np.array([5, 6, 9, 3])
    mh = np.array([60.0, 62.0, 70.0, 63.0])
    y2 = RA.sustained(stay_lm, L, died, sh, hh, mh)
    assert y2.tolist() == [1, 1, 0, 1]                    # a: hours 5, 6 low in [2, 8) and [4, 10); c: death
    on = RA.onsets(sh, hh, mh, {"a": 2, "b": 2, "c": 2})
    assert on == {"a": 5, "b": 3}
    y = np.array([1, 1, 1, 0])
    p = np.array([0.4, 0.1, 0.05, 0.6])
    e = RA.episodes(stay_lm, L, y, p, on, 0.3)
    assert e["episodes"] == 2 and e["detected_share"] == 0.5 and e["lead_median_h"] == 3.0
    assert e["false_alerts_per_patient_day"] == pytest.approx(1 / (4 / 12))


def test_estimands_and_cluster_weights():
    rng = np.random.default_rng(2)
    n = 3000
    subj = np.repeat(np.arange(600), 5).astype(str)
    hosp = np.where(np.arange(n) < 1500, "h1", "h2")
    y = (rng.random(n) < 0.2).astype(float)
    base = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.05, n), 0.01, 0.99)
    P = {"transported": base, "local": base, "euclidean": np.full(n, 0.2)}
    e = RA.estimands(P, y, subj, 0, n_boot=200)
    assert e["D"]["est"] == 0 and e["D"]["equivalent"] and e["R"]["est"] == pytest.approx(1.0)
    w = next(RA.cluster_weights(subj, hosp, 1, 0))
    assert w.sum() == pytest.approx(3000) and np.all(w >= 0)       # equal hospitals: 600 patient weights × 5 rows


# --------------------------------------------------------------------------- block F: leakage guard + whole script
def _frames(tmp_path):
    from test_hypo_external import _xframe

    d = tmp_path / "experiments" / "hypo" / "landmarks"
    d.mkdir(parents=True)
    for i, c in enumerate(CFG["cohorts"]):
        df = _xframe(c, 120, 4, 10 + i).with_columns(pl.lit(1).alias("win_hours"), pl.lit(100.0).alias("los_h"))
        if c == "eicu":                                       # landmarks of windows with no MAP hour
            extra = df.head(30).with_columns(pl.lit(False).alias("known"), pl.lit(0).alias("win_hours"),
                                             (pl.col("L") + 1).alias("L"))
            df = pl.concat([df, extra])
        df.write_parquet(d / f"{c}.parquet")
        stays = df["stay_id"].unique().to_list()
        rng = np.random.default_rng(i)
        v = pl.DataFrame({"stay_id": np.repeat(stays, 12), "h": np.tile(np.arange(12), len(stays)).astype(float) + 0.2,
                          "var": "map_nibp", "value": rng.normal(72, 8, 12 * len(stays))})
        (tmp_path / "experiments" / "hypo" / "extract").mkdir(exist_ok=True)
        v.write_parquet(tmp_path / "experiments" / "hypo" / "extract" / f"{c}_vitals.parquet")


@pytest.fixture(scope="module")
def frozen_chain(tmp_path_factory):
    import os
    tmp = tmp_path_factory.mktemp("review")
    old = os.environ.get("PERISIM_WORKDIR")
    os.environ["PERISIM_WORKDIR"] = str(tmp)
    _frames(tmp)
    from hypo.cv import run
    from hypo.frozen import final, freeze
    import hypo.frozen as F

    seen = {}
    real = F.fit_metric

    def spy(lm, *a, **k):                                     # the frozen metric of each rotation: its training sites
        seen.setdefault("final_sites", []).append(sorted(set(np.asarray(lm.site).astype(str))))
        return real(lm, *a, **k)

    F.fit_metric = spy
    try:
        run(CFG["cohorts"], ["prevalence", "last_map", "euclidean"], None, quick=False, threads=1)
        final(quick=True, threads=1)
    finally:
        F.fit_metric = real
    freeze("TEST", no_git=True, quick=True)
    yield tmp, seen
    if old is None:
        os.environ.pop("PERISIM_WORKDIR", None)
    else:
        os.environ["PERISIM_WORKDIR"] = old


def test_no_supervised_step_sees_the_fold_under_test(frozen_chain, monkeypatch):
    tmp, seen = frozen_chain
    # the frozen artefacts of rotation X never see X (each fit sees exactly two sites)
    assert len(seen["final_sites"]) == 3 and all(len(s) == 2 for s in seen["final_sites"])
    assert sorted(tuple(s) for s in seen["final_sites"]) == sorted(
        tuple(sorted(c for c in CFG["cohorts"] if c != x)) for x in CFG["cohorts"])
    import hypo.external as X
    from hypo.casebase import CaseBase

    rec = {"case_base": [], "local_metric": [], "recal": []}
    real_init, real_fit, real_draw = CaseBase.__init__, X.fit_metric, X.Stage.draw

    def cb_init(self, metric, lm, *a, **k):
        rec["case_base"].append(set(np.asarray(lm.ft.subjects).astype(str)))
        return real_init(self, metric, lm, *a, **k)

    def fit(lm, *a, **k):
        rec["local_metric"].append(set(np.asarray(lm.ft.subjects).astype(str)))
        return real_fit(lm, *a, **k)

    def draw(self, c, f, tr, n, r):
        pos = real_draw(self, c, f, tr, n, r)
        rec["recal"].append(set(np.asarray(self.S.lm(c).ft.subjects)[tr[pos]].astype(str)))
        return pos

    monkeypatch.setattr(CaseBase, "__init__", cb_init)
    monkeypatch.setattr(X, "fit_metric", fit)
    monkeypatch.setattr(X.Stage, "draw", draw)
    st = X.Stage(quick=True, threads=1)
    st.run(["eicu"], folds=[0], parts=["transported", "local", "comps", "curve"])
    test_subjects = set(np.asarray(st.S.lm("eicu").ft.subjects)[st.S.split("eicu", 0)[1]].astype(str))
    for k in rec:
        assert rec[k], k
        assert all(not (s & test_subjects) for s in rec[k]), f"{k} saw a patient of the fold under test"


def test_review_script_end_to_end(frozen_chain):
    tmp, _ = frozen_chain
    from hypo.external import Stage
    from hypo.frozen import external_dir

    Stage(quick=True, threads=1).run(["eicu"])                                  # every fold of eICU
    assert RA.main(["--quick", "--units", "eicu", "--threads", "1", "--n-src", "2",
                    "--blocks", "est", "A", "B", "C", "D", "E", "G"]) == 0
    import json
    res = json.loads((external_dir().with_name("review") / "review.json").read_text())
    u = "hypo_eicu"
    assert {"D", "R", "dnb", "cluster_hospital"} <= set(res["est"][u])
    assert "all" in res["A"][u]["by_n"] and "perisim" in res["A"][u]["by_n"]["all"]
    assert {"raw_euclidean", "shapes_uniform", "shapes_nca", "score_1d", "additive_n500"} <= set(res["B"][u]["arms"])
    assert res["C"][u]["n_src"] == 2
    assert res["D"][u]["windows"]["dropped_no_map_hour"] == 30
    assert res["E"][u]["artefact_quantiles"]["embedding_rebuild_gap"] == 0
    assert (tmp / "experiments" / "hypo" / "reports" / "review.md").exists()


def test_cluster_weights_with_patients_in_several_hospitals():
    subj = np.array(["a", "a", "b", "c", "c"])
    hosp = np.array(["h1", "h2", "h1", "h3", "h2"])          # h2 holds only patients first seen elsewhere
    for w in RA.cluster_weights(subj, hosp, 50, 1):
        assert np.all(np.isfinite(w)) and np.all(w >= 0)
        assert w[0] == w[1] and w[3] == w[4]                 # a patient's rows share one weight
