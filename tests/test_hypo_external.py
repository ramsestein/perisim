"""Phases 3–4 of `hypo` on synthetic landmarks: the weighted calibration fits, the γ vote, the MAP threshold, the
drift lookup, source gaps inside a site, and the whole chain final → freeze → external stage → report, with the
freeze's refusals."""

import copy

import numpy as np
import polars as pl
import pytest

from hypo.casebase import CaseBase
from hypo.config import load
from hypo.data import concat
from hypo.data import load as load_lm
from hypo.drift import embed, knn_p1, lookup, value_codes
from hypo.external import gamma_vote
from hypo.frozen import Stop, threshold
from hypo.metric import fit_metric
from hypo.stats import calibration_line_w, citl_w, logit
from perisim.evaluation.metrics import calibration_in_the_large, calibration_line
from test_hypo_model import SPACE, _frame

CFG = load()


def _xframe(cohort: str, n_pat: int, per: int, seed: int) -> pl.DataFrame:
    """`_frame` plus the carried columns and, in eICU, three hospitals."""
    df = _frame(cohort, n_pat, per, seed)
    rng = np.random.default_rng(seed + 100)
    n = df.height
    pid = np.repeat(np.arange(n_pat), per)
    extra = {"y_first": df["y"].to_numpy() | (rng.random(n) < 0.05), "art_line": rng.random(n) < 0.5,
             "vaso_start": rng.random(n) < 0.1, "map_min_next": df["map_last"].to_numpy() - rng.random(n) * 10,
             "icu_discharge_24h": rng.random(n) < 0.2, "died_in_window": np.zeros(n, bool)}
    if cohort == "eicu":
        extra["hospital"] = np.where(pid < 0.4 * n_pat, "h_a", np.where(pid < 0.73 * n_pat, "h_b", "h_c"))
    return df.with_columns([pl.Series(k, v) for k, v in extra.items()])


@pytest.fixture(scope="module")
def mcfg():
    m = copy.deepcopy(CFG["metric"])
    m["shapes"].update(n_models=4, max_train=2000, max_val=1000, n_explain=600, subsample=300, n_reference_pairs=2000)
    m["nca"].update(epochs=4, n_candidates=100)
    m.update(alpha_queries=500)
    return m


@pytest.fixture(scope="module")
def metric_and_sites(mcfg):
    dev = concat([load_lm("mimic", CFG, _xframe("mimic", 300, 5, 1)), load_lm("eicu", CFG, _xframe("eicu", 300, 5, 2))])
    return fit_metric(dev, mcfg, SPACE, seed=0, threads=1), dev


def test_weighted_calibration_fits_match_perisim():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(2, 5, 4000), 1e-4, 1 - 1e-4)
    y = (rng.random(4000) < np.clip(1.3 * p, 0, 1)).astype(float)
    a, b = calibration_line(p, y)
    aw, bw = calibration_line_w(logit(p), y)
    assert aw == pytest.approx(a, abs=1e-8) and bw == pytest.approx(b, abs=1e-8)
    assert citl_w(logit(p), y) == pytest.approx(calibration_in_the_large(p, y), abs=1e-8)
    w = rng.integers(0, 3, 4000).astype(float)                   # weights = repeated rows
    rep = np.repeat(np.arange(4000), w.astype(int))
    assert citl_w(logit(p), y, w) == pytest.approx(calibration_in_the_large(p[rep], y[rep]), abs=1e-8)


def test_no_alpha_lets_a_unanimous_neighbourhood_predict_zero_or_one():
    k = int(CFG["metric"]["k"])
    I, valid, own = np.arange(k)[None], np.ones((1, k), bool), np.zeros((1, k), bool)
    for y in (np.zeros(k, int), np.ones(k, int)):
        for a in CFG["metric"]["alpha_grid"]:
            assert 0 < gamma_vote(y, I, valid, own, 1.0, float(a), 0.2)[0] < 1


def test_threshold_reads_where_the_odds_double():
    x = np.array([55.0, 65, 75, 85, 95])
    lo = np.array([2.0, 1.0, 0.3, 0.0, -0.1])                   # rises ln 2 above 85 between 75 and 65
    t = threshold(x, lo, 85.0, np.log(2))
    assert 65 < t < 75 and t == pytest.approx(75 - 10 * (np.log(2) - 0.3) / 0.7)
    assert np.isnan(threshold(x, np.zeros(5), 85.0, np.log(2)))


def test_gamma_one_is_the_plain_vote_and_lookup_reproduces_the_metric(metric_and_sites, mcfg):
    metric, dev = metric_and_sites
    cb = CaseBase(metric, dev, mcfg)
    cb.fit_alpha(0)
    q = load_lm("mimic", CFG, _xframe("mimic", 60, 5, 9))
    pr = cb.predict(q)
    own = np.zeros_like(pr["I"], dtype=bool)
    p1 = gamma_vote(dev.y, pr["I"], pr["valid"], own, 1.0, cb.alpha[cb.k], float(dev.y.mean()))
    assert np.allclose(p1, pr["p"])
    look = lookup(metric.enc, metric.w)
    codes = value_codes(metric.enc, dev.ft, look)
    exact = np.concatenate([lk["cols"] for lk in look if lk["exact"]])
    E = metric.embed(dev) * metric.sqrt_w_dims
    assert np.abs(embed([lk["points"] for lk in look], codes)[:, exact] - E[:, exact]).max() < 1e-5
    if all(lk["exact"] for lk in look):
        _, code_ref = np.unique(dev.ft.subjects.astype(str), return_inverse=True)
        p = knn_p1([lk["points"] for lk in look], codes, code_ref, dev.y, value_codes(metric.enc, q.ft, look),
                   cb.k, cb.alpha[cb.k], float(dev.y.mean()), int(mcfg["fetch_factor"]))
        assert np.allclose(p, pr["p"])


def test_a_source_gap_adds_nothing_inside_the_site(metric_and_sites):
    metric, _ = metric_and_sites
    ins = load_lm("inspire", CFG, _xframe("inspire", 50, 4, 5))
    E = metric.embed(ins) * metric.sqrt_w_dims
    for v in CFG["model"]["source_missing"]["inspire"]:
        for j, n in enumerate(metric.enc.feature_names):
            if n == v:
                assert np.ptp(E[:, metric.enc.dim_feature == j], axis=0).max() == 0


def test_final_freeze_external_stage_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    d = tmp_path / "experiments" / "hypo" / "landmarks"
    d.mkdir(parents=True)
    for i, c in enumerate(CFG["cohorts"]):
        _xframe(c, 120, 4, 10 + i).write_parquet(d / f"{c}.parquet")
    from hypo.cv import run
    from hypo.external import Stage
    from hypo.external_report import report
    from hypo.frozen import check, external_dir, final, final_dir, freeze

    with pytest.raises(Stop):                                     # nothing frozen yet
        check()
    run(CFG["cohorts"], ["prevalence", "last_map"], None, quick=False, threads=1)   # Phase 2 folds + comparators
    with pytest.raises(Stop):                                     # no frozen artefacts yet
        freeze("TEST", no_git=True, quick=True)
    final(quick=True, threads=1)
    assert len(list(final_dir().glob("*.pkl"))) == 3 + 3 * len(CFG["external"]["comparators"])
    with pytest.raises(Stop):                                     # quick artefacts, real freeze
        freeze("TEST", no_git=True)
    # the rehearsal: eICU served by the metric learned on MIMIC-IV + eICU; never touches INSPIRE as external
    dry = Stage(quick=True, threads=1, dry=True)
    assert dry.swap["eicu"] == "inspire" and "eicu" in dry.development("eicu")
    dry.run(["eicu"], folds=[0], parts=["transported", "hospitals", "drift"])
    dnames = {p.name for p in dry.out.glob("*")}
    assert dry.out.name == "external_dryrun" and "transported_eicu_f0.parquet" in dnames
    assert "drift_ext-eicu_inspire.json" not in dnames and "drift_ext-eicu_mimic.json" in dnames
    assert report(quick=True, dry=True)["dry"]
    assert not external_dir().exists() or not any(external_dir().glob("*.parquet"))
    rec = freeze("TEST", no_git=True, quick=True)
    assert rec["registration"] == "TEST" and check()["hashes"] == rec["hashes"]
    st = Stage(quick=True, threads=1)
    st.run(["eicu"], folds=[0, 1])
    out = external_dir()
    names = {p.name for p in out.glob("*")}
    for u in ("transported", "local", "comps", "curve", "hospitals"):
        assert f"{u}_eicu_f0.parquet" in names and f"{u}_eicu_f1.parquet" in names
    for c in CFG["cohorts"]:
        assert f"drift_ext-eicu_{c}.json" in names and f"shapes_{c}.json" in names
    assert "drift-hospitals_ext-eicu.json" in names
    T = pl.read_parquet(out / "transported_eicu_f0.parquet")
    assert {"p_gamma1", "p_gamma2", "p_gamma4", "p_gamma8", "own_share"} <= set(T.columns)
    assert np.allclose(T["p_gamma1"], T["p"])                     # γ = 1 is the plain vote
    with pytest.raises(Stop):                                     # external results exist: no second freeze
        freeze("TEST2", no_git=True, quick=True)
    res = report(quick=True)
    s = res["sites"]["eicu"]
    assert {"H1", "H2", "H3", "H4", "H5"} <= set(s["primary"])
    assert s["n_landmarks"] < len(load_lm("eicu", CFG))           # folds 0–1 only
    assert (tmp_path / "experiments" / "hypo" / "reports" / "external_quick.md").exists()
    # a frozen artefact that changes stops the stage
    pk = final_dir() / "comp_ext-eicu_last_map.pkl"
    pk.write_bytes(pk.read_bytes() + b"\0")
    with pytest.raises(Stop):
        check()
