"""Experiment `roll` on synthetic data: the prequential loop, the arms, the update policy, the endpoints.

The brief's list (Phase 2): each patient is evaluated once and before entering; FIFO; exclusion by patient; the same
sequence for every arm; the ML trigger and delay work as declared; determinism; a case base equal to `hypo`'s
reproduces `hypo` (synthetic here; on the real data with PERISIM_TEST_ROLL_REAL=1)."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from roll import endpoints as EP
from roll import metrics as Mx
from roll import synthetic
from roll.arms import ML, Perisim
from roll.config import dry_settings, load
from roll.knn import CaseBase
from roll.loop import Run
from roll.ml import Model, closed_test
from roll.monitor import Davis, monitoring_rows
from roll.streams import build

from hypo.casebase import CaseBase as HypoCaseBase
from hypo.config import load as load_hypo


@pytest.fixture(scope="module")
def env():
    cfg = dry_settings(load())
    cfg["dry"]["patients"] = {"inspire": 500, "mimic": 800, "eicu": 1200}
    cfg["dry"]["landmarks_per_patient"] = 5
    cfg["dry"]["eicu_hospitals"] = 3
    hcfg = synthetic.quick_hcfg(load_hypo())
    mcfg = synthetic.quick_mcfg(load_hypo())
    cs = synthetic.cohorts(cfg, hcfg, seed=3)
    fz = synthetic.frozen(cs, hcfg, cfg["ml"]["families"], threads=2)
    return cfg, hcfg, mcfg, cs, fz


def _stream(env, name="mimic_temporal", order=0, N=None, hospital=""):
    cfg, _, _, cs, _ = env
    spec = cfg["streams"][name]
    st = build(name, spec, cs, cfg, order, N or int(spec["N_primary"]), hospital)
    st.make_windows(cfg["windows"]["W"])
    return st


@pytest.fixture(scope="module")
def ran(env):
    cfg, hcfg, mcfg, cs, fz = env
    st = _stream(env)
    run = Run(st, fz, cfg, mcfg, synthetic.SPACE, threads=2)
    calls = []
    fifo = next(a for a in run.arms if isinstance(a, Perisim) and a.mode == "fifo")
    orig = fifo.predict

    def spy(run_, rows):                      # the queried patients are never in the case base
        assert not np.isin(run_.pcode[rows], fifo.cb.code).any()
        calls.append(rows)
        return orig(run_, rows)

    fifo.predict = spy
    res = run.execute()
    return st, run, res, calls


# --------------------------------------------------------------------------- the loop
def test_each_patient_is_evaluated_once_and_before_entering(ran):
    st, run, res, _ = ran
    ev = run.pos_of[run.pos_of >= 0]
    rows = np.concatenate([run.rows_of(u) for u in st.seq])
    assert np.array_equal(np.sort(rows), np.sort(np.where(run.pos_of >= 0)[0]))     # every evaluated row once
    assert ev.size == rows.size
    pos_of_unit = {int(u): i for i, u in enumerate(st.seq)}
    for e in res["arms"]["perisim_fifo"]:
        if e["event"] == "rotate":
            assert all(pos_of_unit[u] < e["pos"] for u in e["in"])                   # entered after evaluation
    assert not set(int(u) for u in st.start) & set(pos_of_unit)                      # the start is never evaluated


def test_fifo_keeps_the_last_n_patients(ran):
    st, run, res, _ = ran
    fifo = next(a for a in run.arms if isinstance(a, Perisim) and a.mode == "fifo")
    expect = (list(st.start) + list(st.seq))[-st.N:]
    assert list(fifo.units) == [int(u) for u in expect]
    assert list(run.fifo) == list(fifo.units)                                        # the ML arms' update data
    cum = next(a for a in run.arms if isinstance(a, Perisim) and a.mode == "cumulative")
    assert len(cum.units) == len(st.start) + len(st.seq)
    stat = next(a for a in run.arms if isinstance(a, Perisim) and a.mode == "static")
    assert list(stat.units) == [int(u) for u in st.start]


def test_neighbours_are_distinct_patients_never_the_query_s_own(env):
    cfg, _, mcfg, cs, fz = env
    c = cs["mimic"]
    Ew = fz.Ew("mimic", c)
    pcode = np.unique(c.lm.ft.subjects.astype(str), return_inverse=True)[1]
    cb = CaseBase(Ew, c.lm.y, pcode, mcfg)
    cb.set_rows(np.arange(len(c.lm)))
    q = np.arange(0, len(c.lm), 7)
    D, I, valid = cb.search(q)
    nb = cb.code[I]
    assert not ((nb == pcode[q][:, None]) & valid).any()
    for i in range(len(q)):
        v = nb[i][valid[i]]
        assert v.size == np.unique(v).size


def test_every_arm_reads_the_same_sequence(ran, env):
    st, run, res, calls = ran
    masks = {a: np.isfinite(p) for a, p in run.pred.items()}
    first = next(iter(masks.values()))
    assert all(np.array_equal(first, m) for m in masks.values())
    assert np.array_equal(np.concatenate(calls), np.concatenate([run.rows_of(u) for u in st.seq]))
    assert _stream(env).digest() == st.digest()
    assert _stream(env, order=1).digest() != st.digest()


def test_windows_stay_inside_segments_and_steps_inside_windows(ran, env):
    st = ran[0]
    W = env[0]["windows"]["W"]
    for w in st.windows:
        seg = set(st.seg[w["start"]:w["end"]])
        assert seg == {w["seg"]}
        assert w["end"] - w["start"] >= int(W[st.pop[w["seg"]]]) / 2 or len([x for x in st.windows if x["seg"] == w["seg"]]) == 1
    for a, b, wi in st.steps(10):
        assert st.windows[wi]["start"] <= a < b <= st.windows[wi]["end"] and b - a <= 10


def test_the_trigger_the_delay_and_the_refractory_period(ran, env):
    st, run, res, _ = ran
    B = int(env[0]["loop"]["step"])
    seen = 0
    for name, log in res["arms"].items():
        if "_realistic" not in name:
            continue
        arm = next(a for a in run.arms if a.name == name)
        last_deploy = 0
        pending = None
        for e in log:
            if e["event"] == "trigger":
                seen += 1
                assert pending is None
                pending = e
            elif e["event"] == "deploy" and e.get("phase") == "update":
                assert pending is not None and e["trigger_pos"] == pending["pos"]
                assert pending["pos"] + pending["delay"] <= e["pos"] < pending["pos"] + pending["delay"] + B
                if pending["delay"] == 0:
                    assert e["pos"] == pending["pos"]
                last_deploy, pending = e["pos"], None
            elif e["event"] == "check":
                assert pending is None                                        # no check while an update is pending
                if e["decided"]:
                    assert e["pos"] - last_deploy >= run.M(st.seg[e["pos"] - 1])  # refractory: M patients since
        if arm.months == 0:
            assert all(e["delay"] == 0 for e in log if e["event"] == "trigger")
    assert seen > 0


def test_the_run_is_deterministic(env):
    cfg, hcfg, mcfg, cs, fz = env
    st = _stream(env, name="inspire_control")
    d1 = Run(st, fz, cfg, mcfg, synthetic.SPACE, threads=2).execute()["predictions_digest"]
    d2 = Run(_stream(env, name="inspire_control"), fz, cfg, mcfg, synthetic.SPACE, threads=1).execute()["predictions_digest"]
    assert d1 == d2


# --------------------------------------------------------------------------- PERISIM = hypo
def test_a_hypo_case_base_is_reproduced(env):
    cfg, _, mcfg, cs, fz = env
    c = cs["eicu"]
    metric = fz.metrics["eicu"]
    lm = c.lm
    E = metric.embed(lm)
    rng = np.random.default_rng(0)
    subj = lm.ft.subjects.astype(str)
    test_p = rng.choice(np.unique(subj), 200, replace=False)
    te, tr = np.where(np.isin(subj, test_p))[0], np.where(~np.isin(subj, test_p))[0]
    h = HypoCaseBase(metric, lm.take(tr), mcfg, E=E[tr])
    ha = h.fit_alpha(1234)
    hp = h.predict(lm.take(te), E=E[te])
    pcode = np.unique(subj, return_inverse=True)[1]
    r = CaseBase(fz.Ew("eicu", c), lm.y, pcode, mcfg)
    r.set_rows(tr)
    ra = r.fit_alpha(1234)
    rp = r.predict(te)
    assert ra["alpha"] == ha["alpha"] and ra["ood_threshold"] == ha["ood_threshold"]
    assert np.array_equal(rp["p"], hp["p"]) and np.array_equal(rp["ood"], hp["ood"])


@pytest.mark.skipif(os.environ.get("PERISIM_TEST_ROLL_REAL") != "1", reason="real data: PERISIM_TEST_ROLL_REAL=1")
def test_a_window_equal_to_the_cohort_reproduces_hypo_on_the_real_data():
    import polars as pl

    from hypo.cv import Sites, cv_dir
    from hypo.frozen import final_dir, load_pickle
    from hypo.config import workdir

    hcfg = load_hypo()
    S = Sites(hcfg, cv_dir(False))
    lm = S.lm("inspire")
    tr, te = S.split("inspire", 0)
    metric = load_pickle(final_dir() / "metric_ext-inspire.pkl")
    Ew = np.ascontiguousarray(metric.embed(lm) * metric.sqrt_w_dims, dtype=np.float32)
    pcode = np.unique(lm.ft.subjects.astype(str), return_inverse=True)[1]
    r = CaseBase(Ew, lm.y, pcode, hcfg["metric"])
    r.set_rows(tr)
    r.fit_alpha(int(hcfg["seed"]) + 100 * 0)
    p = r.predict(te)["p"]
    ref = pl.read_parquet(workdir() / "external" / "transported_inspire_f0.parquet").sort("row")
    assert np.array_equal(ref["row"].to_numpy(), te)
    assert np.array_equal(ref["p"].to_numpy(), p)


# --------------------------------------------------------------------------- the update hierarchy
def _toy(seed, n_pat=400, per=5, shift=0.0, slope=1.0):
    rng = np.random.default_rng(seed)
    n = n_pat * per
    code = np.repeat(np.arange(n_pat), per)
    X = rng.normal(0, 1, (n, 3))
    z = 1.2 * X[:, 0] - 0.8 * X[:, 1] - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-(shift + slope * z)))).astype(np.int64)
    p0 = 1 / (1 + np.exp(-z))
    return X, y, code, p0


@pytest.mark.parametrize("shift,slope,expected", [(0.0, 1.0, "keep"), (1.0, 1.0, "intercept"), (0.0, 0.4, "logistic")])
def test_closed_testing_picks_the_needed_update(shift, slope, expected):
    X, y, code, p0 = _toy(5, shift=shift, slope=slope)
    m = Model("logreg", lambda r: p0[r])
    ucfg = load()["update"]
    new, rec = closed_test(m, np.arange(len(y)), y, code, X, {"C": 1.0}, synthetic.SPACE, ucfg, 9, 1)
    assert rec["choice"] == expected
    assert rec["parameters_changed"] == {"keep": 0, "intercept": 1, "logistic": 2}[expected]


def test_a_refit_wins_when_the_model_is_wrong():
    X, y, code, _ = _toy(6, n_pat=600)
    bad = 1 / (1 + np.exp(-(0.3 * X[:, 2])))                      # unrelated to the outcome
    m = Model("logreg", lambda r: bad[r])
    new, rec = closed_test(m, np.arange(len(y)), y, code, X, {"C": 1.0}, synthetic.SPACE, load()["update"], 9, 1)
    assert rec["choice"] == "refit" and new.origin == "refit"


def test_monitoring_window_needs_m_patients_and_e_events():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    units = [np.array([i]) for i in range(100)]
    assert monitoring_rows(units[:40], 50, 1, y) is None
    r = monitoring_rows(units, 30, 40, y)                          # the last 30 hold 30 events: extended to 40
    assert r.size == 40 and y[r].sum() == 40
    assert monitoring_rows(units, 30, 200, y) is None


def test_adwin_fires_on_an_increase_of_the_error_only():
    rng = np.random.default_rng(1)
    d = Davis(0.05, 0.01, 50, 10)
    d.err = list(np.clip(rng.normal(0.05, 0.02, 400), 0, 1))
    assert not d.fires()
    d.err += list(np.clip(rng.normal(0.15, 0.02, 200), 0, 1))
    assert d.fires()
    d.err = list(np.clip(rng.normal(0.15, 0.02, 300), 0, 1)) + list(np.clip(rng.normal(0.05, 0.02, 300), 0, 1))
    assert not d.fires()                                            # a fall of the error is not a drift alarm


# --------------------------------------------------------------------------- data and endpoints
def test_admission_gate_refuses_bad_landmarks(env):
    from roll.data import admission_gate

    lm = copy.deepcopy(env[3]["inspire"].lm)
    lm.ft.num["map_last"][0] = 400.0
    lm.ft.num["age"][1] = 10.0
    lm.ft.ids[3] = lm.ft.ids[2]
    ok = admission_gate(lm, env[1]["bounds"])
    assert not ok[0] and not ok[1] and not ok[3] and ok[2] and ok[4:].all()


def test_window_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    p = np.round(rng.random(2000), 2)
    y = (rng.random(2000) < p).astype(float)
    w = rng.integers(0, 3, 2000).astype(float)
    assert Mx.auroc(p, y, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    assert Mx.auprc(p, y, w) == pytest.approx(average_precision_score(y, p, sample_weight=w), abs=1e-12)
    W = np.stack([w, np.ones(2000)])
    assert Mx.auroc(p, y, W)[1] == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_moving_blocks_stay_inside_segments():
    seg = np.array(["a"] * 10 + ["b"] * 4 + ["c"] * 1, dtype=object)
    for b in EP.mbb(seg, 3, 50, 0):
        assert b.size == seg.size
        assert (seg[b] == seg).all()                                # each segment resampled to its own size


def test_dersimonian_laird():
    y, se = np.array([0.10, 0.30, 0.20, -0.05]), np.array([0.10, 0.10, 0.20, 0.15])
    r = EP.dersimonian_laird(y, se)
    w = 1 / se ** 2
    q = (w * (y - (w * y).sum() / w.sum()) ** 2).sum()
    tau2 = max(0, (q - 3) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    assert r["tau2"] == pytest.approx(tau2) and r["estimate"] == pytest.approx((ws * y).sum() / ws.sum())
    assert EP.dersimonian_laird(np.array([0.2, 0.2]), np.array([0.1, 0.1]))["tau2"] == 0.0


def test_t2_stream_arms_and_scores():
    from roll import t2

    rng = np.random.default_rng(2)
    n_pat, per = 300, 2
    subj = np.repeat([f"mimic:{i}" for i in range(n_pat)], per)
    blocks = np.repeat(rng.choice(["2008 - 2010", "2011 - 2013"], n_pat), per)
    d = {"Ew": rng.normal(0, 1, (n_pat * per, 5)).astype(np.float32), "y": rng.integers(0, 4, n_pat * per),
         "subject": subj, "block": blocks, "k": 20, "alpha_grid": [0.5, 1.0], "ood_percentile": 99.0}
    cfg = load()
    cfg["t2"].update(N=60, W=30)
    cfg["endpoints"]["boot"] = 50
    runs = [t2.run(d, cfg, r) for r in range(2)]
    fr, log = runs[0]
    assert fr.height == sum(per for _ in range(log["stream"]["evaluated"]))       # every evaluated stay once
    P = np.column_stack([fr[f"p_perisim_fifo_{c}"].to_numpy() for c in range(4)])
    assert np.allclose(P.sum(1), 1)
    out = t2.endpoints(runs, cfg, 3)
    assert set(out["arms"]["rps"]) == set(t2.ARMS)
    assert t2.rps(np.full((1, 4), 0.25), np.array([0]))[0] == pytest.approx((0.75 ** 2 + 0.5 ** 2 + 0.25 ** 2) / 3)
    assert t2.rps(np.eye(4)[[2]], np.array([2]))[0] == 0.0


def test_endpoints_run_on_a_stream(ran, env):
    st, run, res, _ = ran
    out = EP.stream_endpoints([(run.frame(), res)], env[0], 1)
    c = out["contrasts"]["lightgbm_realistic"]
    assert set(c) >= {"H1_auroc", "H2_auroc_ratio", "H3_auroc_p10", "H4_ece", "H4_trigger_share"}
    assert np.isfinite(c["H1_auroc"]["estimate"]) and c["H1_auroc"]["ci"][0] <= c["H1_auroc"]["ci"][1]
    assert np.isnan(out["arms"]["prevalence"]["auroc_mean"])        # nothing to rank
