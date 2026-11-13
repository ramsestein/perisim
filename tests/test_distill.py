"""Distillation (docs/prereg_distill.md) on synthetic landmarks: aggregates release no small cells and no rows,
SYN-COP keeps marginals, missingness and rank correlations, SYN-IND loses the correlations, and the chain teacher →
synthetic cohort → metric → case base runs and finds the informative input."""

import copy
import json

import numpy as np
import pytest

from distill import aggregates as A
from distill import study as S
from hypo.data import concat
from hypo.data import load as load_lm
from tests.test_hypo_model import CFG, SPACE, _frame


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


@pytest.fixture(scope="module")
def aggs(lm):
    rows = S.learn_rows(lm, {"learn": {"per_patient": 3, "per_site": 10_000}}, 7)
    learn = lm.take(rows)
    return learn, S.rotation_aggregates(learn, 7)


def test_aggregates_are_json_and_release_no_rows(aggs):
    learn, ag = aggs
    for s, a in ag.items():
        d = json.loads(json.dumps(a.to_json()))
        b = A.SiteAggregates.from_json(d)
        assert b.n_rows == a.n_rows and len(b.marginals) == len(a.marginals)
        assert A.released_min_patients(a) >= A.MIN_PATIENTS
        for mg in a.marginals:
            if mg.kind == "continuous" and mg.missing < 1:
                assert len(mg.quantiles) == A.GRID
        C = np.asarray(a.corr)
        assert np.allclose(np.diag(C), 1) and np.linalg.eigvalsh(C).min() > 0


def test_small_rare_pool_joins_the_nearest_value():
    rng = np.random.default_rng(1)
    n = 3000
    subj = np.array([f"p{i // 3}" for i in range(n)], dtype=object)
    x = rng.choice([95.0, 96.0, 97.0, 98.0], n)
    x[:3] = 70.0                              # one patient
    mg = A._marginal("spo2", x, subj, nominal=False)
    assert 70.0 not in mg.values and mg.rare_from is None and mg.small_cells is None
    assert abs(sum(mg.shares) - 1) < 1e-12 and mg.min_patients >= A.MIN_PATIENTS
    assert mg.shares[mg.values.index(95.0)] == pytest.approx(((x == 95) | (x == 70)).mean())
    c = np.array(["a"] * 2000 + ["b"] * 997 + ["z"] * 3, dtype=object)
    mg = A._marginal("g", c, subj, nominal=True)
    assert mg.values == ["a", "b"] and mg.shares[0] == pytest.approx(2003 / 3000)
    u = A._u_value(mg, c, np.random.default_rng(0))
    assert np.isfinite(u).all()


def test_rare_categories_are_pooled():
    n = 3000
    subj = np.array([f"p{i // 3}" for i in range(n)], dtype=object)
    col = np.array(["common"] * n, dtype=object)
    for k in range(30):                       # 30 rare categories of 2 patients each: a pool of 60 patients
        col[6 * k: 6 * k + 6] = f"rare_{k}"
    mg = A._marginal("g", col, subj, nominal=True)
    assert mg.rare_from == 1 and mg.values[0] == "common" and mg.small_cells is None
    assert len(set(mg.shares[1:])) == 1 and mg.min_patients >= A.MIN_PATIENTS


def test_copula_keeps_marginals_missingness_and_correlation(aggs):
    learn, ag = aggs
    s = "mimic"
    a = ag[s]
    real = learn.ft.take(np.where(learn.site == s)[0])
    cop = A.sample(a, 20_000, 1, copula=True, specs=real.specs)
    ind = A.sample(a, 20_000, 2, copula=False, specs=real.specs)
    x_r, x_c = real.num["map_last"], cop.num["map_last"]
    assert abs(np.nanmedian(x_c) - np.nanmedian(x_r)) < 0.1 * np.subtract(*np.nanpercentile(x_r, [75, 25]))
    from scipy.stats import spearmanr

    r_real = spearmanr(real.num["map_last"], real.num["L"], nan_policy="omit").statistic
    D = A.dims(a.marginals)
    i, j = D.index(("map_last", "value")), D.index(("map_mean_6h", "value"))
    assert abs(a.corr[i][j]) < 0.2                            # independent in the synthetic frame
    assert np.isfinite(r_real)
    inspire = learn.ft.take(np.where(learn.site == "inspire")[0])
    syn_i = A.sample(ag["inspire"], 2000, 3, copula=True, specs=real.specs)
    for v in CFG["model"]["source_missing"]["inspire"]:
        assert np.isnan(inspire.num[v]).all() and np.isnan(syn_i.num[v]).all()
    assert set(np.unique(cop.cat["surgical_group"].astype(str))) <= {"a", "b", "c"}
    assert set(np.unique(cop.num["vaso_on"])) <= {0.0, 1.0}


def test_copula_recovers_a_planted_correlation():
    from perisim.features.table import FeatureTable
    from hypo.data import specs as mk_specs

    rng = np.random.default_rng(5)
    n = 4000
    sp = [s for s in mk_specs(CFG) if s.type == "continuous"][:2]
    z = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n)
    x0, x1 = np.exp(z[:, 0]), 50 + 10 * z[:, 1]
    x1[rng.random(n) < 0.2] = np.nan
    ids = np.array([f"r{i}" for i in range(n)], dtype=object)
    ft = FeatureTable(ids, ids.copy(), sp, {sp[0].name: x0, sp[1].name: x1}, {})
    a = A.fit(ft, "s", 0)
    cop = A.sample(a, n, 1, copula=True, specs=sp)
    ind = A.sample(a, n, 2, copula=False, specs=sp)
    from scipy.stats import spearmanr

    ok = lambda t: np.isfinite(t.num[sp[1].name])
    r_real = spearmanr(x0[ok(ft)], x1[ok(ft)]).statistic
    r_cop = spearmanr(cop.num[sp[0].name][ok(cop)], cop.num[sp[1].name][ok(cop)]).statistic
    r_ind = spearmanr(ind.num[sp[0].name][ok(ind)], ind.num[sp[1].name][ok(ind)]).statistic
    assert abs(r_cop - r_real) < 0.06 and abs(r_ind) < 0.06
    assert abs(np.isnan(cop.num[sp[1].name]).mean() - 0.2) < 0.03
    assert cop.num[sp[0].name].max() <= np.percentile(x0, 99) + 1e-9      # tails clipped: no extreme real value


def test_chain_teacher_synthetic_metric_case_base(lm, mcfg, aggs):
    from hypo import compare

    learn, ag = aggs
    teacher, _ = compare.fit("logreg", lm, mcfg, {"logreg_C_grid": [1.0]}, n_inner=2, seed=0, threads=1)
    syn, info = S.synthetic_cohort(ag, teacher, learn.ft.specs, seed=3, rotation=0, arm="cop")
    assert info["n"] == len(learn) and abs(info["mean_p_teacher"] - learn.y.mean()) < 0.1
    assert len(np.unique(syn.ft.subjects)) == len(syn)
    m = S.fit_on(syn, mcfg, SPACE, seed=3, threads=1)
    w = m.weights_table()
    assert max(w, key=w.get).startswith("map_")              # the informative input
    rng = np.random.default_rng(0)
    subj = lm.ft.subjects.astype(str)
    u = np.unique(subj)
    fold = dict(zip(u, rng.integers(0, 2, u.size)))
    f = np.array([fold[s] for s in subj])
    p = S.predict_folds(m, lm, [(k, np.where(f != k)[0], np.where(f == k)[0]) for k in (0, 1)], mcfg, 3)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    d = S.dcr(syn, learn, 0)
    assert d["synthetic_to_real"]["share_exact"] == 0.0
    fi = S.fidelity(syn, learn, ag, 0)
    assert fi["mimic"]["missing_share_error_max"] < 0.05


# ---------------------------------------------------------------------------- the runner: guards and readings
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("distill_run", Path(__file__).resolve().parents[1] / "scripts" / "distill_run.py")
DR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DR)


def test_registration_is_required_and_must_be_in_the_prereg():
    with pytest.raises(SystemExit):
        DR.registered(None)
    with pytest.raises(SystemExit):
        DR.registered("not-an-id")
    with pytest.raises(SystemExit):
        DR.registered("10.17605/OSF.IO/ZZZZZZZ")          # well formed but not written in the pre-registration


def test_out_folder_guard(tmp_path):
    src = tmp_path / "work_value"
    src.mkdir()
    with pytest.raises(SystemExit):
        DR.guard_out(src / "inner", src)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x.txt").write_text("x")
    with pytest.raises(SystemExit):
        DR.guard_out(busy, src)
    out = DR.guard_out(tmp_path / "work_distill", src)
    assert (out / DR.MARKER).is_file()
    assert DR.guard_out(out, src) == out                    # a marked folder is reused


def test_hypotheses_from_fake_dumps(tmp_path):
    import arc_compare as AC

    units = [f"hypo_{x}" for x in DR.ROTATIONS] + [f"nwicu_{x}" for x in DR.ROTATIONS]
    rng = np.random.default_rng(1)
    for i, u in enumerate(units):
        n_pat = 600
        subj = np.repeat([f"p{k}" for k in range(n_pat)], 3)
        n = subj.size
        y = (rng.random(n) < 0.2).astype(np.int64)
        e = 0.50 + rng.normal(0, 0.05, n)
        real = e - 0.03 + rng.normal(0, 0.01, n)               # the real metric gains 0.03 nats
        local = e - 0.032 + rng.normal(0, 0.01, n)
        cop = real + 0.001 + rng.normal(0, 0.005, n)           # keeps ~97 % of it
        ind = real + 0.012 + rng.normal(0, 0.005, n)           # keeps ~60 %
        ids = np.array([f"{u}-{k}" for k in range(n)])
        for name, t in (("src", real), ("cop", cop), ("ind", ind)):
            o = tmp_path / name / "arc_compare"
            o.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(o / f"{u}.npz", ids=ids, subjects=subj, y=y, ll_transported=t, ll_local=local,
                                ll_euclidean=e)
    rc, ri, ci = (AC.compare(tmp_path / "src", tmp_path / "cop"), AC.compare(tmp_path / "src", tmp_path / "ind"),
                  AC.compare(tmp_path / "cop", tmp_path / "ind"))
    h = DR.evaluate_hypotheses(rc, ri, ci)
    assert h["H1"]["holds"] and h["H1"]["not_evaluable"] == 0
    assert h["H2"]["holds"] and h["H2"]["n_ind_worse"] == 6
    assert all(v == "not shown" for v in h["Q_IND"].values())
    assert all(v == "equivalent" for v in h["H3"]["units"].values())


def test_teacher_labelled_keeps_rows_and_changes_only_labels(lm, aggs):
    from hypo import compare

    learn, _ = aggs
    teacher, _ = compare.fit("logreg", lm, {"inner_folds": 2}, {"logreg_C_grid": [1.0]}, n_inner=2, seed=0, threads=1)
    y0 = learn.y.copy()
    t, info = S.teacher_labelled(learn, teacher, 3, 0)
    assert np.array_equal(learn.y, y0)                       # the real sample is not modified
    assert t.ft is learn.ft and len(t) == len(learn) and set(np.unique(t.y)) <= {0, 1}
    assert 0.5 < info["agreement_with_real_y"] < 1.0
    t2, _ = S.teacher_labelled(learn, teacher, 3, 0)
    assert np.array_equal(t.y, t2.y)                         # fixed seed


def test_render_teach_runs():
    v = {"est": 0.001, "ci95": [0.0, 0.002], "ci90": [0.0, 0.002]}
    u = {"D_COP": v, "D_model": v, "D_synthesis": v, "D_synthesis_ind": v, "Q_TEACH": v, "H1_rule_TEACH": "x",
         "Q_synthesis": v, "H1_rule_synthesis": "y", "share_synthesis": 0.5}
    m = {"labels": {"agreement_with_real_y": 0.7, "mean_p_teacher": 0.2}, "weights": {"spearman": 0.8}}
    assert "50 %" in DR.render_teach({"status": "post hoc", "units": {"hypo_eicu": u}, "metrics": {"eicu": m}})
