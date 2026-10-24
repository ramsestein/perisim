"""Scientific decisions of 2026-10-22 (docs/DECISIONS.md): cohort (D1/D8, C5, D12, D13), targets (D5, D6, D7,
D10), method (B2/B4, B6, B7), comparators and external validation (E13, C4, E14)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import polars as pl

from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, conform
from perisim.targets.build import build_targets

T0 = 100000.0
CUT = T0 - 120


def _stays(n, **extra):
    d = {"stay_id": [f"S{i}" for i in range(n)], "subject_id": [f"P{i}" for i in range(n)],
         "icu_in_min": [T0 - 3 * 1440] * n, "surgery_start_min": [T0 - 3.5 * 1440] * n,
         "icu_out_min": [T0] * n, "hosp_disch_min": [T0 + 20 * 1440] * n, "death_min": [None] * n,
         "discharge_location": ["home"] * n}
    d.update(extra)
    return pl.DataFrame(d, schema_overrides={"death_min": pl.Float64})


def _inp(rows):
    df = pl.DataFrame(rows, schema=["stay_id", "variable", "value", "time_min"], orient="row") if rows else \
        pl.DataFrame(schema={"stay_id": pl.Utf8, "variable": pl.Utf8, "value": pl.Float64, "time_min": pl.Float64})
    return conform(df, INPUTS_SCHEMA)


def _no_outcomes():
    return conform(pl.DataFrame(), OUTCOMES_SCHEMA)


def _cr(sid, pairs):
    return [(sid, "lab_creatinine", v, t) for t, v in pairs]


# --------------------------------------------------------------------------- targets
def test_kdigo_rolling_windows(cfg):
    d = 1440.0
    rows = []
    # S0: AKI in the ICU (baseline 2.0), ward nadir 1.0, then +0.4 within 48 h → stage 1 (the v0 rule saw nothing)
    rows += _cr("S0", [(CUT - 60, 2.0), (T0 + 2 * d, 1.0), (T0 + 5 * d, 1.0), (T0 + 6 * d, 1.4)])
    # S1: high baseline 6.0 → 6.4: the ≥ 4.0 mg/dL rule needs a baseline below 4.0 (D5) → stage 1, not 3
    rows += _cr("S1", [(CUT - 60, 6.0), (T0 + 1 * d, 6.4)])
    # S2: slow drift 1.0 → 1.6 over 10 days: ≥ 1.5 × baseline (the baseline is always a reference) → stage 1
    rows += _cr("S2", [(CUT - 60, 1.0), (T0 + 4 * d, 1.2), (T0 + 7 * d, 1.4), (T0 + 10 * d, 1.6)])
    # S3: 1.0 → 3.2 → stage 3 by ratio
    rows += _cr("S3", [(CUT - 60, 1.0), (T0 + 2 * d, 3.2)])
    # S4: late +0.3 relative to the baseline only (no recent nadir): not AKI
    rows += _cr("S4", [(CUT - 60, 1.0), (T0 + 8 * d, 1.25), (T0 + 10 * d, 1.3)])
    tg, _, _ = build_targets(_stays(5), _inp(rows), _no_outcomes(), cfg.targets, "inspire", 2.0, 6.0)
    st = dict(zip(tg["stay_id"], tg["aki_stage"]))
    assert st == {"S0": 1, "S1": 1, "S2": 1, "S3": 3, "S4": 0}


def test_kdigo_ignores_creatinine_during_rrt(cfg):
    rows = _cr("S0", [(CUT - 3000, 2.0), (CUT - 60, 0.8), (T0 + 1440, 2.1)])
    rows += [("S0", "support_rrt", 1.0, CUT - 300), ("S0", "support_rrt", 1.0, CUT - 200)]
    # surgery_start AFTER the RRT records would make it chronic; here RRT starts after the surgery (acute)
    tg, _, _ = build_targets(_stays(1), _inp(rows), _no_outcomes(), cfg.targets, "inspire", 2.0, 6.0)
    r = tg.row(0, named=True)
    assert r["chronic_rrt"] is False
    assert r["cr_base"] == 2.0           # the 0.8 measured within 24 h after the RRT run is not a baseline
    assert r["aki_stage"] == 0


def test_chronic_rrt_not_applicable(cfg):
    pre = T0 - 4 * 1440                  # before the index surgery
    rows = [("S0", "marker:chronic_rrt", 1.0, pre), ("S0", "support_rrt", 1.0, T0 + 2 * 1440)]
    rows += _cr("S0", [(CUT - 60, 6.0), (T0 + 1440, 6.5)])
    rows += [("S1", "support_rrt", 1.0, T0 + 2 * 1440)]                      # acute: new RRT after t0 = severe
    tg, _, _ = build_targets(_stays(2), _inp(rows), _no_outcomes(), cfg.targets, "inspire", 2.0, 6.0)
    d = {r["stay_id"]: r for r in tg.to_dicts()}
    assert d["S0"]["chronic_rrt"] and d["S0"]["rrt_start"] is None and d["S0"]["aki_3"] is None
    assert d["S0"]["t1"] == 0            # the remaining components decide (not applicable ≠ unobserved)
    assert d["S1"]["rrt_start"] == 1 and d["S1"]["t1"] == 2


def test_medication_coverage_makes_t1_undefined(cfg):
    rows = [("S0", "src:medication", 1.0, T0 - 2000)]
    rows += [("S0", "atc:J01CR05", 1.0, T0 + 30 * 60 + h * 60) for h in range(0, 48, 8)]
    rows += [("S1", "atc:J01CR05", 1.0, T0 + 30 * 60 + h * 60) for h in range(0, 48, 8)]   # no coverage marker
    outc = conform(pl.DataFrame([("S1", "icu_admission", T0 + 3000, "minute")],
                                schema=["stay_id", "component", "time_min", "precision"], orient="row"), OUTCOMES_SCHEMA)
    tg, _, diag = build_targets(_stays(2), _inp(rows), outc, cfg.targets, "mimic", 2.0, 6.0, med_coverage_tracked=True)
    d = {r["stay_id"]: r for r in tg.to_dicts()}
    assert d["S0"]["antibiotic_course"] == 1 and d["S0"]["t1"] == 1
    assert d["S1"]["antibiotic_course"] is None and d["S1"]["transfusion"] is None
    assert d["S1"]["t1"] is None                     # undefined whatever the outcome (coverage-based exclusion)
    assert d["S1"]["t1_severe"] == 1                 # severe / not severe stays defined
    assert diag["n_medication_unobserved"] == 1


def test_inspire_readmission_proxy(cfg):
    rows = [("S0", "support_vent", 1.0, T0 + 2 * 1440 + h * 60) for h in range(5)]
    rows += [("S1", "support_ecmo", 1.0, T0 + 3 * 1440)]
    for cohort, expected in (("inspire", {"S0": 1, "S1": 1}), ("mimic", {"S0": 0, "S1": 0})):
        tg, _, _ = build_targets(_stays(2), _inp(rows), _no_outcomes(), cfg.targets, cohort, 2.0, 6.0)
        assert dict(zip(tg["stay_id"], tg["icu_readmission"])) == expected
    tg, _, _ = build_targets(_stays(2), _inp(rows), _no_outcomes(), cfg.targets, "inspire", 2.0, 6.0)
    assert tg["icu_readmission_proxy"].to_list() == [1, 1] and tg["icu_readmission_icu_entry"].to_list() == [0, 0]


# --------------------------------------------------------------------------- cohort
def test_return_and_timestamped_limitation(cfg):
    from perisim.cohort.select import select_index_stays

    n = 4
    st = pl.DataFrame({"stay_id": [f"S{i}" for i in range(n)], "subject_id": [f"P{i}" for i in range(n)],
                       "hadm_id": [f"H{i}" for i in range(n)], "icu_in_min": [0.0] * n, "icu_out_min": [3000.0] * n,
                       "age": [60.0] * n, "postop": [True] * n, "died_in_icu": [False] * n, "icu_to_ward": [True] * n,
                       "organ_donor": [False] * n, "limitation_known": [None] * n,
                       "limitation_min": [None, 2000.0, 2950.0, None], "return_within_min": [None, None, None, 120.0]},
                      schema_overrides={"limitation_known": pl.Boolean, "limitation_min": pl.Float64,
                                        "return_within_min": pl.Float64})
    idx, flow = select_index_stays(st, cfg.cohorts, "mimic", margin_hours=2.0)
    # S1: DNR recorded before the cutoff → excluded; S2: recorded after the cutoff → kept; S3: back to the OR in 2 h
    assert sorted(idx["stay_id"].to_list()) == ["S0", "S2"]
    steps = [f["step"] for f in flow]
    assert any("OR / PACU" in s for s in steps)


def _w(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        df.write_csv(fh)


def test_mimic_or_evidence_and_reoperation(tmp_path, cfg):
    """D1/D8: PACU transfer = surgery; codes only without OR/PACU transfers; a surgical service alone is not
    postoperative; reoperation from a later PACU transfer has minute precision; D12 Code Status; C5."""
    from perisim.data.mimic import MimicExtractor

    d = tmp_path / "mimic"
    H = [11, 22, 33, 44]
    _w(pl.DataFrame({"subject_id": [1, 2, 3, 4], "hadm_id": H, "admittime": ["2150-01-01 08:00:00"] * 4,
                     "dischtime": ["2150-01-15 12:00:00"] * 4, "deathtime": [None] * 4,
                     "admission_type": ["ELECTIVE"] * 4, "admission_location": ["PHYSICIAN REFERRAL"] * 4,
                     "discharge_location": ["HOME"] * 4, "hospital_expire_flag": [0] * 4}), d / "hosp/admissions.csv.gz")
    _w(pl.DataFrame({"subject_id": [1, 2, 3, 4], "gender": ["M"] * 4, "anchor_age": [60] * 4, "anchor_year": [2150] * 4}),
       d / "hosp/patients.csv.gz")
    _w(pl.DataFrame({"subject_id": [1, 2, 3, 4], "hadm_id": H, "stay_id": [101, 202, 303, 404],
                     "intime": ["2150-01-01 14:00:00"] * 4, "outtime": ["2150-01-03 10:00:00"] * 4,
                     "first_careunit": ["SICU"] * 4}), d / "icu/icustays.csv.gz")
    sicu = "Surgical Intensive Care Unit (SICU)"
    tr = []
    tid = 0
    for s, h in zip([1, 2, 3, 4], H):
        def add(unit, a, b):
            nonlocal tid
            tid += 1
            tr.append((s, h, tid, "transfer", unit, a, b))
        if h in (11, 44):   # PACU before the ICU
            add("Med/Surg", "2150-01-01 08:00:00", "2150-01-01 10:00:00")
            add("PACU", "2150-01-01 12:00:00", "2150-01-01 14:00:00")
        else:
            add("Med/Surg", "2150-01-01 08:00:00", "2150-01-01 14:00:00")
        add(sicu, "2150-01-01 14:00:00", "2150-01-03 10:00:00")
        if h == 44:         # C5: back to the PACU 2 h after the ICU exit
            add("PACU", "2150-01-03 12:00:00", "2150-01-03 13:00:00")
        add("Med/Surg", "2150-01-03 13:00:00", "2150-01-15 12:00:00")
        if h == 11:         # reoperation 4 days later (minute precision)
            add("PACU", "2150-01-07 09:30:00", "2150-01-07 11:00:00")
    _w(pl.DataFrame(tr, schema=["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"],
                    orient="row"), d / "hosp/transfers.csv.gz")
    _w(pl.DataFrame({"subject_id": [1, 2, 3, 4], "hadm_id": H, "transfertime": ["2150-01-01 08:00:00"] * 4,
                     "curr_service": ["SURG", "CSURG", "NSURG", "SURG"]}), d / "hosp/services.csv.gz")
    # hadm 22: no PACU, CABG code the day of the ICU admission → procedure fallback; hadm 33: service only
    _w(pl.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22], "seq_num": [1, 1], "chartdate": ["2150-01-01"] * 2,
                     "icd_code": ["0DTJ0ZZ", "02100Z9"], "icd_version": [10, 10]}), d / "hosp/procedures_icd.csv.gz")
    _w(pl.DataFrame({"hadm_id": [11, 22], "itemid": [223758, 223758], "charttime": ["2150-01-01 15:00:00"] * 2,
                     "value": ["Full code", "DNR / DNI"], "subject_id": [1, 2], "stay_id": [101, 202],
                     "valuenum": [None, None]}, schema_overrides={"valuenum": pl.Float64}), d / "icu/chartevents.csv.gz")
    ex = MimicExtractor(cfg, d, tmp_path / "cache")
    st = {r["hadm_id"]: r for r in ex.extract_stays().to_dicts()}
    assert st["MIMIC-11"]["postop_evidence"] == "or_transfer" and st["MIMIC-11"]["postop"]
    assert st["MIMIC-22"]["postop_evidence"] == "procedure" and st["MIMIC-22"]["postop"]
    assert st["MIMIC-33"]["postop_evidence"] == "service_only" and not st["MIMIC-33"]["postop"]
    assert st["MIMIC-44"]["return_within_min"] == 120.0
    assert st["MIMIC-22"]["limitation_min"] is not None and st["MIMIC-11"]["limitation_min"] is None
    reop = ex._or_transfers(["11"]).sort("t_in")
    assert reop.height == 2                          # index PACU + reoperation PACU (minute precision)


# --------------------------------------------------------------------------- encoding (B6, B7) and NCA (B2/B4)
def _ft(cat_vals, hier_vals):
    from perisim.features.build import FeatureSpec
    from perisim.features.table import FeatureTable

    n = len(cat_vals)
    specs = [FeatureSpec("grp", "nominal", "grp", "surgery"), FeatureSpec("pcs_l1", "hier", "pcs", "surgery")]
    ids = np.arange(n).astype(str).astype(object)
    return FeatureTable(ids, ids, specs, {}, {"grp": np.array(cat_vals, dtype=object),
                                              "pcs_l1": np.array(hier_vals, dtype=object)})


def test_unseen_category_gets_prior_and_rare_codes_mismatch():
    from perisim.metric.encoding import Encoder

    rng = np.random.default_rng(0)
    cats = list(rng.choice(["a", "b"], 200)) + ["rare"] * 3
    hier = list(rng.choice(["0", "B"], 200)) + ["X", "Y", None]
    y = rng.integers(0, 3, len(cats))
    enc = Encoder(m=5, nominal_min_count=10, hier_min_count=10).fit(_ft(cats, hier), y, 3)
    nom = next(b for b in enc.blocks if b.kind == "nominal")
    q = enc._raw_block(nom, _ft(["never_seen", "rare"], ["0", "0"]))
    np.testing.assert_allclose(q[0], nom.params["unseen"])           # unseen → P(y) (B7)
    np.testing.assert_allclose(q[1], nom.params["table"]["__other__"][0])   # seen but rare → pooled "other"
    hb = next(b for b in enc.blocks if b.kind == "hier")
    e = enc._raw_block(hb, _ft(["a"] * 4, ["0", "X", "Z", None]))
    d2 = lambda i, j: float(((e[i] - e[j]) ** 2).sum())  # noqa: E731
    assert abs(d2(0, 1) - 1.0) < 1e-9                                # kept vs rare: full mismatch (B6)
    assert d2(1, 2) == 0.0                                           # two rare codes: pooled "other" (declared)
    assert abs(d2(1, 3) - 1.0) < 1e-9                                # rare vs missing: mismatch


def test_nca_group_prior_and_free_scale():
    from perisim.features.build import FeatureSpec
    from perisim.features.table import FeatureTable
    from perisim.metric.encoding import Encoder
    from perisim.metric.nca import DiagonalNCA

    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    num = {"inf": x, "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)}
    specs = [FeatureSpec(k, "continuous", k, "icu") for k in num]
    ids = np.arange(n).astype(str).astype(object)
    enc = Encoder().fit(FeatureTable(ids, ids, specs, num, {}), y, 2)
    E = enc.transform(FeatureTable(ids, ids, specs, num, {}))
    groups = np.array([0, 0, 1])                    # "inf" shares a SAGE group with "noise_a"
    prior = np.array([0.25, 0.25, 0.5])             # the group totals are equal, the split inside group 0 is equal
    nca = DiagonalNCA(enc.dim_feature, enc.n_features, 2, n_candidates=60, batch_size=128, lr=0.1, seed=0)
    nca.fit(E, y, prior=prior, groups=groups, lam=50.0, epochs=30, subjects=ids)
    w = nca.weights
    assert w[0] > 3 * w[1]                          # the NCA splits WITHIN the group (B4)
    assert abs(np.log((w[0] + w[1]) / w[2])) < 0.5  # the group totals stay at the prior ratio (1 : 1)
    assert abs(nca.params_["logc"] - np.log(10.0)) > 0.05   # the global scale moves freely at large λ (B2)


# --------------------------------------------------------------------------- comparators and external validation
def test_lgbm_tuned_in_inner_folds():
    from perisim.baselines.parametric import fit_lgbm_calibrated

    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
    folds = [(np.arange(0, 200), np.arange(200, 300)), (np.arange(100, 300), np.arange(0, 100))]
    f, info = fit_lgbm_calibrated(X, y, {"min_child_samples": 10}, folds, 2, 0, 1,
                                  grid={"num_leaves": [3, 7], "learning_rate": [0.1], "max_estimators": 60,
                                        "early_stopping_rounds": 5})
    assert info["num_leaves"] in (3, 7) and 1 <= info["n_estimators"] <= 60
    p = f(X[:5])
    np.testing.assert_allclose(p.sum(1), 1.0)


def test_crossfit_labels_never_use_own_los(cfg):
    from perisim.experiment import CohortData, labels_for
    from perisim.external import patient_halves

    rng = np.random.default_rng(3)
    n = 120
    subj = np.array([f"P{i // 2}" for i in range(n)], dtype=object)
    half = patient_halves(subj, 7)
    assert all(len(set(half[subj == s])) == 1 for s in np.unique(subj))    # patient-level halves

    def labels(los):
        D = CohortData("x", type("F", (), {"__len__": lambda self: n})(), np.zeros(n, int), np.zeros(n, int), los,
                       {}, np.zeros(n), np.zeros(n))
        V = {h: labels_for("t2", D, np.where(half == h)[0], cfg.targets)[0] for h in (0, 1)}
        return np.where(half == 0, V[1], V[0])

    los = rng.exponential(5, n)
    base = labels(los)
    i = int(np.where(half == 0)[0][0])
    los2 = los.copy()
    los2[i] = 1000.0                                 # change one stay's own LOS
    after = labels(los2)
    same_half = (half == 0) & (np.arange(n) != i)
    assert (base[same_half] == after[same_half]).all()   # its half's labels use the OTHER half's cut-points
    assert after[i] == 2


def test_loo_prior_excludes_own_patient():
    from perisim.knn.predict import KNNModel

    E = np.random.default_rng(4).normal(size=(6, 2)).astype(np.float32)
    y = np.array([0, 0, 1, 1, 1, 1])
    subj = np.array(["a", "a", "b", "c", "d", "e"], dtype=object)
    kn = KNNModel(E, np.ones(2, dtype=np.float32), y, 2, subj, [2])
    pr = kn.loo_prior()
    np.testing.assert_allclose(pr[0], [0.0, 1.0])    # patient "a" (both class 0) left out
    np.testing.assert_allclose(pr[2], [0.4, 0.6])


def test_patient_output_records():
    from perisim.knn.predict import merge_patient_outputs, patient_pieces

    ids = np.array(["S1", "S2"], dtype=object)
    I = np.array([[0, 1], [1, 0]])
    D = np.array([[0.25, 1.0], [0.0, 4.0]])
    valid = np.ones_like(I, dtype=bool)
    p1 = patient_pieces(ids, np.array([1.0, 2.0]), "t1", np.array([[.6, .3, .1], [.2, .3, .5]]), 2,
                        kth_pct=np.array([10.0, 90.0]), ood=np.array([False, True]), I=I, D=D, valid=valid,
                        ref_ids=np.array(["R1", "R2"], dtype=object), carried={"death": np.array([0.1, np.nan])})
    p2 = patient_pieces(ids[:1], None, "t2", np.array([[.2, .5, .3]]), 2)
    rec = {r["patient_id"]: r for r in merge_patient_outputs([p1, p2])}
    assert rec["S1"]["p_T1"]["none"] == 0.6 and rec["S1"]["p_T2"]["medium"] == 0.5
    assert rec["S2"]["p_T2"] is None and rec["S2"]["ood"] is True
    assert rec["S1"]["neighbors"][0] == {"case_id": "R1", "distance": 0.5}
    assert rec["S2"]["carried_summary"]["death"] is None
