"""v0.6 fix stream B (AUDIT 4, 13, 14 and the low-severity extraction items): KDIGO boundaries with tolerance, float
ids downstream of the filter, medication interval end, GCS by distinct components, deterministic last value, empty
inputs, null hadm_id, INSPIRE limitation values and the stay-level metamorphic leak check."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, conform
from perisim.targets.build import build_targets

T0 = 100000.0
CUT = T0 - 120
D = 1440.0


def _stays(ids, **extra):
    n = len(ids)
    d = {"stay_id": list(ids), "subject_id": [f"P{i}" for i in ids],
         "icu_in_min": [T0 - 3 * D] * n, "surgery_start_min": [T0 - 3.5 * D] * n,
         "icu_out_min": [T0] * n, "hosp_disch_min": [T0 + 20 * D] * n, "death_min": [None] * n,
         "discharge_location": ["home"] * n}
    d.update(extra)
    return pl.DataFrame(d, schema_overrides={"death_min": pl.Float64})


def _inp(rows):
    df = pl.DataFrame(rows, schema=["stay_id", "variable", "value", "time_min"], orient="row")
    return conform(df, INPUTS_SCHEMA)


# --------------------------------------------------------------------------- 1. KDIGO boundaries (AUDIT 4)
CASES = {   # stay: (baseline, post value, time after t0, expected stage)
    "abs_09_12": (0.9, 1.2, 1 * D, 1),       # +0.3 exactly within 48 h (1.2 − 0.9 = 0.29999… in floats)
    "abs_21_24": (2.1, 2.4, 1 * D, 1),
    "abs_08_11": (0.8, 1.1, 1 * D, 1),
    "abs_below": (0.9, 1.19, 1 * D, 0),      # just below +0.3
    "x3_11_33": (1.1, 3.3, 1 * D, 3),        # ×3 exactly (3.3 / 1.1 = 2.9999…)
    "x3_below": (1.1, 3.29, 1 * D, 2),
    "x15_22_33": (2.2, 3.3, 4 * D, 1),       # ×1.5 exactly, outside the 48-h window (ratio rule only)
    "x15_below": (2.2, 3.29, 4 * D, 0),
}


def test_kdigo_boundaries_count(cfg):
    rows = []
    for sid, (b, v, dt, _) in CASES.items():
        rows += [(sid, "lab_creatinine", b, CUT - 60), (sid, "lab_creatinine", v, T0 + dt)]
    tg, _, _ = build_targets(_stays(CASES), _inp(rows), conform(pl.DataFrame(), OUTCOMES_SCHEMA), cfg.targets,
                             "inspire", 2.0, 6.0)
    got = dict(zip(tg["stay_id"], tg["aki_stage"]))
    assert got == {k: c[3] for k, c in CASES.items()}


# --------------------------------------------------------------------------- MIMIC synthetic fixtures
def _gz(df: pl.DataFrame, path: Path) -> None:
    with gzip.open(path, "wb") as fh:
        df.write_csv(fh)


@pytest.fixture(scope="module")
def mimic_csv(tmp_path_factory, cfg):
    from perisim.cohort.select import select_index_stays
    from perisim.data.mimic import MimicExtractor
    from perisim.data.synthetic import synth_mimic

    root = tmp_path_factory.mktemp("mimic_csv")
    synth_mimic(root, 120, np.random.default_rng(5))
    src = root / "mimic-iv" / "3.1"
    ex = MimicExtractor(cfg, src, root / "cache")
    cand = ex.extract_stays()
    idx, _ = select_index_stays(cand, cfg.cohorts, "mimic")
    inp, out = ex.extract_events(idx)
    return src, cand, idx, inp, out


def _by_var(df: pl.DataFrame) -> dict:
    return dict(df.group_by(pl.col("variable").str.split("_").list.first()).len().iter_rows())


def test_float_ids_match_downstream(tmp_path, cfg, mimic_csv):
    """AUDIT 14: Float64 itemid / icd_version / hospital_expire_flag (e.g. a pandas parquet export) give the same
    stays, inputs and outcomes as the CSV text ids."""
    from perisim.data.mimic import MimicExtractor

    src, cand, idx, inp, out = mimic_csv
    dst = tmp_path / "3.1"
    shutil.copytree(src, dst)
    floats = {"hosp/labevents": ["itemid"], "icu/chartevents": ["itemid"], "icu/inputevents": ["itemid"],
              "icu/procedureevents": ["itemid"], "hosp/procedures_icd": ["icd_version"],
              "hosp/admissions": ["hospital_expire_flag"]}
    for t, cols in floats.items():
        df = pl.read_csv(dst / f"{t}.csv.gz", infer_schema_length=0)
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in cols])
        (dst / f"{t}.csv.gz").unlink()
        df.write_parquet(dst / f"{t}.parquet")
    ex = MimicExtractor(cfg, dst, tmp_path / "cache")
    cand2 = ex.extract_stays()
    assert cand2.sort("stay_id").equals(cand.sort("stay_id"))     # pcs codes (icd_version) and deaths (expire flag)
    inp2, out2 = ex.extract_events(idx)
    assert _by_var(inp2) == _by_var(inp)
    for v in ("lab", "vital", "support"):
        assert _by_var(inp).get(v, 0) > 0
    assert out2.height == out.height


def test_mimic_gcs_needs_distinct_components(tmp_path, cfg, mimic_csv):
    """Low-severity: a repeated eye component must not stand in for a missing verbal one (pl.len() did)."""
    from perisim.data.mimic import MimicExtractor

    src, _, idx, inp, _ = mimic_csv
    dst = tmp_path / "3.1"
    shutil.copytree(src, dst)
    ce = pl.read_csv(dst / "icu/chartevents.csv.gz", infer_schema_length=0)
    g = ce.filter(pl.col("itemid").is_in(["220739", "223900", "223901"]))
    keys = (g.group_by(["hadm_id", "charttime"]).len().filter(pl.col("len") == 3).sort(["hadm_id", "charttime"]))
    keys = keys.join(pl.DataFrame({"hadm_id": [h.split("-", 1)[1] for h in idx["hadm_id"].to_list()]}),
                     on="hadm_id").head(5)
    assert keys.height
    sel = pl.struct("hadm_id", "charttime").is_in(keys.select(pl.struct("hadm_id", "charttime")).to_series().implode())
    eye = ce.filter(sel & (pl.col("itemid") == "220739"))
    ce2 = pl.concat([ce.filter(~(sel & (pl.col("itemid") == "223900"))), eye])   # verbal replaced by a 2nd eye
    _gz(ce2, dst / "icu/chartevents.csv.gz")
    inp2, _ = MimicExtractor(cfg, dst, tmp_path / "cache").extract_events(idx)
    n1 = inp.filter(pl.col("variable") == "vital_gcs").height
    n2 = inp2.filter(pl.col("variable") == "vital_gcs").height
    assert n1 > n2 and n1 - n2 <= keys.height                    # the incomplete sets give no GCS (if in the window)


def test_mimic_events_empty_inputs(cfg, mimic_csv, tmp_path):
    """pl.concat([]) guard: stays without any event give empty frames with the common schema."""
    from perisim.data.mimic import MimicExtractor

    src, _, idx, _, _ = mimic_csv
    ghost = idx.head(1).with_columns(pl.lit("MIMIC-ghost-subject").alias("subject_id"),
                                     pl.lit("MIMIC-ghost-hadm").alias("hadm_id"))
    inp, out = MimicExtractor(cfg, src, tmp_path / "cache").extract_events(ghost)
    assert inp.height == 0 and inp.schema == pl.Schema(INPUTS_SCHEMA)
    assert out.schema == pl.Schema(OUTCOMES_SCHEMA)


# --------------------------------------------------------------------------- INSPIRE
@pytest.fixture(scope="module")
def inspire_src(tmp_path_factory):
    from perisim.data.synthetic import synth_inspire

    root = tmp_path_factory.mktemp("inspire")
    synth_inspire(root, 80, np.random.default_rng(7))
    return root / "inspire"


def test_inspire_gcs_duplicate_component_not_summed(tmp_path, cfg, inspire_src):
    from perisim.cohort.select import select_index_stays
    from perisim.data.inspire import InspireExtractor

    dst = tmp_path / "inspire"
    shutil.copytree(inspire_src, dst)
    ex = InspireExtractor(cfg, dst, tmp_path / "c1")
    idx, _ = select_index_stays(ex.extract_stays(), cfg.cohorts, "inspire")
    inp, _ = ex.extract_events(idx)
    wv = pl.read_csv(dst / "ward_vitals.csv", infer_schema_length=0)
    dup = wv.filter(pl.col("item_name") == "gcs_e")
    pl.concat([wv, dup]).write_csv(dst / "ward_vitals.csv")       # every eye component charted twice
    inp2, _ = InspireExtractor(cfg, dst, tmp_path / "c2").extract_events(idx)
    g1 = inp.filter(pl.col("variable") == "vital_gcs").sort(["stay_id", "time_min"])
    g2 = inp2.filter(pl.col("variable") == "vital_gcs").sort(["stay_id", "time_min"])
    assert g1.height > 0 and g1.equals(g2)


def test_inspire_limitation_uses_its_own_values(tmp_path, cfg, inspire_src):
    from perisim.data.inspire import InspireExtractor

    dst = tmp_path / "inspire"
    shutil.copytree(inspire_src, dst)
    ops = pl.read_csv(dst / "operations.csv", infer_schema_length=0)
    ops.with_columns(pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.lit("1")).otherwise(pl.lit("DNR"))
                     .alias("lim_col")).write_csv(dst / "operations.csv")
    opt = {"sources": {"inspire": {"tables": {"operations": {"optional_columns": {"limitation": "lim_col"}}}}}}
    c0 = cfg.with_overrides({"harmonization": opt})
    st = InspireExtractor(c0, dst, tmp_path / "c0").extract_stays()
    assert not st["limitation_known"].any()                      # no values configured: nothing matches
    c1 = cfg.with_overrides({"harmonization": {**opt, "sources": {"inspire": {
        **opt["sources"]["inspire"], "values": {"limitation_true": ["DNR"]}}}}})
    st1 = InspireExtractor(c1, dst, tmp_path / "c1").extract_stays()
    assert 0 < st1["limitation_known"].sum() < st1.height        # "1" (a donor value) no longer counts


# --------------------------------------------------------------------------- cohort
def test_null_hadm_is_its_own_hospitalization(cfg):
    from perisim.cohort.select import select_index_stays

    n = 4
    st = pl.DataFrame({
        "stay_id": ["A", "B", "C", "D"], "subject_id": ["p1", "p2", "p3", "p3"],
        "hadm_id": [None, None, "h3", "h3"], "icu_in_min": [0.0, 10.0, 0.0, 5000.0],
        "icu_out_min": [3000.0, 3010.0, 3000.0, 8000.0], "age": [60.0] * n, "postop": [True] * n,
        "died_in_icu": [False] * n, "icu_to_ward": [True] * n, "return_within_min": [None] * n,
        "organ_donor": [False] * n, "limitation_known": [False] * n, "limitation_min": [None] * n,
    }, schema_overrides={"return_within_min": pl.Float64, "limitation_min": pl.Float64})
    idx, _ = select_index_stays(st, cfg.cohorts, "inspire")
    assert idx["stay_id"].to_list() == ["A", "B", "C"]


# --------------------------------------------------------------------------- features
def _fstays(n=1):
    return pl.DataFrame({
        "stay_id": [f"S{i}" for i in range(n)], "icu_in_min": [T0 - 3 * D] * n, "icu_out_min": [T0] * n,
        "surgery_start_min": [T0 - 3.5 * D] * n, "surgery_end_min": [T0 - 3.4 * D] * n, "age": [60.0] * n,
        "sex_male": [1.0] * n, "asa": [2.0] * n, "emergency": [0.0] * n, "anesthesia_type": ["GENERAL"] * n,
        "surgical_group": ["gs"] * n, "admit_source": ["home"] * n, "pcs_code": ["0DTJ4ZZ"] * n, "bmi": [25.0] * n,
        "hosp_disch_min": [T0 + 5 * D] * n, "death_min": [None] * n,
    }, schema_overrides={"death_min": pl.Float64})


def test_medication_end_before_start_kept(cfg):
    """AUDIT 13: eICU drugstopoffset = 0 makes end < start; the record counts from its start."""
    from perisim.features.build import build_features

    t = CUT - 300
    rows = pl.DataFrame({"stay_id": ["S0"] * 3, "variable": ["atc:C03CA01", "atc:N02AB03", "src:medication"],
                         "value": [1.0] * 3, "time_min": [t, t, t], "end_min": [t + 600, T0 - 3 * D, None]})
    X, _, _ = build_features(_fstays(), conform(rows, INPUTS_SCHEMA), cfg, "eicu")
    assert X["med_C03C"][0] == 1.0 and X["med_N02A"][0] == 1.0


def test_last_value_ties_deterministic():
    from perisim.features.build import _last

    n = 2000
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"stay_id": np.repeat([f"s{i}" for i in range(n)], 2), "variable": ["lab_wbc"] * (2 * n),
                       "value": np.tile([1.0, 2.0], n), "time_min": np.repeat(rng.integers(0, 1000, n).astype(float), 2)})
    res = {tuple(_last(df.sample(fraction=1.0, shuffle=True, seed=k), "lab_", "last_").sort("stay_id")["value"])
           for k in range(5)}
    assert len(res) == 1 and set(next(iter(res))) == {2.0}


def test_build_features_empty_inputs(cfg):
    from perisim.features.build import build_features

    X, _, prov = build_features(_fstays(2), conform(pl.DataFrame(), INPUTS_SCHEMA), cfg, "inspire")
    assert X.height == 2 and prov.height == 2


def test_metamorphic_check_covers_stay_level_post_t0(cfg, monkeypatch):
    import perisim.features.build as fb

    st = _fstays(3)
    rows = conform(pl.DataFrame({"stay_id": ["S0"], "variable": ["lab_wbc"], "value": [9.0], "time_min": [CUT - 60]}),
                   INPUTS_SCHEMA)
    assert fb.metamorphic_leak_check(st, rows, cfg, "inspire") == []
    tr = fb.truncate_stays_at_cutoff(st, 2.0)
    assert tr["hosp_disch_min"].is_null().all() and tr["surgery_start_min"].equals(st["surgery_start_min"])

    real = fb.build_features

    def leaky(stays, inputs, cfg_, cohort):       # a feature built from the hospital discharge (post-t0)
        X, specs, prov = real(stays, inputs, cfg_, cohort)
        los = stays.select("stay_id", (pl.col("hosp_disch_min") - pl.col("icu_out_min")).alias("age"))
        return X.drop("age").join(los, on="stay_id", how="left"), specs, prov

    monkeypatch.setattr(fb, "build_features", leaky)
    assert fb.metamorphic_leak_check(st, rows, cfg, "inspire") == ["age"]
