"""§4.4.1 — No input may have a timestamp > t0 − 2 h. Checked row by row.

With PERISIM_TEST_WORKDIR (on the pod) the real matrices of all cohorts are checked; otherwise the
construction is checked on synthetic data, including a "trap" row after the cutoff.
"""

from __future__ import annotations

import polars as pl
import pytest

from perisim.features.build import build_features, check_no_leakage


def test_real_feature_matrices(real_workdir):
    if real_workdir is None:
        pytest.skip("no real workdir")
    wd, variant = real_workdir
    provs = list((wd / "derived" / variant).glob("*/feature_provenance.parquet"))
    assert provs, "no feature_provenance in the workdir"
    for p in provs:
        prov = pl.read_parquet(p)
        assert prov["cutoff_min"].null_count() == 0
        bad = check_no_leakage(prov)
        assert bad.height == 0, f"{p.parent.name}: {bad.height} rows with inputs after the cutoff"


def test_synthetic_no_leakage(synthetic_inspire, cfg):
    s = synthetic_inspire
    X, specs, prov = build_features(s["stays"], s["inputs"], cfg, "inspire")
    assert X.height == s["stays"].height
    assert check_no_leakage(prov).height == 0


def test_trap_row_after_cutoff_is_ignored(synthetic_inspire, cfg):
    """An absurd lab value 1 minute after the cutoff must not alter the matrix."""
    s = synthetic_inspire
    st = s["stays"].head(5)
    margin = float(cfg.experiment["time"]["cutoff_margin_hours"]) * 60
    trap = st.select("stay_id", pl.lit("lab_lactate").alias("variable"), pl.lit(29.0).alias("value"),
                     (pl.col("icu_out_min") - margin + 1).alias("time_min"), pl.lit(None, pl.Float64).alias("end_min"))
    ids = st["stay_id"].to_list()
    inp = pl.concat([s["inputs"].filter(pl.col("stay_id").is_in(ids)), trap], how="vertical_relaxed")
    X0, _, _ = build_features(st, s["inputs"].filter(pl.col("stay_id").is_in(ids)), cfg, "inspire")
    X1, _, prov = build_features(st, inp, cfg, "inspire")
    assert check_no_leakage(prov).height == 0
    assert X0["last_lactate"].to_list() == X1["last_lactate"].to_list()


def test_metamorphic_no_leakage_synthetic(synthetic_inspire, cfg):
    """AUDIT C1: X must not change when post-cutoff rows are injected — for EVERY input family."""
    from perisim.features.build import metamorphic_leak_check

    s = synthetic_inspire
    st = s["stays"].head(40)
    ids = st["stay_id"].to_list()
    margin = float(cfg.experiment["time"]["cutoff_margin_hours"]) * 60
    after = (pl.col("icu_out_min") - margin + 30).alias("time_min")
    inj = pl.concat([
        st.select("stay_id", pl.lit(v).alias("variable"), pl.lit(x).alias("value"), after,
                  pl.lit(None, pl.Float64).alias("end_min"))
        for v, x in [("lab_lactate", 25.0), ("vital_hr", 190.0), ("support_vaso", 1.0), ("support_vent", 1.0),
                     ("atc:C01CA03", 1.0), ("atc:J01CR05", 1.0), ("dx:R57", 1.0), ("anthro_weight", 300.0)]])
    inp = pl.concat([s["inputs"].filter(pl.col("stay_id").is_in(ids)), inj], how="vertical_relaxed")
    assert metamorphic_leak_check(st, inp, cfg, "inspire") == []


def test_real_metamorphic(real_workdir, cfg):
    if real_workdir is None:
        pytest.skip("no real workdir")
    from perisim.features.build import metamorphic_leak_check

    wd, variant = real_workdir
    for c in ("inspire", "mimic", "eicu"):
        cm = wd / "common" / c
        if not (cm / "stays_index.parquet").exists():
            continue
        st = pl.read_parquet(cm / "stays_index.parquet")
        sample = st.sample(min(500, st.height), seed=1)
        inp = pl.read_parquet(cm / "inputs_long.parquet").filter(pl.col("stay_id").is_in(sample["stay_id"].to_list()))
        assert metamorphic_leak_check(sample, inp, cfg, c) == [], c


def test_metamorphic_check_detects_a_leak(synthetic_inspire, cfg):
    """Positive control: a matrix built WITHOUT the cutoff must be flagged."""
    from perisim.features.build import build_features, metamorphic_leak_check

    s = synthetic_inspire
    st = s["stays"].head(40)
    inp = s["inputs"].filter(pl.col("stay_id").is_in(st["stay_id"].to_list()))
    leaky_cfg = cfg.with_overrides({"experiment": {"time": {"cutoff_margin_hours": -24 * 30}}})
    X_leaky, _, _ = build_features(st, inp, leaky_cfg, "inspire")      # "cutoff" 30 days after t0
    assert metamorphic_leak_check(st, inp, cfg, "inspire", X=X_leaky)
