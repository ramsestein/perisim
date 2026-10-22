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
