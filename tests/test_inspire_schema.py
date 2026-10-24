"""INSPIRE as the lead's copy ships it (schema_inspire, 2026-10-24): GCS by components, route column, one ATC
column."""

import polars as pl


def test_gcs_total_from_components(synthetic_inspire):
    g = synthetic_inspire["inputs"].filter(pl.col("variable") == "vital_gcs")
    assert g.height > 0
    assert g["value"].min() >= 3 and g["value"].max() <= 15
    assert (g["value"] == 15).all()          # E4 + M6 + V5 charted together in the synthetic data


def test_topical_epinephrine_is_neither_atc_input_nor_vasopressor(synthetic_inspire):
    s = synthetic_inspire
    inp = s["inputs"]
    assert inp.filter(pl.col("variable") == "atc:C01CA24").height == 0      # route "ex" only, in the synthetic data
    vaso = (inp.filter(pl.col("variable") == "support_vaso")
            .join(s["stays"].select("stay_id", "icu_in_min"), on="stay_id"))
    assert vaso.filter((pl.col("time_min") - pl.col("icu_in_min") - 17).abs() < 1e-6).height == 0
    assert inp.filter(pl.col("variable") == "support_vaso").height > 0       # IV norepinephrine still counts


def test_schema_accepts_a_single_atc_column(tmp_path, cfg):
    import numpy as np

    from perisim.data.discover import schema_report
    from perisim.data.synthetic import synth_inspire

    synth_inspire(tmp_path, 40, np.random.default_rng(1))
    rep = schema_report(cfg, "inspire", tmp_path / "inspire", tmp_path / "cache", 10)
    assert rep["ok"], rep["tables"]["medications"]
    assert set(rep["tables"]["medications"]["optional_missing"]) == {"atc_code2", "atc_code3"}
