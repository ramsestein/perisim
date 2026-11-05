"""INSPIRE as the lead's copy ships it (schema_inspire, 2026-10-24): GCS by components, route column, one ATC
column; the lead's answers to OPEN_QUESTIONS 32-34 (CTS split by ICD-10-PCS, MAP from SBP / DBP)."""

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


def test_cts_is_split_by_icd10_pcs(synthetic_inspire):
    st = synthetic_inspire["stays"]
    cardiac = st.filter(pl.col("surgical_group") == "cardiac")
    lung = st.filter(pl.col("pcs_code") == "0BTC0")            # CTS operations with a respiratory-system code
    assert cardiac.height > 0 and lung.height > 0
    assert cardiac["pcs_code"].str.starts_with("02").all()
    assert (lung["surgical_group"] == "thoracic").all()


def test_map_is_derived_where_only_sbp_and_dbp_are_charted(synthetic_inspire):
    s = synthetic_inspire
    m = (s["inputs"].filter(pl.col("variable") == "vital_map")
         .join(s["stays"].select("stay_id", "icu_in_min"), on="stay_id")
         .with_columns((pl.col("time_min") - pl.col("icu_in_min")).alias("dt")))
    charted, derived = m.filter(pl.col("dt") == 0), m.filter(pl.col("dt") == 120)   # MAP charted at 0 h, not at 2 h
    assert derived.height > 0
    assert derived["value"].is_between(60, 100).all()          # (≈120 + 2·≈60) / 3
    assert charted.group_by("stay_id").len()["len"].max() == 1   # a charted MAP is never duplicated by a derived one
