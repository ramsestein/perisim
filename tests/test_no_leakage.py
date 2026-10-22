"""§4.4.1 — Ningún input puede tener marca temporal > t0 − 2 h. Se comprueba fila a fila.

Con PERISIM_TEST_WORKDIR (en el pod) se comprueban las matrices reales de todas las cohortes; si no, se
comprueba la construcción sobre datos sintéticos, incluida una fila "trampa" posterior al corte.
"""

from __future__ import annotations

import polars as pl
import pytest

from perisim.features.build import build_features, check_no_leakage


def test_real_feature_matrices(real_workdir):
    if real_workdir is None:
        pytest.skip("sin workdir real")
    wd, variant = real_workdir
    provs = list((wd / "derived" / variant).glob("*/feature_provenance.parquet"))
    assert provs, "no hay feature_provenance en el workdir"
    for p in provs:
        prov = pl.read_parquet(p)
        assert prov["cutoff_min"].null_count() == 0
        bad = check_no_leakage(prov)
        assert bad.height == 0, f"{p.parent.name}: {bad.height} filas con inputs posteriores al corte"


def test_synthetic_no_leakage(synthetic_inspire, cfg):
    s = synthetic_inspire
    X, specs, prov = build_features(s["stays"], s["inputs"], cfg, "inspire")
    assert X.height == s["stays"].height
    assert check_no_leakage(prov).height == 0


def test_trap_row_after_cutoff_is_ignored(synthetic_inspire, cfg):
    """Una analítica absurda 1 minuto después del corte no puede alterar la matriz."""
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
