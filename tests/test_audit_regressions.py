"""Regression tests for findings of docs/AUDIT.md."""

from __future__ import annotations

import numpy as np
import polars as pl

from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, conform, map_measurements, variables_for


def test_b1_consensus_does_not_collapse_when_a_model_ignores_a_group():
    from perisim.importance.sage import aggregate

    imp = np.array([[0.4, 0.17, 0.43]] * 4 + [[0.5, 0.0, 0.5]])
    se = np.array([[0.01, 0.01, 0.01]] * 4 + [[0.01, 0.0, 0.01]])
    agg = aggregate(imp, se)
    assert agg[1] > 0.1                                   # was ~1e-4 with plain inverse-variance weights
    homog = np.array([[0.5, 0.3, 0.2]] * 5)
    np.testing.assert_allclose(aggregate(homog, np.full((5, 3), 0.01)), [0.5, 0.3, 0.2], atol=1e-9)


def test_d2_fahrenheit_temperatures_are_kept(cfg):
    specs = [s for s in variables_for(cfg.harmonization, "mimic", "chartevents") if "223761" in s["items"]]
    df = pl.DataFrame({"itemid": ["223761", "223761", "223761"], "valuenum": ["98.6", "101.3", "30"]})
    out = map_measurements(df, specs, "itemid", "valuenum")
    np.testing.assert_allclose(sorted(out["value"].to_list()), [37.0, 38.5], atol=0.01)   # 30 °F is implausible


def test_d9_death_after_day_30_stays_in_t2(cfg):
    from perisim.targets.build import build_targets

    t0 = 10000.0
    stays = pl.DataFrame({"stay_id": ["S0", "S1"], "subject_id": ["P0", "P1"], "icu_out_min": [t0, t0],
                          "hosp_disch_min": [t0 + 40 * 1440, t0 + 5 * 1440],
                          "death_min": [t0 + 40 * 1440, t0 + 5 * 1440], "discharge_location": ["death", "death"]})
    tg, _, _ = build_targets(stays, conform(pl.DataFrame(), INPUTS_SCHEMA), conform(pl.DataFrame(), OUTCOMES_SCHEMA),
                             cfg.targets, "inspire", 2.0, 6.0)
    d = {r["stay_id"]: r for r in tg.to_dicts()}
    assert d["S0"]["los_post_icu_days"] == 30.0 and d["S0"]["death"] == 0     # alive at day 30 → top category
    assert d["S1"]["los_post_icu_days"] is None and d["S1"]["death"] == 1     # died within 30 d → not in T2


def test_e5_swift_gcs_14_scores_6():
    from perisim.baselines.parametric import swift_score

    s = swift_score([24, 24, 24], [500, 500, 500], [15, 14, 11], [40, 40, 40], ["direct"] * 3)
    assert list(s) == [0, 6, 6]


def test_e4_binary_brier_scale():
    from perisim.evaluation.metrics import row_brier_binary

    np.testing.assert_allclose(row_brier_binary(np.array([0.8, 0.3]), np.array([1, 0])), [0.04, 0.09])
