"""§4.2 — Definiciones operativas: KDIGO, curso nuevo de antibiótico, reingreso, T2/T3 en el fold."""

from __future__ import annotations

import numpy as np
import polars as pl

from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, conform
from perisim.targets.build import build_targets, t2_categories, t2_thresholds, t3_labels


def _stays(n=4):
    return pl.DataFrame({"stay_id": [f"S{i}" for i in range(n)], "subject_id": [f"P{i}" for i in range(n)],
                         "icu_out_min": [10000.0] * n, "hosp_disch_min": [10000.0 + 10 * 1440] * n,
                         "death_min": [None] * n, "discharge_location": ["home"] * n},
                        schema_overrides={"death_min": pl.Float64})


def _inp(rows):
    return conform(pl.DataFrame(rows, schema=["stay_id", "variable", "value", "time_min"], orient="row"), INPUTS_SCHEMA)


def test_kdigo_and_antibiotic(cfg):
    t0 = 10000.0
    cut = t0 - 120
    rows = [("S0", "lab_creatinine", 1.0, cut - 60), ("S0", "lab_creatinine", 3.2, t0 + 3 * 1440),   # LRA 3
            ("S1", "lab_creatinine", 1.0, cut - 60), ("S1", "lab_creatinine", 1.6, t0 + 2 * 1440),   # LRA 1
            ("S2", "lab_creatinine", 1.0, cut - 60), ("S2", "lab_creatinine", 1.1, t0 + 2 * 1440)]   # sin LRA
    # S3: curso nuevo de J01 que empieza a t0 + 30 h y dura 2 días; S2: J01 continuado desde antes (no cuenta)
    rows += [("S3", "atc:J01CR05", 1.0, t0 + 30 * 60 + h * 60) for h in range(0, 48, 8)]
    rows += [("S2", "atc:J01DB04", 1.0, t0 - 60 + h * 60) for h in range(0, 72, 8)]
    tg, car, _ = build_targets(_stays(), _inp(rows), conform(pl.DataFrame(), OUTCOMES_SCHEMA), cfg.targets, "inspire",
                               2.0, 6.0)
    d = {r["stay_id"]: r for r in tg.to_dicts()}
    assert d["S0"]["aki_stage"] == 3 and d["S0"]["t1"] == 2
    assert d["S1"]["aki_stage"] == 1 and d["S1"]["t1"] == 1
    assert d["S2"]["aki_stage"] == 0 and d["S2"]["antibiotic_course"] == 0
    assert d["S3"]["antibiotic_course"] == 1 and d["S3"]["t1"] == 1


def test_readmission_and_reduced_cohort(cfg):
    t0 = 10000.0
    outc = conform(pl.DataFrame([("S0", "icu_admission", t0 + 2880, "minute"), ("S1", "icu_admission", t0 - 3000, "minute")],
                                schema=["stay_id", "component", "time_min", "precision"], orient="row"), OUTCOMES_SCHEMA)
    tg, _, _ = build_targets(_stays(2), _inp([]), outc, cfg.targets, "eicu", 2.0, 6.0)
    d = {r["stay_id"]: r for r in tg.to_dicts()}
    assert d["S0"]["icu_readmission"] == 1 and d["S0"]["t1_grave"] == 1
    assert d["S1"]["icu_readmission"] == 0
    assert d["S0"]["t1"] is None                      # eICU: solo grave / no grave
    assert d["S0"]["aki_12"] is None                   # no observable en eICU


def test_t2_t3_thresholds_from_training_only(cfg):
    los = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], float)
    thr = t2_thresholds(los[:6], cfg.targets)
    cats = t2_categories(los, thr)
    assert cats[-1] == -1 and set(cats[:-1]) <= {0, 1, 2}
    t1 = np.array([0, 0, 0, 1, 2, 0, 0, 0, 0, 2])
    lab = t3_labels(t1, los, (2.5, 6.5))
    assert lab[0] == 0 and lab[2] == 1 and lab[3] == 2 and lab[4] == 3 and lab[7] == 2 and lab[9] == 3
