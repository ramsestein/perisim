"""MIMIC-IV prescriptions where a hospitalization has no eMAR (lead, 2026-10-24) and the antibiotic course over order
intervals."""

import polars as pl

from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, conform
from perisim.targets.build import build_targets

T0 = 100000.0


def _stays(n):
    return pl.DataFrame({"stay_id": [f"S{i}" for i in range(n)], "subject_id": [f"P{i}" for i in range(n)],
                         "icu_in_min": [T0 - 3 * 1440] * n, "surgery_start_min": [T0 - 3.5 * 1440] * n,
                         "icu_out_min": [T0] * n, "hosp_disch_min": [T0 + 20 * 1440] * n, "death_min": [None] * n,
                         "discharge_location": ["home"] * n}, schema_overrides={"death_min": pl.Float64})


def _inp(rows):
    df = pl.DataFrame(rows, schema=["stay_id", "variable", "value", "time_min", "end_min"], orient="row")
    return conform(df, INPUTS_SCHEMA)


def test_route_filter_on_the_real_mimic_route_codes(cfg):
    """Route codes of the prescriptions inventory (schema_mimic, 2026-10-24)."""
    import re

    rx = re.compile(cfg.harmonization["sources"]["mimic"]["prescriptions_route_exclude_regex"])
    local = ["TP", "IH", "TD", "BOTH EYES", "RIGHT EYE", "LEFT EYE", "DWELL", "NU", "NEB", "ID", "LOCK", "INHALATION",
             "OU", "VG", "INTRAPLEURAL", "NAS", "OD", "BOTH EARS", "FLUSH", "OS", "LEFT EAR", "RIGHT EAR", "AD", "IRR",
             "TRANSDERMAL", "INTRAVITREAL", "AU", "INTERSPACE", "INTRALESIONAL", "AS", "SUBDERMAL", "PERIPHNERVE",
             "LUMBAR PLEXUS"]
    systemic = ["IV", "PO/NG", "PO", "SC", "IV DRIP", "IM", "PR", "ORAL", "IVPCA", "NG", "IV BOLUS", "SL", "DIALYS",
                "REPLACE", "IT", "IP", "IV INFUSION", "G TUBE", "J TUBE", "SUBCUT", "BU", "BUCCAL", "IO", "NG/OG", "OG",
                "SC INFUSION", "IA", "IVCI", "INTRA-ARTERIALLY"]
    assert [r for r in local if not rx.search(r)] == []
    assert [r for r in systemic if rx.search(r)] == []


def test_antibiotic_course_from_order_intervals(cfg):
    day = 1440
    rows = [(f"S{i}", "src:medication", 1.0, T0 - 2000, None) for i in range(4)]
    rows += [("S0", "atc:J01CR05", 1.0, T0 + 1.5 * day, T0 + 4.5 * day)]          # one 3-day order after t0 + 24 h
    rows += [("S1", "atc:J01CR05", 1.0, T0 + 1.5 * day, T0 + 1.5 * day + 600)]   # a 10-hour order: not 2 days
    rows += [("S2", "atc:J01DB04", 1.0, T0 - day, T0 + 2 * day),                 # still running from before t0 …
             ("S2", "atc:J01CR05", 1.0, T0 + 1.5 * day, T0 + 4.5 * day)]         # … so the switch is not a new course
    rows += [("S3", "atc:J01CR05", 1.0, T0 + 1.5 * day + h * 480, None) for h in range(7)]   # eMAR points, as before
    tg, _, _ = build_targets(_stays(4), _inp(rows), conform(pl.DataFrame(), OUTCOMES_SCHEMA), cfg.targets, "mimic",
                             2.0, 6.0, med_coverage_tracked=True)
    d = {r["stay_id"]: r["antibiotic_course"] for r in tg.to_dicts()}
    assert d == {"S0": 1, "S1": 0, "S2": 0, "S3": 1}


def test_mimic_prescriptions_cover_the_hospitalizations_without_emar(tmp_path, cfg):
    import numpy as np

    from perisim.cohort.select import select_index_stays
    from perisim.data.mimic import MimicExtractor
    from perisim.data.synthetic import synth_mimic

    synth_mimic(tmp_path, 300, np.random.default_rng(5))
    ex = MimicExtractor(cfg, tmp_path / "mimic-iv" / "3.1", tmp_path / "cache")
    idx, _ = select_index_stays(ex.extract_stays(), cfg.cohorts, "mimic")
    inp, _ = ex.extract_events(idx)
    covered = set(inp.filter(pl.col("variable") == "src:medication")["stay_id"].to_list())
    j01 = inp.filter(pl.col("variable").str.starts_with("atc:J01"))
    assert j01["end_min"].is_not_null().any()               # order intervals reach the inputs
    assert len(covered) > 0.8 * idx.height                   # eMAR + prescriptions cover most stays
    assert inp.filter(pl.col("variable") == "atc:J01XA01").height == 0     # vancomycin lock only: route excluded
