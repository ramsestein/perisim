"""Landmark builder of the `hypo` experiment: time rule (no leakage), outcome window, eligibility, supports."""

import polars as pl
import pytest

from hypo.config import load
from hypo.landmarks import build, hourly, support_hours

CFG = load()


def _stays(**over):
    base = {"stay_id": ["A", "B", "C", "D"], "subject_id": ["a", "b", "c", "d"], "los_h": [30.0, 10.0, 9.5, 30.0],
            "death_h": [None, None, 9.0, None], "limitation_h": [None, None, None, 5.0],
            "surgery_end_to_icu_h": [1.0, 1.0, 1.0, 1.0], "hospital": ["h1", "h1", "h2", "h2"]}
    base.update(over)
    return pl.DataFrame(base, schema_overrides={"death_h": pl.Float64, "limitation_h": pl.Float64})


def _vitals():
    rows = []
    for s in ("A", "B", "C", "D"):
        for hr in range(-1, 40):
            rows.append((s, hr + 0.5, "map_nibp", 80.0))
            rows.append((s, hr + 0.5, "hr", 80.0))
    df = pl.DataFrame(rows, schema=["stay_id", "h", "var", "value"], orient="row")
    df = df.with_columns(
        pl.when((pl.col("stay_id") == "A") & (pl.col("var") == "map_nibp") & (pl.col("h") == 7.5)).then(60.0)
        .when((pl.col("stay_id") == "A") & (pl.col("var") == "hr") & (pl.col("h") == 6.5)).then(200.0)
        .when((pl.col("stay_id") == "B") & (pl.col("var") == "map_nibp") & (pl.col("h") == 11.5)).then(50.0)
        .otherwise(pl.col("value")).alias("value"))
    # stay D: one hour with three readings, the first < 65 but the median ≥ 65
    extra = pl.DataFrame([("D", 3.1, "map_nibp", 60.0), ("D", 3.6, "map_nibp", 70.0)],
                         schema=["stay_id", "h", "var", "value"], orient="row")
    return pl.concat([df.filter(~((pl.col("stay_id") == "D") & (pl.col("h") == 3.5) & (pl.col("var") == "map_nibp"))),
                      extra, pl.DataFrame([("D", 3.8, "map_nibp", 72.0)], schema=["stay_id", "h", "var", "value"],
                                          orient="row")])


def _labs():
    return pl.DataFrame([("A", -10.0, "creatinine", 1.0), ("A", 5.5, "creatinine", 2.0)],
                        schema=["stay_id", "h", "var", "value"], orient="row")


def _supports():
    return pl.DataFrame([("A", "vaso", 3.5, 3.5)], schema=["stay_id", "kind", "h", "h_end"], orient="row")


@pytest.fixture(scope="module")
def lm():
    return build(_stays(), hourly(_vitals()), _labs(), support_hours(_supports()), CFG)


def row(lm, s, L):
    r = lm.filter((pl.col("stay_id") == s) & (pl.col("L") == L))
    return r.row(0, named=True) if r.height else None


def test_inputs_never_see_hour_L_or_later(lm):
    r = row(lm, "A", 6)
    assert r["hr_last"] == 80.0                    # the 200 bpm of hour 6 is in the future at L = 6
    assert row(lm, "A", 8)["hr_last"] == 80.0       # hour 7: back to 80 (the history is hours 2–7)
    assert row(lm, "A", 8)["map_last"] == 60.0       # hour 7 is the last hour at L = 8
    r4 = row(lm, "A", 4)                            # the 5.5-h creatinine is in the future at L = 4
    assert r4["creatinine_last"] == 1.0 and r4["creatinine_age_h"] == pytest.approx(14.0)
    assert r["creatinine_last"] == 2.0 and r["creatinine_age_h"] == pytest.approx(0.5)


def test_outcome_window_is_L_to_L_plus_horizon(lm):
    assert row(lm, "A", 2)["y"] and row(lm, "A", 6)["y"]      # hour 7 inside [2, 8) and [6, 12)
    assert not row(lm, "A", 8)["normotensive"]               # hour 7 is its last hour: not the analysis population
    assert not row(lm, "A", 10)["y"]                         # [10, 16): no hypotension
    assert row(lm, "A", 2)["band"] == 0 and row(lm, "A", 10)["band"] == 2


def test_hours_after_icu_discharge_do_not_count(lm):
    r = row(lm, "B", 6)
    assert not r["y"] and r["icu_discharge_24h"]             # the 50 mmHg of hour 11 is after the discharge (10 h)
    assert row(lm, "B", 10) is None                          # not in the ICU at L = 10


def test_death_in_window_is_hypotension_and_dead_or_limited_stays_leave(lm):
    r = row(lm, "C", 4)
    assert r["died_in_window"] and r["y"] and r["known"] and r["band"] == 0
    assert row(lm, "C", 10) is None
    assert row(lm, "D", 4) is not None and row(lm, "D", 6) is None      # limitation of treatment at 5 h


def test_hourly_median_vs_first_reading(lm):
    r = row(lm, "D", 2)                                      # hour 3: readings 60, 70, 72 → median 70, first 60
    assert not r["y"] and r["y_first"]


def test_supports_and_slope(lm):
    assert row(lm, "A", 4)["vaso_on"] and row(lm, "A", 4)["vaso_hours"] == 1
    assert not row(lm, "A", 6)["vaso_on"]                    # charted at hour 3: older than the 2-h lookback
    assert row(lm, "A", 2)["vaso_start"] and not row(lm, "A", 4)["vaso_start"]
    v = pl.DataFrame([("E", h + 0.5, "map_inv", 70.0 + h) for h in range(0, 12)],
                     schema=["stay_id", "h", "var", "value"], orient="row")
    st = _stays(stay_id=["E"], subject_id=["e"], los_h=[30.0], death_h=[None], limitation_h=[None],
                surgery_end_to_icu_h=[0.0], hospital=["h1"])
    e = build(st, hourly(v), _labs().clear(), support_hours(_supports().clear()), CFG)
    r = e.filter(pl.col("L") == 8).row(0, named=True)
    assert r["map_slope_6h"] == pytest.approx(1.0) and r["art_line"] and r["map_hours_6h"] == 6


def test_charted_points_merge_into_runs_while_gaps_are_short():
    pts = pl.DataFrame([("V", "vent", h, h) for h in (0.5, 3.0, 7.5, 14.0)] + [("V", "vaso", 1.0, 5.0)],
                       schema=["stay_id", "kind", "h", "h_end"], orient="row")
    hrs = support_hours(pts, gap_h=6)
    vent = sorted(hrs.filter(pl.col("kind") == "vent")["hour"].to_list())
    assert vent == list(range(0, 8)) + [14]                 # 0.5 → 7.5 one run (gaps < 6); 14.0 alone (gap 6.5)
    assert sorted(hrs.filter(pl.col("kind") == "vaso")["hour"].to_list()) == [1, 2, 3, 4, 5]   # interval unchanged
    no_runs = support_hours(pts).filter(pl.col("kind") == "vent")["hour"].to_list()
    assert sorted(no_runs) == [0, 3, 7, 14]                                   # without the rule: only the charted hours


def test_recent_supports_source_missing_and_unknown_group():
    st = _stays(stay_id=["A"], subject_id=["a"], los_h=[40.0], death_h=[None], limitation_h=[None],
                surgery_end_to_icu_h=[None], hospital=["h1"]).with_columns(pl.lit(None, pl.Utf8).alias("surgical_group"))
    sup = pl.DataFrame([("A", "vaso", 1.5, 1.5)], schema=["stay_id", "kind", "h", "h_end"], orient="row")
    v = _vitals().filter(pl.col("stay_id") == "A")
    lm = build(st, hourly(v), _labs(), support_hours(sup), CFG)
    r = lm.filter(pl.col("L") == 20).row(0, named=True)
    assert r["vaso_24h"] and not r["vaso_on"] and r["surgical_group"] == "unknown"
    assert not lm.filter(pl.col("L") == 26).row(0, named=True)["vaso_24h"]      # hour 1 is > 24 h before L = 26
    assert r["hours_since_surgery"] is None                                    # no surgery times: missing, not L
    li = build(st, hourly(v), _labs(), support_hours(sup), CFG, cohort="inspire")
    ri = li.filter(pl.col("L") == 4).row(0, named=True)
    assert ri["vaso_on"] is None and ri["vaso_hours"] is None and ri["vaso_24h"]   # INSPIRE: "running now" unknown
