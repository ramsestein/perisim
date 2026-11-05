"""The helpers of scripts/posthoc_alignment.py (exploratory): the measure, T2's draws, the check against the
2026-11-04 pairs and the competence rows (DECISIONS 2026-11-04 and 2026-11-05)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_alignment as A  # noqa: E402


def test_overlap_counts_shared_neighbours_both_ways():
    a = np.array([[1, 2, 3, 4]])
    b = np.array([[4, 3, 9, -1]])                               # two shared; −1 (no neighbour) never matches
    assert A.overlap(a, b, 4)[0] == pytest.approx(0.5)
    assert A.overlap(a, a, 4)[0] == 1.0


def test_t2_draws_are_disjoint_halves_of_whole_patients():
    subj = np.concatenate([np.arange(1200), np.arange(300)]).astype(str)   # 1,200 patients, 300 with two stays
    pool = np.arange(subj.size) + 7                             # pool positions in the cohort
    draws = A.t2_size_draws(subj, pool, seed=1, si=0)
    names = [n for n, _, _ in draws]
    assert names == [f"n{n}_{d}" for n in A.SIZES for d in "ab"]
    by = {n: (rows, s) for n, rows, s in draws}
    sp = dict(zip(pool, subj))
    for n in A.SIZES:
        (ra, sa), (rb, sb) = by[f"n{n}_a"], by[f"n{n}_b"]
        pa, pb = {sp[r] for r in ra}, {sp[r] for r in rb}
        m = 1200 // 2 if n == "half" else int(n)
        if 2 * m > 1200:                                        # fewer patients than two draws need
            continue
        assert len(pa) == len(pb) == m and not pa & pb and sa != sb
        assert set(ra) == {r for r in pool if sp[r] in pa}      # every stay of a drawn patient
    again = A.t2_size_draws(subj, pool, seed=1, si=0)
    assert all(np.array_equal(r1, r2) for (_, r1, _), (_, r2, _) in zip(draws, again))


def test_pairs_reproduce_flags_any_change():
    old = {"a|b": {"k50_row": {"est": 0.3}, "huh": {"est": 0.2}, "pred": {"mean_tv": 0.1}}}
    assert A.t2_pairs_reproduce({"a|b": {"k50_row": {"est": 0.3}, "huh": {"est": 0.2}, "pred": {}}}, old) == []
    assert A.t2_pairs_reproduce({"a|b": {"k50_row": {"est": 0.3}, "huh": {"est": 0.21}, "pred": {}}}, old) == ["a|b"]
    assert A.t2_pairs_reproduce({}, old) == ["a|b"]


def test_competence_rows_put_both_experiments_on_one_axis():
    pair = {"huh": {"est": 0.25}, "k50_row": {"est": 0.12}}
    H = {"mimic": {"sizes": [{"name": "n500_a", "n_patients": 500}, {"name": "n500_b", "n_patients": 500}],
                   "metrics": {"n500_a": {"logloss": 0.5, "n_learn": 1500}, "n500_b": {"logloss": 0.4, "n_learn": 1490}},
                   "pairs": {"n500_a|n500_b": pair}}}
    N = {"eicu": {"sizes": [{"name": "n500_a", "n_patients": 500, "n_rows": 530}],      # its b failed
                  "metrics": {"n500_a": {"logloss": 1.0}}, "pairs": {}, "ll_prevalence": 1.1}}
    rows = A.competence_rows(H, N, {"mimic": 0.6})
    assert len(rows) == 1                                       # a draw without its other half is left out
    r = rows[0]
    assert r["experiment"] == "hypo" and r["learning_rows"] == [1500, 1490] and r["huh"] == 0.25
    assert r["competence"] == pytest.approx([1 - 0.5 / 0.6, 1 - 0.4 / 0.6])
