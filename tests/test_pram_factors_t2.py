"""scripts/pram_factors_t2.py on synthetic PRAM-like cohorts: pipeline, votes, rule, report."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("catboost")
import pram_build as PB  # noqa: E402
import pram_factors_t2 as T2  # noqa: E402


def _cohort(n: int, seed: int, site: str) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {c: rng.normal(size=n) for c in PB.FEATURES}
    cols["sex_male"] = rng.integers(0, 2, n).astype(float)
    lin = 1.5 * cols["scr_feat_max"] - 1.0 * cols["gcs_total_min"] + 0.5 * cols["age"]
    for c in ("inr_min", "temp_c_last"):
        v = cols[c]
        v[rng.random(n) < 0.3] = np.nan                     # missingness for the imputer
    df = pl.DataFrame({"id": [f"{site}{i}" for i in range(n)], "site": site, **cols})
    return df.with_columns(pl.Series("aki", rng.random(n) < 1 / (1 + np.exp(-(lin - 1.5)))),
                           pl.Series("death", rng.random(n) < 1 / (1 + np.exp(-(0.8 * lin - 2.0)))))


def test_dev_split_proportions():
    df = _cohort(1000, 0, "m")
    s = T2.dev_split(df)
    assert abs((s == 0).mean() - 0.56) < 0.01 and abs((s == 1).mean() - 0.14) < 0.01


def test_embed_cosine_and_diag():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    E = T2.embed("cos", X, None)
    assert np.allclose(E[0], [0.6, 0.8]) and np.allclose(E[1], 0)
    assert np.allclose(T2.embed("ncad", X, np.array([2.0, 8.0]))[0], [3 / np.sqrt(2) * np.sqrt(2), 4 / np.sqrt(2) * np.sqrt(8)])


def test_knn_votes_drop_self_in_loo():
    rng = np.random.default_rng(1)
    Eb = rng.normal(size=(300, 3)).astype(np.float32)
    yb = (Eb[:, 0] > 0).astype(float)
    r = T2.knn_votes(Eb, yb, Eb[:5], False, {"k": 10, "alpha_queries": 300, "alpha_grid": [0.1, 1, 10]}, 0)
    assert r["ours"].shape == (5,) and np.all((r["inv"] >= 0) & (r["inv"] <= 1)) and r["alpha"] in (0.1, 1, 10)


def test_t2_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    w = tmp_path / "experiments" / "pram"
    w.mkdir(parents=True)
    _cohort(900, 1, "m").write_parquet(w / "mimic.parquet")
    _cohort(700, 2, "e").write_parquet(w / "eicu.parquet")
    (w / "build.md").write_text("# build\n\nheader\n| step | x |\n")
    monkeypatch.setattr(T2, "N_BOOT", 100)
    monkeypatch.setattr(T2, "NCA_N", {"ncaf": 300, "ncaf2k": 200, "ncaf10k": 400})
    assert T2.main(["--threads", "1", "--quick"]) == 0
    out = json.loads((w / "t2.json").read_text())
    assert set(out["outcomes"]) == {"aki", "death"} and "reproduced" in out["r0"]
    r = out["outcomes"]["aki"]["a"]
    assert set(r["rule"]["switch"]) == set(T2.SWITCH) and set(r["rule"]["effects"]) == {"U", "V", "I"}
    assert "ncad_t|ours|mix_gbm" in r["cells"] and "ncaf10k|inv|alone" in r["cells"]
    rep = (w / "t2.md").read_text()
    assert "R0: replication" in rep and "H-S" in rep
