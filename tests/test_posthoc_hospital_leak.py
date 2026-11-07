"""scripts/posthoc_leak_sensitivity.py and scripts/posthoc_hospital_transport.py."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_hospital_transport as H  # noqa: E402
import posthoc_leak_sensitivity as K  # noqa: E402


def test_states_before_uses_only_records_before_L():
    sup = pl.DataFrame({"stay_id": [1, 1, 2, 3, 3], "kind": ["vaso", "vaso", "vaso", "vent", "sedation"],
                        "h": [3.5, 9.0, 1.0, 5.0, 0.2], "h_end": [3.5, 9.0, 20.0, 5.0, 0.2]})
    keys = pl.DataFrame({"stay_id": [1, 1, 2, 2, 3], "L": [6, 10, 2, 22, 6]})
    s = K.states_before(sup, keys, look=2)
    # stay 1: a point at 3.5 is 2 h old at L = 6 (hour 3 < 4): off; the point at 9 is seen at L = 10: on.
    # v0.5 merged 3.5 and 9.0 into one run (gap < 6 h), marking L = 6 on from a record after L.
    assert s["vaso_on"].tolist() == [False, True, True, True, False]   # stay 2: interval to 20 reaches hour 20 at L = 22
    assert s["vent_on"].tolist() == [False, False, False, False, True]      # 5.0 reaches hour 4 at L = 6
    assert not s["sedation_on"].any()


def _chain(tmp_path, monkeypatch, parts):
    from hypo.config import load
    from test_hypo_external import _xframe

    cfg = load()
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    d = tmp_path / "experiments" / "hypo" / "landmarks"
    d.mkdir(parents=True)
    frames = {}
    for i, c in enumerate(cfg["cohorts"]):
        frames[c] = _xframe(c, 120, 4, 10 + i)
        frames[c].write_parquet(d / f"{c}.parquet")
    from hypo.cv import run
    from hypo.external import Stage
    from hypo.frozen import final, freeze

    run(cfg["cohorts"], ["prevalence", "euclidean", "lightgbm"], None, quick=False, threads=1)
    final(quick=True, threads=1)
    freeze("TEST", no_git=True, quick=True)
    Stage(quick=True, threads=1).run(["eicu"], folds=[0, 1], parts=parts)
    return frames


def test_hospital_chain(tmp_path, monkeypatch):
    _chain(tmp_path, monkeypatch, ["transported", "local"])
    monkeypatch.setattr(H, "MIN_PATIENTS", 10)
    monkeypatch.setattr(H, "N_BOOT", 200)
    assert H.main(["--threads", "1", "--quick"]) == 0
    from hypo.frozen import external_dir
    out = json.loads((external_dir().with_name("posthoc_hospital_transport") / "hospitals.json").read_text())
    assert out["n_eligible"] >= 2 and set(out["pooled"]["arms"]) == set(H.ARMS)
    assert "Q1_within_system_equivalent_pooled" in out["rule"]


def test_leak_chain(tmp_path, monkeypatch):
    frames = _chain(tmp_path, monkeypatch, ["transported", "local", "comps"])
    ex = tmp_path / "experiments" / "hypo" / "extract"
    ex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for c, f in frames.items():
        st = f["stay_id"].unique().to_numpy()
        n = st.size * 3
        pl.DataFrame({"stay_id": rng.choice(st, n), "kind": rng.choice(["vaso", "sedation", "vent"], n),
                      "h": rng.uniform(0, 40, n)}).with_columns(pl.col("h").alias("h_end")) \
            .write_parquet(ex / f"{c}_supports.parquet")
    import posthoc_state_test as T
    monkeypatch.setattr(T, "N_BOOT", 200)
    assert K.main(["--sites", "eicu", "--threads", "1", "--quick"]) == 0
    from hypo.frozen import external_dir
    out = json.loads((external_dir().with_name("posthoc_leak_sensitivity") / "leak.json").read_text())
    assert "eicu" in out["reclass"] and "all_unchanged" in out["compare"]
