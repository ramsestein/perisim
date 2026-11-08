"""Branch arc-metric (docs/prereg_arc.md §5): per-row log-losses of one arm, for the paired contrast arc vs value.

Usage:  PERISIM_ARM=<arc|value> PERISIM_WORKDIR=<that arm's work folder> python scripts/arc_dump.py
        [--part all|hypo|nwicu|t2] [--profile local]

Run once per arm, after that arm's experiments have finished. Nothing is fitted or refitted: the predictions each
experiment already saved are read (T2: rebuilt from its frozen artefacts exactly as scripts/posthoc_equivalence.py
does, and checked against the transport stage to 1e-6). For every unit it writes, in the arm's work folder
(patient level, never exported), arc_compare/<unit>.npz with:

- ids, subjects, y of the evaluated rows (the pairing key: scripts/arc_compare.py refuses units whose rows differ
  between the arms);
- ll_transported, ll_local, ll_euclidean: per-row log-loss of the transported metric, of the metric learned at the
  site, and of the Euclidean kNN (neutral axes, which do not depend on the arm).

Units (§5 of the pre-registration):
- hypo_<X>: `hypo`'s external stage at site X ∈ {inspire, mimic, eicu}, five folds (transported and local) and
  Phase 2's Euclidean kNN, as scripts/posthoc_equivalence.py part A;
- nwicu_<e>: NWICU, the frozen metric of rotation e (transported), the NWICU-learned metric (local) and the
  Euclidean kNN, as scripts/nwicu_transport.py;
- t2_<c>: T2's transport stage at external cohort c, A(i) frozen INSPIRE metric, A(iii) local refit and the
  Euclidean kNN with the same case base, as scripts/posthoc_equivalence.py part B.

Also, for `hypo`'s three frozen metrics, arc_compare/shapes_<rotation>.json (aggregate): per shaped continuous /
ordinal variable its consensus curve g_j, its weight, the arc length, the net change and the backtracking index
1 − net / arc (0 = monotone; §6.3). The curves do not depend on the arm (the ensemble is fitted on the model matrix
with the same seeds), which scripts/arc_compare.py checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("arc.dump")


def _out(base: Path) -> Path:
    d = base / "arc_compare"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(path: Path, *, ids, subjects, y, ll: dict, info: dict) -> None:
    np.savez_compressed(path, ids=np.asarray(ids).astype(str), subjects=np.asarray(subjects).astype(str),
                        y=np.asarray(y), **{f"ll_{k}": np.asarray(v, dtype=np.float64) for k, v in ll.items()})
    path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    LOG.info("%s: %d rows, log-loss %s", path.name, len(ids),
             {k: round(float(np.mean(v)), 5) for k, v in ll.items()})


def part_hypo(base: Path, arm: str) -> list[str]:
    from hypo.external import Stage
    from hypo.external_report import _on, _phase2, _units
    from hypo.frozen import check
    from hypo.stats import logloss_rows

    rec = check()
    st = Stage(threads=4)
    done = []
    for X in st.cfg["cohorts"]:
        T = _units(f"transported_{X}_f*.parquet")
        if T is None:
            LOG.warning("hypo %s: no external-stage units", X)
            continue
        lm = st.S.lm(X)
        rows = T["row"].to_numpy()
        p = {"transported": T["p"].to_numpy(), "local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
             "euclidean": _on(_phase2(X, "euclidean"), rows, "p")}
        if any(v is None for v in p.values()):
            raise RuntimeError(f"hypo {X}: a unit is missing ({[k for k, v in p.items() if v is None]})")
        ok = np.all([np.isfinite(v) for v in p.values()], axis=0)
        rows, p = rows[ok], {k: v[ok] for k, v in p.items()}
        y = lm.y[rows]
        _save(_out(base) / f"hypo_{X}.npz", ids=lm.ft.ids[rows], subjects=lm.ft.subjects[rows], y=y,
              ll={k: logloss_rows(v, y) for k, v in p.items()},
              info={"arm": arm, "experiment": "hypo", "site": X, "freeze": rec.get("tag"), "rows_dropped": int((~ok).sum())})
        done.append(f"hypo_{X}")
    check()
    return done


def shape_table(metric) -> dict:
    """Per shaped continuous / ordinal variable: g_j (rounded), weight share, arc length, net change, backtracking."""
    enc, w = metric.enc, np.asarray(metric.w, dtype=np.float64)
    out = {}
    for j, b in enumerate(enc.blocks):
        if b.shape is None or b.kind not in ("continuous", "ordinal"):
            continue
        g = np.asarray(b.shape, dtype=np.float64)
        arc = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
        net = float(np.linalg.norm(g[-1] - g[0]))
        out[b.name] = {"weight_share": float(w[j] / w.sum()), "arc": arc, "net": net,
                       "backtracking": float(1 - net / arc) if arc > 0 else 0.0,
                       "g": np.round(g, 6).tolist()}
    return out


def part_shapes(base: Path, arm: str) -> list[str]:
    from hypo.config import load
    from hypo.frozen import final_dir, load_pickle

    done = []
    for e in load()["cohorts"]:
        p = final_dir() / f"metric_ext-{e}.pkl"
        if not p.is_file():
            continue
        (_out(base) / f"shapes_{e}.json").write_text(json.dumps({"arm": arm, "rotation": e,
                                                                  "variables": shape_table(load_pickle(p))}, indent=1),
                                                     encoding="utf-8")
        done.append(f"shapes_{e}")
    return done


def part_nwicu(base: Path, arm: str) -> list[str]:
    import nwicu_transport as NT
    from hypo.cv import settings
    from hypo.frozen import check
    from hypo.stats import logloss_rows

    units = NT.work() / "units"
    if not units.is_dir():
        LOG.warning("nwicu: no units in %s", units)
        return []
    rec = check()
    cfg, _, _ = settings(False)
    lm = NT.landmarks(cfg)
    P = NT.gather(units, len(lm))
    y = lm.y.astype(np.int64)
    done = []
    for e in NT.ROTATIONS:
        ll = {"transported": logloss_rows(P[f"knn_{e}"], y), "local": logloss_rows(P["local"], y),
              "euclidean": logloss_rows(P["euclidean"], y)}
        _save(_out(base) / f"nwicu_{e}.npz", ids=lm.ft.ids, subjects=lm.ft.subjects, y=y, ll=ll,
              info={"arm": arm, "experiment": "nwicu", "rotation": e, "freeze": rec.get("tag")})
        done.append(f"nwicu_{e}")
    check()
    return done


def part_t2(base: Path, arm: str, profile: str | None) -> list[str]:
    import _frozen
    import posthoc_equivalence as EQ
    from perisim.config import load_config
    from perisim.pipeline import Pipeline
    from perisim.transport.analyses import context
    from perisim.transport.core import check_hashes, frozen_hashes

    pipe = Pipeline(load_config(None, profile, {"experiment": {"paths": {"workdir": str(base)}}}))
    cfg, ws = pipe.cfg, pipe.ws
    tpath = ws.artifacts() / "transport.json"
    if not tpath.is_file():
        LOG.warning("t2: no transport stage in %s", ws.root)
        return []
    tj = json.loads(tpath.read_text(encoding="utf-8"))
    _frozen.check_t2_freeze(cfg, ws)
    before = frozen_hashes(ws, ["t2"])
    args = {"names": pipe.feature_names(), "dev": cfg.experiment["cohorts"]["development"],
            "externals": [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]}
    done = []
    for c in args["externals"]:
        ll, info = EQ.t2_unit(cfg, ws, "t2", c, args)
        for k, v in EQ._published(tj, "t2", c).items():
            if abs(float(ll[k].mean()) - float(v)) > 1e-6:
                raise RuntimeError(f"t2 {c}: {k} log-loss {ll[k].mean():.6f} does not reproduce the stage's {v:.6f}")
        ctx = context(cfg, ws, "t2", c, args)                   # the rows of t2_unit: test rows with a label
        y_all = ctx["lab"]["own"]
        te = ctx["split"]["test"]
        q = te[y_all[te] >= 0]
        _save(_out(base) / f"t2_{c}.npz", ids=ctx["X"].ft.ids[q], subjects=info["subjects"], y=y_all[q], ll=ll,
              info={"arm": arm, "experiment": "t2", "site": c})
        done.append(f"t2_{c}")
    check_hashes(before, frozen_hashes(ws, ["t2"]))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["all", "hypo", "nwicu", "t2"], default="all")
    ap.add_argument("--profile", default=os.environ.get("PERISIM_PROFILE") or "local")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from perisim.arc_guard import current_arm, ensure_arc_workdir

    arm = current_arm()
    base = ensure_arc_workdir()
    done = []
    if a.part in ("all", "hypo"):
        done += part_hypo(base, arm)
        done += part_shapes(base, arm)
    if a.part in ("all", "nwicu"):
        done += part_nwicu(base, arm)
    if a.part in ("all", "t2"):
        done += part_t2(base, arm, a.profile or None)
    LOG.info("arm %s: %d units in %s", arm, len(done), base / "arc_compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
