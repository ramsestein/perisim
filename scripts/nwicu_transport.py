"""Transport of the frozen `hypo` metrics to NWICU (pre-registration: docs/prereg_nwicu.md).

Runs with `hypo`'s frozen code (PERISIM_SRC=frozen/hypo/src, or a v0.5 checkout); the freeze record is checked at the
start and the end. Reads the landmarks built by scripts/nwicu_build.py.

Usage:  python scripts/nwicu_transport.py [--threads 8] [--quick]
Outputs (aggregates): <workdir>/experiments/nwicu/transport.json, transport.md; per-fold predictions in units/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_fixed_metric as FM  # noqa: E402

LOG = logging.getLogger("nwicu")
SITE = "nwicu"
ROTATIONS = ("inspire", "mimic", "eicu")
N_BOOT = 2000
R_MARGIN, MIN_DEN = 0.80, 0.005


TAG = "nwicu"


def work() -> Path:
    return Path(os.environ.get("PERISIM_WORKDIR", "work")) / "experiments" / TAG


def landmarks(cfg: dict):
    import polars as pl
    from hypo.data import load

    lm = load(SITE, cfg, frame=pl.read_parquet(work() / "landmarks" / f"{SITE}.parquet"))
    return lm


def write(path: Path, rows: np.ndarray, p: np.ndarray, info: dict) -> None:
    import polars as pl
    pl.DataFrame({"row": rows, "p": p}).write_parquet(path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float))


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl
    d = pl.read_parquet(path)
    return d["row"].to_numpy(), d["p"].to_numpy()


def run_arms(cfg: dict, mcfg: dict, space: dict, lm, fold: np.ndarray, units: Path, threads: int) -> None:
    from hypo import compare
    from hypo.casebase import CaseBase
    from hypo.data import learning_sample
    from hypo.frozen import final_dir, load_pickle
    from hypo.metric import fit_metric
    from hypo.stats import recalibrate

    seed0 = int(cfg["seed"])
    metrics = {e: load_pickle(final_dir() / f"metric_ext-{e}.pkl") for e in ROTATIONS}
    comps = {e: load_pickle(final_dir() / f"comp_ext-{e}_lightgbm.pkl") for e in ROTATIONS}
    E = {e: m.embed(lm) for e, m in metrics.items()}
    for f in sorted(set(fold.tolist())):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        seed = seed0 + 100 * f
        ref, q = lm.take(tr), lm.take(te)
        t0 = time.perf_counter()
        for e in ROTATIONS:
            path = units / f"knn_{e}_f{f}.parquet"
            if not path.exists():
                cb = CaseBase(metrics[e], ref, mcfg, E=E[e][tr])
                a = cb.fit_alpha(seed)
                write(path, te, np.asarray(cb.predict(q, E=E[e][te])["p"], dtype=np.float64), {"alpha": a.get("alpha")})
            path = units / f"lgbm_{e}_f{f}.parquet"
            if not path.exists():
                p_te = np.asarray(comps[e](q), dtype=np.float64)
                p_tr = np.asarray(comps[e](ref), dtype=np.float64)
                write(path, te, p_te, {})
                write(units / f"lgbmrecal_{e}_f{f}.parquet", te, recalibrate(p_tr, ref.y, p_te), {})
        st_v = vaso_state(lm)
        for e in ROTATIONS:                    # pre-registered descriptive: same-state neighbours (B0) by vasopressor
            path = units / f"b0_{e}_f{f}.parquet"
            if not path.exists():
                p = np.full(te.size, np.nan)
                for s_ in (False, True):
                    qi, bi = te[st_v[te] == s_], tr[st_v[tr] == s_]
                    if qi.size == 0:
                        continue
                    if bi.size < int(mcfg.get("k", 50)) + 1:
                        bi = tr
                    cb = CaseBase(metrics[e], lm.take(bi), mcfg, E=E[e][bi])
                    cb.fit_alpha(seed)
                    p[np.isin(te, qi)] = np.asarray(cb.predict(lm.take(qi), E=E[e][qi])["p"], dtype=np.float64)
                write(path, te, p, {})
        path = units / f"local_f{f}.parquet"
        if not path.exists():
            rows = learning_sample(ref, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)
            m = fit_metric(ref.take(rows), mcfg, space, seed=seed, threads=threads)
            cb = CaseBase(m, ref, mcfg)
            a = cb.fit_alpha(seed)
            write(path, te, np.asarray(cb.predict(q)["p"], dtype=np.float64),
                  {"alpha": a.get("alpha"), "weights": m.weights_table()})
        for name in ("euclidean", "lightgbm"):
            path = units / f"{name}_f{f}.parquet"
            if not path.exists():
                pred, _ = compare.fit(name, ref, mcfg, space["baselines"], n_inner=int(cfg["cv"]["inner_folds"]),
                                      seed=seed, threads=threads)
                write(path, te, np.asarray(pred(q), dtype=np.float64), {})
        LOG.info("nwicu fold %d done (%.0f s)", f, time.perf_counter() - t0)


def vaso_state(lm) -> np.ndarray:
    v = np.asarray(lm.ft.num["vaso_on"], dtype=np.float64)
    return np.where(np.isfinite(v), v > 0.5, False)


def gather(units: Path, n: int) -> dict:
    arms = [f"knn_{e}" for e in ROTATIONS] + [f"lgbm_{e}" for e in ROTATIONS] + \
        [f"lgbmrecal_{e}" for e in ROTATIONS] + ["local", "euclidean", "lightgbm"] + [f"b0_{e}" for e in ROTATIONS]
    P = {}
    for a in arms:
        p = np.full(n, np.nan)
        for path in sorted(units.glob(f"{a}_f*.parquet")):
            r, v = read(path)
            p[r] = v
        if not np.isfinite(p).all():
            raise RuntimeError(f"nwicu: arm {a} has rows without a prediction")
        P[a] = p
    return P


def summarise(P: dict, y: np.ndarray, subj: np.ndarray, state: np.ndarray | None) -> dict:
    from hypo.stats import calibration_line_w, citl_w, logit, logloss_rows
    from sklearn.metrics import roc_auc_score

    B0 = {a: P.pop(a) for a in [k for k in P if k.startswith("b0_")]}
    losses = {a: logloss_rows(p, y) for a, p in P.items()}
    b0_loss = {a: logloss_rows(p, y) for a, p in B0.items()}
    est = {a: float(v.mean()) for a, v in losses.items()}
    B = FM._boot(losses, subj, N_BOOT, 20261107)
    arms = {}
    for a, p in P.items():
        x = logit(np.clip(p, 1e-6, 1 - 1e-6))
        _, slope = calibration_line_w(x, y.astype(float))
        arms[a] = {"logloss": est[a], "logloss_ci95": FM._ci(B[a]), "auroc": float(roc_auc_score(y, p)),
                   "brier": float(np.mean((p - y) ** 2)), "citl": citl_w(x, y.astype(float)), "slope": float(slope)}

    def R(arm):
        den_b, den = B["euclidean"] - B["local"], est["euclidean"] - est["local"]
        lo, hi = FM._ci(den_b)
        if den < MIN_DEN or lo <= 0:
            return {"est": None, "verdict": "not evaluable", "denominator": {"est": den, "ci95": [lo, hi]}}
        rb = (B["euclidean"] - B[arm]) / den_b
        c90 = FM._ci(rb, 0.90)
        return {"est": (est["euclidean"] - est[arm]) / den, "ci90": c90,
                "verdict": "equivalent" if c90[0] >= R_MARGIN else "not shown equivalent",
                "denominator": {"est": den, "ci95": [lo, hi]}}

    def D(a, b):
        return {"est": est[a] - est[b], "ci95": FM._ci(B[a] - B[b])}

    out = {"n_landmarks": int(y.size), "n_patients": int(np.unique(subj.astype(str)).size),
           "prevalence": float(y.mean()), "arms": arms,
           "R": {e: R(f"knn_{e}") for e in ROTATIONS},
           "knn_vs_lgbm_recal": {e: D(f"knn_{e}", f"lgbmrecal_{e}") for e in ROTATIONS},
           "lightgbm_local_vs_best_knn": None}
    best = min(ROTATIONS, key=lambda e: est[f"knn_{e}"])
    out["lightgbm_local_vs_best_knn"] = {"best": best, **D("lightgbm", f"knn_{best}")}
    if state is not None and state.any() and (~state).any():          # descriptive: the vasopressor stratum
        out["vaso"] = {"share_on": float(state.mean()),
                       "cost_on": {e: float(np.mean(losses[f"knn_{e}"][state] - losses["local"][state])) for e in ROTATIONS},
                       "cost_off": {e: float(np.mean(losses[f"knn_{e}"][~state] - losses["local"][~state])) for e in ROTATIONS},
                       "b0_repair_on": {e: float(np.mean(losses[f"knn_{e}"][state] - b0_loss[f"b0_{e}"][state]))
                                        for e in ROTATIONS},
                       "b0_minus_knn_all": {e: float(np.mean(b0_loss[f"b0_{e}"] - losses[f"knn_{e}"])) for e in ROTATIONS}}
    return out


def rule(s: dict) -> dict:
    eq = [e for e in ROTATIONS if s["R"][e]["verdict"] == "equivalent"]
    v = lambda d: "kNN better" if d["ci95"][1] < 0 else "kNN worse" if d["ci95"][0] > 0 else "no difference"  # noqa
    return {"primary": "confirmed" if len(eq) == 3 else "partial" if eq else "not confirmed", "equivalent": eq,
            "secondary_1": {e: v(s["knn_vs_lgbm_recal"][e]) for e in ROTATIONS},
            "secondary_2": {e: {"citl": s["arms"][f"knn_{e}"]["citl"], "slope": s["arms"][f"knn_{e}"]["slope"]}
                            for e in ROTATIONS}}


def render(s: dict) -> str:
    def r(d):
        return "not evaluable" if d.get("est") is None else \
            f"{d['est']:.3f} (90 % CI {d['ci90'][0]:.3f}–{d['ci90'][1]:.3f}; {d['verdict']})"
    L = ["# Transport of the frozen hypo metrics to NWICU", "",
         "Pre-registered: docs/prereg_nwicu.md. "
         f"{s['n_patients']:,} patients, {s['n_landmarks']:,} landmarks, outcome {s['prevalence']:.1%}; five patient "
         f"folds; {N_BOOT:,} patient resamples.", "",
         "| arm | refitted on NWICU | log-loss (95 % CI) | AUROC | Brier | CITL | slope |", "|---|---|---|---|---|---|---|"]
    lab = {**{f"knn_{e}": ("kNN, metric of rotation " + e, "case base") for e in ROTATIONS},
           **{f"lgbm_{e}": ("LightGBM of rotation " + e, "nothing") for e in ROTATIONS},
           **{f"lgbmrecal_{e}": ("LightGBM of rotation " + e + ", recalibrated", "recalibration") for e in ROTATIONS},
           "local": ("kNN, local metric", "metric + case base"), "euclidean": ("kNN, Euclidean", "case base"),
           "lightgbm": ("LightGBM, local", "everything")}
    for a, m in s["arms"].items():
        L.append(f"| {lab[a][0]} | {lab[a][1]} | {m['logloss']:.4f} ({m['logloss_ci95'][0]:.4f}–{m['logloss_ci95'][1]:.4f}) "
                 f"| {m['auroc']:.3f} | {m['brier']:.4f} | {m['citl']:+.3f} | {m['slope']:.2f} |")
    L += ["", "## Primary: R of each transported metric", ""]
    L += [f"- rotation {e} (learned on the other two sites): {r(s['R'][e])}" for e in ROTATIONS]
    L += ["", "## Secondary", ""]
    for e in ROTATIONS:
        d = s["knn_vs_lgbm_recal"][e]
        L.append(f"- kNN {e} − recalibrated LightGBM {e}: {d['est']:+.4f} ({d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f})")
    d = s["lightgbm_local_vs_best_knn"]
    L.append(f"- local LightGBM − best transported kNN ({d['best']}): {d['est']:+.4f} ({d['ci95'][0]:+.4f} to "
             f"{d['ci95'][1]:+.4f})")
    if "vaso" in s:
        v = s["vaso"]
        L += ["", f"## Descriptive: vasopressor stratum (on in {v['share_on']:.1%} of landmarks)", ""]
        L += [f"- rotation {e}: transport cost on {v['cost_on'][e]:+.4f}, off {v['cost_off'][e]:+.4f}; B0 (same-state "
              f"neighbours) on the stratum {v['b0_repair_on'][e]:+.4f} (positive = B0 better), overall "
              f"{v['b0_minus_knn_all'][e]:+.4f} (B0 − kNN)" for e in ROTATIONS]
    L += ["", "## Rule (pre-registered)", "", "```", json.dumps(s["rule"], indent=1), "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="quick frozen artefacts (tests)")
    ap.add_argument("--tag", default="nwicu", help="work folder: nwicu (primary) or nwicu_ivonly (post hoc sensitivity)")
    a = ap.parse_args(argv)
    global TAG
    TAG = a.tag
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.cv import outer_folds, settings
    from hypo.frozen import check

    rec = check()
    cfg, mcfg, space = settings(a.quick)
    lm = landmarks(cfg)
    units_dir = work() / "units"
    stamp = units_dir / "run.json"           # audit 2026-11-07: never mix units from another landmark set or setting
    key = {"quick": bool(a.quick), "n": len(lm), "ids_head": [str(x) for x in lm.ft.ids[:5]]}
    if stamp.exists() and json.loads(stamp.read_text()) != key:
        raise RuntimeError(f"nwicu: {units_dir} holds units of another run; move it away first")
    units_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(key))
    fold = outer_folds(lm, int(cfg["cv"]["outer_folds"]), int(cfg["seed"]))
    units = work() / "units"
    units.mkdir(parents=True, exist_ok=True)
    run_arms(cfg, mcfg, space, lm, fold, units, a.threads)
    check()
    P = gather(units, len(lm))
    state = vaso_state(lm)
    s = summarise(P, lm.y.astype(np.float64), lm.ft.subjects, state)
    s["freeze"] = rec.get("tag")
    s["rule"] = rule(s)
    (work() / "transport.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
    (work() / "transport.md").write_text(render(s), encoding="utf-8")
    LOG.info("nwicu: %s", work() / "transport.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
