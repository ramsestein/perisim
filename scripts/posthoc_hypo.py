"""Post-hoc, EXPLORATORY follow-ups of `hypo` (not pre-registered: added on 2026-10-29 at the lead's request, after
the external stage and the Phase 5 report, results/hypo/REPORT.md §12). The frozen artefacts are only read; the
freeze record is checked at the start and the end. Three questions:

B. The weights learned locally. The H1 arm learns a whole metric on each external site's training folds, and its
   weights differ from the transported ones (REPORT §7.4: in MIMIC-IV L 0.13 against 0.28, vasopressors running now
   0.13 against ~0). On the drift test set of fold 0 (S4's case base and test sample) the transported shapes are
   kept and its weights swapped for the local fold-0 metric's: all at once, and one variable at a time for every
   variable whose weight differs by ≥ SWAP_MIN. Reference: the local metric itself on the same rows. Then the
   re-deformation machinery of S4 restarted from the swapped weight for the variables the swap moves most and for
   those the transported metric gives (near-)zero weight, which S4 cannot move (prereg §9). Only fold 0's local
   metric is used: the other folds' training sets contain fold 0's test patients.
A. A threshold for the MAP block. The per-variable thresholds of S5 move because the models trade credit among the
   three correlated MAP summaries (REPORT §7.2). Along a steady MAP (last = 6-h mean = 6-h minimum = x) the three
   log-odds curves are added: the block threshold is the highest x below 85 mmHg where the summed curve rises ln 2,
   and the block odds ratios at a steady 70 and 65 mmHg against 85. The sum treats the three summaries as additive
   (exact for the additive surrogate models, an approximation for the trees). Per site (its own fold-0 shapes,
   refitted with S5's seeds, and the transported metric's) and per eICU hospital, with the spread over the models.
C. Drift over all five folds. S4 (external role only) refitted on folds 1–4 with the same code and settings; fold 0
   is the external stage's own output.

Usage:  python scripts/posthoc_hypo.py [--parts B A C] [--sites inspire mimic eicu] [--folds 1 2 3 4] [--threads 16]
        (run_local.ps1 hypo-posthoc)

Outside src/hypo on purpose (the frozen code hash covers it). Caches and fits (patient level, never exported) in
work/experiments/hypo/posthoc/; outputs (aggregates only): posthoc/posthoc_*.json and reports/posthoc.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypo import drift as D  # noqa: E402
from hypo.data import learning_sample  # noqa: E402
from hypo.external import Stage, hospital_labels  # noqa: E402
from hypo.frozen import check, dump, external_dir, load_pickle, logodds_curve, threshold, thresholds  # noqa: E402
from hypo.stats import PatientBootstrap, ci, logloss_rows  # noqa: E402

LOG = logging.getLogger("hypo.posthoc")
MAPV = ("map_last", "map_mean_6h", "map_min_6h")
SWAP_MIN = 0.02               # |w_local − w_transported| for a one-variable swap
RESTART_TOP = 3               # variables with the largest |Δw| restarted from the swapped weight
GRID = np.arange(40.0, 90.0001, 0.25)
STEADY = (70.0, 65.0)


def out_dir() -> Path:
    return external_dir().with_name("posthoc")


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def _delta(ll1: np.ndarray, ll0: np.ndarray, boot: PatientBootstrap) -> dict:
    d = ll1 - ll0
    bs = [float((boot.rows(wp) * d).sum() / boot.rows(wp).sum()) for wp in boot]
    return {"delta": float(d.mean()), "ci": ci(np.asarray(bs))}


# --------------------------------------------------------------------------- B: weights
def swapped_weights(metric, w_local: dict[str, float], only: list[str] | None = None) -> np.ndarray:
    """The transported metric's raw weights with those of `only` (all when None) replaced by the local metric's,
    rescaled to the transported total (a variable the local metric lacks gets 0)."""
    w = np.asarray(metric.w, float).copy()
    tot = float(w.sum())
    for j, n in enumerate(metric.enc.feature_names):
        if only is None or n in only:
            w[j] = float(w_local.get(n, 0.0)) * tot
    return w


def weights_part(st: Stage, X: str) -> dict:
    out = out_dir() / f"posthoc_B_ext-{X}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t = time.perf_counter()
    xd = st.x["drift"]
    f = int(xd["fold"])
    s = D._Site(st, X, X)
    metric = s.metric
    names = list(metric.enc.feature_names)
    w_tr = metric.weights_table()
    loc_info = json.loads((external_dir() / f"local_{X}_f{f}.json").read_text(encoding="utf-8"))
    w_loc = loc_info["weights"]
    boot = PatientBootstrap(s.subj_te, int(st.x["n_boot"]), st.seed + 41 + st.ci(X))
    cache = out.with_suffix(".pkl")
    fits = load_pickle(cache) if cache.exists() else {}

    def ll_of(key: str, points: list[np.ndarray]) -> np.ndarray:
        if key not in fits:
            fits[key] = s.ll(points)
            dump(fits, cache)
        return fits[key]

    ll0 = ll_of("frozen", s.frozen)
    res = {"rotation": f"external-{X}", "site": X, "fold": f, "n_case_base": s.n_ref, "n_test": int(s.te.size),
           "alpha": s.alpha, "frozen_logloss": float(ll0.mean()), "swap_min": SWAP_MIN}
    # references on the same rows: the external stage's own predictions
    for unit in ("transported", "local"):
        d = pl.read_parquet(external_dir() / f"{unit}_{X}_f{f}.parquet", columns=["row", "p"])
        pos = dict(zip(d["row"].to_list(), range(d.height)))
        p = d["p"].to_numpy()[[pos[r] for r in s.te]]
        res[f"stage_{unit}"] = {"logloss": float(logloss_rows(p, s.y_te).mean()),
                                **_delta(logloss_rows(p, s.y_te), ll0, boot)}
    # all weights local, shapes transported
    look_all = D.lookup(metric.enc, swapped_weights(metric, w_loc))
    res["all_weights_local"] = _delta(ll_of("all", [lk["points"] for lk in look_all]), ll0, boot)
    # one variable at a time
    diffs = sorted(((n, float(w_loc.get(n, 0.0)), float(w_tr[n])) for n in names
                    if abs(float(w_loc.get(n, 0.0)) - float(w_tr[n])) >= SWAP_MIN), key=lambda r: -abs(r[1] - r[2]))
    res["one_variable"] = []
    for n, wl, wt in diffs:
        look = D.lookup(metric.enc, swapped_weights(metric, w_loc, [n]))
        res["one_variable"].append({"name": n, "w_local": wl, "w_transported": wt,
                                    **_delta(ll_of(f"one:{n}", [lk["points"] for lk in look]), ll0, boot)})
    # the re-deformation restarted from the swapped weight
    zero = [n for n in names if float(w_tr[n]) < 0.005 and float(w_loc.get(n, 0.0)) >= 0.01]
    restart = list(dict.fromkeys([r[0] for r in diffs[:RESTART_TOP]] + zero))
    res["restart"] = []
    for n in restart:
        j = names.index(n)
        start = [lk["points"] for lk in D.lookup(metric.enc, swapped_weights(metric, w_loc, [n]))]
        key = f"restart:{n}"
        if key not in fits:
            Er = D.embed(start, s.codes_ref)
            cand = D.search_distinct(Er, s.code_ref, Er[s.a_rows], s.code_ref[s.a_rows], int(xd["k_candidates"]),
                                     s.fetch)
            P1 = D.fit_variable(start, j, s.codes_ref, s.a_rows, s.y_ref, cand, s.k, s.alpha, s.prior1, xd)
            pts = list(start)
            pts[j] = P1
            fits[key] = {"P1": P1, "ll": s.ll(pts)}
            dump(fits, cache)
        r = fits[key]
        cnt = s.counts[j]
        res["restart"].append({
            "name": n, "w_local": float(w_loc.get(n, 0.0)), "w_transported": float(w_tr[n]),
            "eff_weight_frozen": D.effective_weight(s.frozen[j], cnt),
            "eff_weight_start": D.effective_weight(start[j], cnt),
            "eff_weight_fitted": D.effective_weight(r["P1"], cnt),
            "start_vs_frozen": _delta(ll_of(f"one:{n}", start), ll0, boot),
            "fitted_vs_frozen": _delta(r["ll"], ll0, boot),
            "fitted_vs_start": _delta(r["ll"], fits[f"one:{n}"], boot)})
    res["seconds"] = time.perf_counter() - t
    _write(out, res)
    LOG.info("B (weights) external-%s: %.0f s", X, res["seconds"])
    return res


# --------------------------------------------------------------------------- A: the MAP block
def block_curve(enc, g: dict, grid: np.ndarray = GRID) -> np.ndarray:
    """Sum over the MAP summaries of their log-odds curves at a steady MAP x (each linear between its bin
    mid-points, flat beyond them)."""
    return np.sum([np.interp(grid, *logodds_curve(enc, v, g[v])) for v in MAPV], axis=0)


def block_stats(lo: np.ndarray, tcfg: dict, grid: np.ndarray = GRID) -> dict:
    ref, rise = float(tcfg["reference"]), float(tcfg["rise"])
    at = lambda x: float(np.interp(x, grid, lo))  # noqa: E731
    return {"threshold": threshold(grid, lo, ref, rise),
            **{f"or_{int(x)}": float(np.exp(at(x) - at(ref))) for x in STEADY}}


def _summ(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    fin = a[np.isfinite(a)]
    return {"n": int(a.size), "finite": int(fin.size),
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "p10": float(np.percentile(fin, 10)) if fin.size else float("nan"),
            "p90": float(np.percentile(fin, 90)) if fin.size else float("nan")}


def map_block(enc, curves: dict, tcfg: dict) -> dict:
    """The block on the consensus shapes and per accepted model (models aligned by index; when a model lacks one
    summary's curve, the per-model part is left out)."""
    fi = enc.feature_index()
    if any(v not in fi or enc.blocks[fi[v]].shape is None for v in MAPV):
        return {"note": "a MAP summary is not shaped"}
    cons = block_stats(block_curve(enc, {v: enc.blocks[fi[v]].shape for v in MAPV}), tcfg)
    lens = {len(curves.get(v, [])) for v in MAPV}
    res = {"consensus": cons, "aligned": len(lens) == 1}
    if len(lens) == 1:
        per = [block_stats(block_curve(enc, {v: curves[v][m] for v in MAPV}), tcfg) for m in range(lens.pop())]
        res["models"] = {k: _summ([p[k] for p in per]) for k in cons}
    return res


def block_part(st: Stage, sites: list[str]) -> dict:
    out = out_dir() / "posthoc_A_map_block.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t = time.perf_counter()
    xd, tcfg = st.x["drift"], st.x["threshold"]
    f = int(xd["fold"])
    seed = st.seed + 100 * f
    cache = out.with_suffix(".pkl")
    fits = load_pickle(cache) if cache.exists() else {}

    def loop_of(key: str, lm_s, sd: int):
        if key not in fits:
            enc, loop, _ = D._shapes(st, lm_s, sd)
            fits[key] = {"enc": enc, "curves": loop["curves"]}
            dump(fits, cache)
        return fits[key]["enc"], fits[key]["curves"]

    res = {"fold": f, "grid": [float(GRID[0]), float(GRID[-1])], "steady": list(STEADY), "sites": {}}
    for c in sites:
        LOG.info("A (MAP block) %s", c)
        lm = st.S.lm(c)
        tr, _ = st.S.split(c, f)
        pool = lm.take(tr)
        rows = learning_sample(pool, int(st.mcfg["learn"]["per_patient"]), int(st.mcfg["learn"]["per_site"]), seed)
        enc, curves = loop_of(f"site:{c}", pool.take(rows), seed)
        mt = st.metric(c)
        S5 = json.loads((external_dir() / f"shapes_{c}.json").read_text(encoding="utf-8"))
        again = thresholds(enc, curves, None, tcfg)
        site = {"own": map_block(enc, curves, tcfg), "transported": map_block(mt.enc, mt.info["curves"], tcfg),
                "per_variable_own": again,
                "reproduces_S5": {v: abs(again[v]["median"] - S5["thresholds"][v]["median"])
                                  for v in again if v in S5["thresholds"]}, "hospitals": {}}
        if lm.hospital[0] is not None:
            hosp = lm.hospital.astype(str)
            for gi, (h, lab) in enumerate(hospital_labels(lm, int(st.x["hospitals"]["min_patients"])).items()):
                eh, ch = loop_of(f"hospital:{c}:{lab}", lm.take(tr[hosp[tr] == h]), seed + gi + 1)
                site["hospitals"][lab] = {"block": map_block(eh, ch, tcfg),
                                          "per_variable": thresholds(eh, ch, None, tcfg)}
        res["sites"][c] = site
    res["seconds"] = time.perf_counter() - t
    _write(out, res)
    LOG.info("A (MAP block): %.0f s", res["seconds"])
    return res


# --------------------------------------------------------------------------- C: drift over the folds
def folds_part(st: Stage, sites: list[str], folds: list[int]) -> dict:
    res = {"sites": {}}
    for X in sites:
        per = {0: json.loads((external_dir() / f"drift_ext-{X}_{X}.json").read_text(encoding="utf-8"))}
        for f in folds:
            sf = copy.copy(st)
            sf.x = copy.deepcopy(st.x)
            sf.x["drift"]["fold"] = f
            sf.out = out_dir() / f"drift_f{f}"
            sf.out.mkdir(parents=True, exist_ok=True)
            LOG.info("C (drift) external-%s fold %d", X, f)
            D.site_drift(sf, X, X)
            per[f] = json.loads((sf.out / f"drift_ext-{X}_{X}.json").read_text(encoding="utf-8"))
        rows = []
        for f in sorted(per):
            d = per[f]
            rows.append({"fold": f, "frozen_logloss": d["frozen_logloss"], "joint": d["joint"]["delta"],
                         "joint_ci": d["joint"]["ci"], "missing": d["missing"]["delta"],
                         "missing_ci": d["missing"]["ci"],
                         "one_var_ci_below_0": [v["name"] for v in d["variables"] if v["ci"][1] < 0],
                         "one_var_ci_above_0": [v["name"] for v in d["variables"] if v["ci"][0] > 0]})
        names = [v["name"] for v in per[0]["variables"]]
        var = {n: {"mean_delta": float(np.mean([next(v["delta"] for v in per[f]["variables"] if v["name"] == n)
                                                for f in per])),
                   "folds_ci_below_0": int(sum(n in r["one_var_ci_below_0"] for r in rows)),
                   "folds_ci_above_0": int(sum(n in r["one_var_ci_above_0"] for r in rows)),
                   "mean_split_half_r": float(np.nanmean([next(v["split_half_r"] for v in per[f]["variables"]
                                                               if v["name"] == n) for f in per]))}
               for n in names}
        jd = np.array([r["joint"] for r in rows])
        res["sites"][X] = {"folds": rows, "joint_mean": float(jd.mean()), "joint_sd": float(jd.std(ddof=1))
                           if jd.size > 1 else float("nan"), "variables": var}
    _write(out_dir() / "posthoc_C_drift_folds.json", res)
    return res


# --------------------------------------------------------------------------- report
def _f(x, d=4, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _dci(r: dict, d: int = 4) -> str:
    return f"{_f(r['delta'], d)} ({_f(r['ci'][0], d)} to {_f(r['ci'][1], d)})"


def _q(s: dict, d: int = 1) -> str:
    return f"{_f(s['median'], d, False)} ({_f(s['p10'], d, False)}–{_f(s['p90'], d, False)})" if s else "—"


def render(B: dict, A: dict | None, C: dict | None) -> str:
    L = ["# `hypo` post hoc follow-ups (exploratory, not pre-registered)", "",
         "Added on 2026-10-29 at the lead's request after the Phase 5 report (results/hypo/REPORT.md §12). The frozen "
         "artefacts are only read; the freeze record was checked at the start and the end. Log-loss differences: "
         "negative = better than the frozen transported metric; 95 % CIs by patient bootstrap on the drift test set "
         "of fold 0 (S4's case base and test sample).", ""]
    if B:
        L += ["## B. The weights learned locally (fold 0)", "",
              "| site | frozen log-loss | local metric (stage) | all weights local, shapes transported | check: the "
              "stage's transported predictions |", "|---|---|---|---|---|"]
        for X, r in B.items():
            L.append(f"| {X} | {r['frozen_logloss']:.4f} | {_dci(r['stage_local'])} | {_dci(r['all_weights_local'])}"
                     f" | {_dci(r['stage_transported'])} |")
        L += ["", f"One variable at a time (|w_local − w_transported| ≥ {SWAP_MIN}):", "",
              "| site | variable | w local | w transported | Δ log-loss |", "|---|---|---|---|---|"]
        for X, r in B.items():
            for v in r["one_variable"]:
                L.append(f"| {X} | {v['name']} | {v['w_local']:.3f} | {v['w_transported']:.3f} | {_dci(v)} |")
        L += ["", "Re-deformation restarted from the swapped weight (effective weight in the site, units of S4):", "",
              "| site | variable | eff. weight frozen → start → fitted | start − frozen | fitted − frozen | "
              "fitted − start |", "|---|---|---|---|---|---|"]
        for X, r in B.items():
            for v in r["restart"]:
                L.append(f"| {X} | {v['name']} | {v['eff_weight_frozen']:.3f} → {v['eff_weight_start']:.3f} → "
                         f"{v['eff_weight_fitted']:.3f} | {_dci(v['start_vs_frozen'])} | {_dci(v['fitted_vs_frozen'])}"
                         f" | {_dci(v['fitted_vs_start'])} |")
        L.append("")
    if A:
        L += ["## A. The MAP block along a steady MAP (fold-0 shapes)", "",
              "Threshold: highest steady MAP below 85 mmHg where the summed log-odds of the three MAP summaries rise "
              "ln 2; odds ratios of a steady 70 and 65 mmHg against 85. Median (10–90 %) over the accepted models.", "",
              "| site | shapes | block threshold | OR 70 vs 85 | OR 65 vs 85 | per-variable `map_last` |",
              "|---|---|---|---|---|---|"]
        for c, s in A["sites"].items():
            for who in ("own", "transported"):
                b = s[who]
                m = b.get("models", {})
                pv = s["per_variable_own"].get("map_last", {}) if who == "own" else {}
                L.append(f"| {c} | {who} | {_q(m.get('threshold', {}))} | {_q(m.get('or_70', {}), 2)} | "
                         f"{_q(m.get('or_65', {}), 2)} | {_q(pv) if pv else '—'} |")
        L += ["", "Reproduction of S5 (|median per-variable threshold − S5's|, mmHg): " + "; ".join(
            f"{c} " + ", ".join(f"{v} {d:.2f}" for v, d in s["reproduces_S5"].items()) for c, s in A["sites"].items()),
            ""]
        for c, s in A["sites"].items():
            if not s["hospitals"]:
                continue
            H = s["hospitals"]
            L += [f"eICU hospitals ({len(H)}): spread of the hospital medians and median 10–90 % width inside a "
                  "hospital.", "", "| quantity | median | IQR | range | SD between hospitals | width inside |",
                  "|---|---|---|---|---|---|"]
            rows = {"block threshold": [(h["block"].get("models", {}).get("threshold", {})) for h in H.values()],
                    "block OR 70 vs 85": [(h["block"].get("models", {}).get("or_70", {})) for h in H.values()]}
            for v in MAPV:
                rows[f"`{v}` threshold"] = [h["per_variable"].get(v, {}) for h in H.values()]
            for k, lst in rows.items():
                med = np.array([x.get("median", np.nan) for x in lst], float)
                wid = np.array([x.get("p90", np.nan) - x.get("p10", np.nan) for x in lst], float)
                fin = med[np.isfinite(med)]
                if not fin.size:
                    continue
                d = 2 if "OR" in k else 1
                L.append(f"| {k} | {np.median(fin):.{d}f} | {np.percentile(fin, 25):.{d}f}–{np.percentile(fin, 75):.{d}f}"
                         f" | {fin.min():.{d}f}–{fin.max():.{d}f} | {fin.std():.{d}f} | {np.nanmedian(wid):.{d}f} |")
            L.append("")
    if C:
        L += ["## C. Drift over the five folds (external role)", "",
              "| site | fold | frozen log-loss | all variables re-deformed | missing only | one-variable CIs < 0 / > 0 |",
              "|---|---|---|---|---|---|"]
        for X, s in C["sites"].items():
            for r in s["folds"]:
                L.append(f"| {X} | {r['fold']} | {r['frozen_logloss']:.4f} | {_f(r['joint'])} ({_f(r['joint_ci'][0])} "
                         f"to {_f(r['joint_ci'][1])}) | {_f(r['missing'])} ({_f(r['missing_ci'][0])} to "
                         f"{_f(r['missing_ci'][1])}) | {len(r['one_var_ci_below_0'])} / {len(r['one_var_ci_above_0'])} |")
            L.append(f"| {X} | mean (SD) | | {_f(s['joint_mean'])} ({_f(s['joint_sd'], 4, False)}) | | |")
        L += ["", "Variables whose one-variable re-deformation has a CI below 0 in two or more folds: " + "; ".join(
            f"{X}: " + (", ".join(f"{n} ({v['folds_ci_below_0']}/5, mean {_f(v['mean_delta'])})"
                                  for n, v in s["variables"].items() if v["folds_ci_below_0"] >= 2) or "none")
            for X, s in C["sites"].items()), ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="*", default=["B", "A", "C"], choices=["A", "B", "C"])
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    rec = check()
    st = Stage(threads=a.threads)
    sites = a.sites or list(st.cfg["cohorts"])
    out_dir().mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    B = {X: weights_part(st, X) for X in sites} if "B" in a.parts else {}
    A = block_part(st, sites) if "A" in a.parts else None
    C = folds_part(st, sites, a.folds) if "C" in a.parts else None
    check()
    summary = {"freeze": rec["tag"], "parts": a.parts, "sites": sites, "B": B, "A": A, "C": C,
               "seconds": time.perf_counter() - t}
    _write(out_dir() / "posthoc.json", summary)
    rep = external_dir().parent / "reports" / "posthoc.md"
    rep.write_text(render(B, A, C), encoding="utf-8")
    LOG.info("post hoc report: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
