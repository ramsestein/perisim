"""Post-hoc, PRE-SPECIFIED test of the intervention-state mechanism (`hypo`; plan: docs/DECISIONS.md 2026-11-07).

The vasopressor failure (REPORT §15.1, §18) suggested a mechanism: an input that one development source cannot record
(constant within it) gets little weight in the jointly learned metric, and the transported metric then pays where
that intervention changes the physiology; neighbours drawn from the query's own intervention state (B0) repair it
without refitting. INSPIRE records neither vasopressors nor sedation, so the mechanism makes the same prediction for
sedation, whose strata were never examined. Mechanical ventilation, recorded at every site, is the negative control.
The vasopressor is re-run as the reference case, with the comparators (LightGBM, MI) by stratum and R after B0.

Run with `hypo`'s frozen code (PERISIM_SRC=frozen/hypo/src); the freeze record is checked at the start and the end;
frozen artefacts are only read. Nothing is refitted: every arm uses stored predictions or the frozen metric.

Usage:  python scripts/posthoc_state_test.py [--sites inspire mimic eicu] [--threads 8] [--quick]
Outputs (aggregates only): work/experiments/hypo/posthoc_state_test/state_test.json, reports/posthoc_state_test.md;
per-fold predictions in posthoc_state_test/units/ (work folder).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:                                    # v0.6 repository: $PERISIM_SRC (frozen code) first on sys.path
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:                     # a v0.5 checkout: its own src is the frozen code
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_fixed_metric as FM  # noqa: E402  (bootstrap helpers)

LOG = logging.getLogger("posthoc.state_test")
SEED = 20261107
N_BOOT = 2000
GATE_TOL = 1e-9
MIN_EFFECT = 0.001     # nats per landmark: the smallest repair that counts (the A-sel lesson, DECISIONS 2026-11-06)
SAFE = 0.001           # nats per landmark: the largest tolerated overall loss of B0 against the transported arm
MIN_DEN = 0.005        # R's denominator rule (posthoc_equivalence)
R_MARGIN = 0.80
STATES = {"sedation_on": "test", "vent_on": "control", "vaso_on": "reference"}
WEIGHT_VARS = {"sedation_on": ["sedation_on", "sedation_24h"], "vent_on": ["vent_on"],
               "vaso_on": ["vaso_on", "vaso_24h", "vaso_hours"]}


def state_of(lm, var: str) -> np.ndarray:
    v = np.asarray(lm.ft.num[var], dtype=np.float64)
    return np.where(np.isfinite(v), v > 0.5, False)


def predict_b0(st, X: str, f: int, enc, w: np.ndarray, state: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(test rows, p) of fold f with the frozen metric; with `state`, each query votes among same-state neighbours."""
    from hypo.casebase import CaseBase
    from hypo.metric import Metric

    lm = st.S.lm(X)
    tr, te = st.S.split(X, f)
    seed = st.seed + 100 * f
    E = st.embed(X, X)
    groups = [(te, tr)] if state is None else [(te[state[te] == s], tr[state[tr] == s]) for s in (False, True)]
    p = np.full(te.size, np.nan)
    for q, base in groups:
        if q.size == 0:
            continue
        if base.size < int(st.mcfg.get("k", 50)) + 1:      # too few same-state patients: whole case base
            base = tr
        cb = CaseBase(Metric(enc, w), lm.take(base), st.mcfg, E=E[base])
        cb.fit_alpha(seed)
        p[np.isin(te, q)] = np.asarray(cb.predict(lm.take(q), E=E[q])["p"], dtype=np.float64)
    return te, p


def weight_share(enc, w, var: str) -> float:
    names = list(enc.feature_names)
    w = np.asarray(w, dtype=np.float64)
    return float(sum(w[names.index(n)] for n in WEIGHT_VARS[var] if n in names) / w.sum())


# --------------------------------------------------------------------------- statistics
def boot_strata(losses: dict, subj: np.ndarray, masks: dict, n_boot: int, seed: int) -> dict:
    """Replicates of each arm's mean loss in each stratum, with the SAME multinomial patient weights for every
    stratum and arm (a patient with landmarks on and off counts in both, so stratum contrasts keep the pairing)."""
    _, inv = np.unique(np.asarray(subj).astype(str), return_inverse=True)
    P = int(inv.max()) + 1
    cnt = {s: np.bincount(inv[m], minlength=P).astype(np.float64) for s, m in masks.items()}
    sums = {s: {a: np.bincount(inv[m], weights=v[m], minlength=P) for a, v in losses.items()} for s, m in masks.items()}
    rng = np.random.default_rng(seed)
    out = {s: {a: [] for a in losses} for s in masks}
    for start in range(0, n_boot, 250):
        W = rng.multinomial(P, np.full(P, 1.0 / P), size=min(250, n_boot - start)).astype(np.float64)
        for s in masks:
            n = W @ cnt[s]
            for a in losses:
                out[s][a].append((W @ sums[s][a]) / n)
    return {s: {a: np.concatenate(v) for a, v in d.items()} for s, d in out.items()}


def stratum_stats(p: dict, y: np.ndarray, B: dict) -> tuple[dict, dict]:
    from hypo.stats import citl_w, logit, logloss_rows
    from sklearn.metrics import roc_auc_score

    losses = {a: logloss_rows(v, y) for a, v in p.items()}
    est = {a: float(v.mean()) for a, v in losses.items()}
    arms = {}
    for a, v in p.items():
        x = logit(np.clip(v, 1e-6, 1 - 1e-6))
        arms[a] = {"logloss": est[a], "citl": citl_w(x, y.astype(np.float64)),
                   "auroc": float(roc_auc_score(y, v)) if np.unique(y).size == 2 else float("nan")}
    return {"n": int(y.size), "prevalence": float(y.mean()), "arms": arms}, {"est": est, "B": B}


def diff(e: dict, a: str, b: str) -> dict:
    """L(a) − L(b) with its 95 % CI (positive: a is worse)."""
    d = e["B"][a] - e["B"][b]
    return {"est": float(e["est"][a] - e["est"][b]), "ci95": FM._ci(d), "_b": d}


def ratio_r(e: dict, arm: str) -> dict:
    """R = (L_euclid − L_arm) / (L_euclid − L_local) with its 90 % CI; not evaluable on a weak denominator."""
    den_b = e["B"]["euclidean"] - e["B"]["nca_local"]
    den = e["est"]["euclidean"] - e["est"]["nca_local"]
    lo, hi = FM._ci(den_b)
    if den < MIN_DEN or lo <= 0:
        return {"est": None, "verdict": "not evaluable", "denominator": {"est": float(den), "ci95": [lo, hi]}}
    rb = (e["B"]["euclidean"] - e["B"][arm]) / den_b
    ci90 = FM._ci(rb, 0.90)
    return {"est": float((e["est"]["euclidean"] - e["est"][arm]) / den), "ci90": ci90,
            "verdict": "equivalent" if ci90[0] >= R_MARGIN else "not shown equivalent"}


def site_state(st, X: str, var: str, p_base: dict, rows: np.ndarray, units: Path) -> dict | None:
    import polars as pl
    from hypo.external_report import _on

    lm = st.S.lm(X)
    state = state_of(lm, var)
    s_rows = state[rows]
    if s_rows.sum() < 10 or (~s_rows).sum() < 10:
        return None                                     # the intervention is not recorded (or never off) here
    metric = st.metric(X)
    folds = sorted(set(st.S.fold(X).tolist()))
    arm = f"b0_{var}"
    for f in folds:
        path = units / f"{arm}_{X}_f{f}.parquet"
        if not path.exists():
            r, p = predict_b0(st, X, f, metric.enc, np.asarray(metric.w, dtype=np.float64), state)
            pl.DataFrame({"row": r, "p": p}).write_parquet(path)
            LOG.info("state test %s %s fold %d done", X, var, f)
    fs = sorted(units.glob(f"{arm}_{X}_f*.parquet"))
    p = dict(p_base)
    p["b0"] = _on(pl.concat([pl.read_parquet(x) for x in fs]).sort("row"), rows, "p")
    if not np.all(np.isfinite(p["b0"])):
        raise RuntimeError(f"state test {X} {var}: rows without a B0 prediction")
    from hypo.stats import logloss_rows

    y, subj = lm.y[rows], lm.ft.subjects[rows]
    out = {"share_of_landmarks_on": float(s_rows.mean()), "weight_share": weight_share(metric.enc, metric.w, var)}
    masks = {"all": np.ones(rows.size, bool), "off": ~s_rows, "on": s_rows}
    seed = SEED + 10 * sorted(st.S.cfg["cohorts"]).index(X) + list(STATES).index(var)
    Bs = boot_strata({a: logloss_rows(v, y) for a, v in p.items()}, subj, masks, N_BOOT, seed)
    E = {}
    for name, m in masks.items():
        out[name], E[name] = stratum_stats({a: v[m] for a, v in p.items()}, y[m], Bs[name])
    # transport cost by stratum (failure), the stratum contrast, B0's repair and its overall safety
    c_on, c_off = diff(E["on"], "nca_transported", "nca_local"), diff(E["off"], "nca_transported", "nca_local")
    out["cost_on"] = {k: v for k, v in c_on.items() if k != "_b"}
    out["cost_off"] = {k: v for k, v in c_off.items() if k != "_b"}
    out["cost_contrast"] = {"est": c_on["est"] - c_off["est"], "ci95": FM._ci(c_on["_b"] - c_off["_b"])}
    rep = diff(E["on"], "nca_transported", "b0")
    out["repair_on"] = {k: v for k, v in rep.items() if k != "_b"}
    saf = diff(E["all"], "b0", "nca_transported")
    out["safety_all"] = {"est": saf["est"], "upper95": saf["ci95"][1]}
    out["R"] = {"transported": ratio_r(E["all"], "nca_transported"), "b0": ratio_r(E["all"], "b0")}
    if "lightgbm_transported" in p and "lightgbm_local" in p:
        out["lightgbm_cost_on"] = {k: v for k, v in diff(E["on"], "lightgbm_transported", "lightgbm_local").items()
                                   if k != "_b"}
        out["lightgbm_cost_off"] = {k: v for k, v in diff(E["off"], "lightgbm_transported", "lightgbm_local").items()
                                    if k != "_b"}
    failure = c_on["ci95"][0] > 0 and out["cost_contrast"]["ci95"][0] > 0
    repairs = rep["est"] >= MIN_EFFECT and rep["ci95"][0] > 0
    safe = saf["ci95"][1] < SAFE
    out["verdict"] = {"failure": bool(failure), "repairs": bool(repairs), "safe": bool(safe)}
    return out


def run_site(st, X: str, units: Path, sites_states: list[str]) -> dict:
    import polars as pl
    from hypo.external_report import _on, _phase2, _units

    t0 = time.perf_counter()
    metric = st.metric(X)
    w = np.asarray(metric.w, dtype=np.float64)
    T = _units(f"transported_{X}_f*.parquet")
    r0, p0 = predict_b0(st, X, 0, metric.enc, w, None)        # gate: the stored transported arm, fold 0
    gap = np.abs(_on(T, r0, "p") - p0)
    if not np.all(np.isfinite(gap)) or float(gap.max()) > GATE_TOL:
        raise RuntimeError(f"state test {X}: the rebuilt transported arm does not reproduce the stored predictions")
    LOG.info("state test %s: gate passed (max |Δp| = %.2e)", X, float(gap.max()))
    rows = T["row"].to_numpy()
    p = {"nca_transported": T["p"].to_numpy(),
         "nca_local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
         "euclidean": _on(_phase2(X, "euclidean"), rows, "p")}
    comps = _units(f"comps_{X}_f*.parquet")
    if comps is not None and "p_lightgbm" in comps.columns:
        p["lightgbm_transported"] = _on(comps, rows, "p_lightgbm")
    lg = _phase2(X, "lightgbm")
    if lg is not None:
        p["lightgbm_local"] = _on(lg, rows, "p")
    mi_dir = units.parent.with_name("posthoc_fixed_metric") / "units"
    mi_files = sorted(mi_dir.glob(f"mi_transported_{X}_f*.parquet"))
    if mi_files:
        p["mi_transported"] = _on(pl.concat([pl.read_parquet(x) for x in mi_files]).sort("row"), rows, "p")
    ok = np.all([np.isfinite(v) for v in p.values()], axis=0)
    if ok.sum() != rows.size:
        LOG.warning("state test %s: %d rows without a prediction in some arm are left out", X, rows.size - ok.sum())
    rows = rows[ok]
    p = {a: v[ok] for a, v in p.items()}
    res = {"n": int(rows.size), "states": {}}
    for var in sites_states:
        r = site_state(st, X, var, p, rows, units)
        if r is not None:
            res["states"][var] = r
    res["seconds"] = time.perf_counter() - t0
    return res


def rule(sites: dict, weights: dict) -> dict:
    """The pre-specified reading (DECISIONS 2026-11-07)."""
    out = {}
    # P1, weights: sedation's share in the metrics learned with INSPIRE (MIMIC-IV, eICU targets) < INSPIRE target's
    ws = weights.get("sedation_on", {})
    if all(k in ws for k in ("inspire", "mimic", "eicu")):
        out["P1_weight"] = bool(ws["mimic"] < ws["inspire"] and ws["eicu"] < ws["inspire"])
    # P2, sedation: wherever its stratum fails, B0 repairs it; B0 is safe at every site where sedation is recorded
    sed = {X: r["states"]["sedation_on"]["verdict"] for X, r in sites.items() if "sedation_on" in r["states"]}
    if sed:
        out["P2_sedation"] = {
            "failures": [X for X, v in sed.items() if v["failure"]],
            "repaired_where_failed": all(v["repairs"] for v in sed.values() if v["failure"]),
            "safe_everywhere": all(v["safe"] for v in sed.values()),
        }
        f = out["P2_sedation"]
        f["reading"] = ("no failure to repair (the mechanism's weight prediction alone is tested)" if not f["failures"]
                        else "confirmed" if f["repaired_where_failed"] and f["safe_everywhere"] else "not confirmed")
    # P3, ventilation (control): no stratum-specific failure and no repair at any site
    ven = {X: r["states"]["vent_on"]["verdict"] for X, r in sites.items() if "vent_on" in r["states"]}
    if ven:
        out["P3_vent_control"] = {"holds": all(not v["failure"] and not v["repairs"] for v in ven.values()),
                                  "per_site": ven}
    return out


# --------------------------------------------------------------------------- report
def _e(d: dict, k: str = "ci95", n: int = 4) -> str:
    return f"{d['est']:+.{n}f} ({d[k][0]:+.{n}f} to {d[k][1]:+.{n}f})"


def _r(d: dict) -> str:
    if d.get("est") is None:
        return "not evaluable"
    return f"{d['est']:.3f} (90 % CI {d['ci90'][0]:.3f}–{d['ci90'][1]:.3f}; {d['verdict']})"


ARM = {"nca_transported": "NCA transported", "nca_local": "NCA local", "euclidean": "Euclidean",
       "b0": "B0 (same-state neighbours)", "lightgbm_transported": "LightGBM transported",
       "lightgbm_local": "LightGBM local", "mi_transported": "MI transported"}


def render(res: dict) -> str:
    L = ["# The intervention-state mechanism: sedation (test), ventilation (control), vasopressor (reference)", "",
         "Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Cost = L(transported) − L(local) "
         "(positive = transport pays); repair = L(transported) − L(B0) (positive = B0 better); "
         f"{N_BOOT:,} patient resamples. Minimum repair {MIN_EFFECT} nats; B0 safe if its overall upper 95 % limit "
         f"against the transported arm is below {SAFE} nats.", ""]
    for var, role in STATES.items():
        L += [f"## {var} ({role})", ""]
        for X, r in res["sites"].items():
            s = r["states"].get(var)
            if s is None:
                L += [f"- {X}: not recorded at this site (or no stratum of 10 landmarks or more).", ""]
                continue
            v = s["verdict"]
            L += [f"### {X}: on in {s['share_of_landmarks_on']:.1%} of landmarks; weight share in the transported "
                  f"metric {s['weight_share']:.2%}", "",
                  "| stratum | landmarks | prevalence | " + " | ".join(ARM[a] for a in s["all"]["arms"]) + " |",
                  "|---|---|---|" + "---|" * len(s["all"]["arms"])]
            for name in ("all", "off", "on"):
                u = s[name]
                L.append(f"| {name} | {u['n']:,} | {u['prevalence']:.1%} | " + " | ".join(
                    f"{m['logloss']:.4f} / {m['auroc']:.3f} / {m['citl']:+.3f}" for m in u["arms"].values()) + " |")
            L += ["", "(cells: log-loss / AUROC / CITL)", "",
                  f"- transport cost, on: {_e(s['cost_on'])}; off: {_e(s['cost_off'])}; contrast: "
                  f"{_e(s['cost_contrast'])}",
                  f"- B0 repair on the stratum: {_e(s['repair_on'])}; B0 overall loss upper 95 %: "
                  f"{s['safety_all']['upper95']:+.4f}",
                  f"- R transported: {_r(s['R']['transported'])}; R after B0: {_r(s['R']['b0'])}"]
            if "lightgbm_cost_on" in s:
                L.append(f"- LightGBM transported − local, on: {_e(s['lightgbm_cost_on'])}; off: "
                         f"{_e(s['lightgbm_cost_off'])}")
            L += [f"- verdict: failure {'yes' if v['failure'] else 'no'}, repairs {'yes' if v['repairs'] else 'no'}, "
                  f"safe {'yes' if v['safe'] else 'no'}", ""]
    L += ["## Rule (pre-specified)", "", "```", json.dumps(res["rule"], indent=1), "```", "",
          "Weight shares by target (the metric learned on the other two sites):", ""]
    for var, ws in res["weights"].items():
        L.append(f"- {var}: " + ", ".join(f"{X} {v:.2%}" for X, v in ws.items()))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["inspire", "mimic", "eicu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="quick frozen artefacts (tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    st = Stage(quick=a.quick, threads=a.threads)
    od = external_dir().with_name("posthoc_state_test")
    (od / "units").mkdir(parents=True, exist_ok=True)
    res = {"freeze": rec.get("tag"), "sites": {}, "weights": {}}
    for var in STATES:
        res["weights"][var] = {X: weight_share(st.metric(X).enc, st.metric(X).w, var) for X in a.sites}
    for X in a.sites:
        res["sites"][X] = run_site(st, X, od / "units", list(STATES))
    check()
    res["rule"] = rule(res["sites"], res["weights"])
    (od / "state_test.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_state_test.md"
    rep.write_text(render(res), encoding="utf-8")
    LOG.info("state test: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
