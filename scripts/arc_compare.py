"""Branch arc-metric (docs/prereg_arc.md §5–§6): the paired contrast arc vs value on the same rows.

Usage:  python scripts/arc_compare.py --value <work_value> --arc <work_arc>

Reads the per-row log-losses that scripts/arc_dump.py wrote in each arm's work folder (arc_compare/<unit>.npz),
checks that every unit evaluates the same rows with the same labels in both arms (otherwise it stops), and computes
with ONE set of patient-level bootstrap replicates per unit (2,000 multinomial patient weights):

- primary (§6.1)  Q = (L_euclid − L_arc) / (L_euclid − L_value), the transported metrics: the share of the value
  placement's gain over the Euclidean kNN that the arc placement achieves. Bioequivalence, two-sided: **equivalent**
  when the 90 % CI of Q lies inside [0.80, 1.25]; **not evaluable** when the value placement's gain has a point
  estimate below 0.005 nats or a 95 % CI including 0; otherwise **not equivalent**, with the side named (arc worse
  if the lower limit is < 0.80, arc better if the upper limit is > 1.25).
- primary (§6.1)  D = L_arc − L_value (nats), paired, 95 % CI: arc better (upper < 0), arc worse (lower > 0), or no
  difference.
- secondary (§6.2)  transport within each arm, R = (L_euclid − L_transported) / (L_euclid − L_local), the published
  rule (equivalent when the lower 90 % limit is ≥ 0.80), and ΔR = R_arc − R_value with its 95 % CI.
- secondary (§6.2)  Q and D for the locally learned metrics.

Descriptive (§6.3): for `hypo`'s frozen metrics, the shaped variables' backtracking index 1 − net / arc (how much of
the path along g_j the value placement cancels) and weight in each arm; the curves g_j themselves must be the same in
both arms (largest difference reported; > 1e-9 stops the script).

L_euclid is the value arm's Euclidean kNN; the arc arm's must equal it (the neutral axes do not depend on the arm):
the largest difference is reported and a difference > 1e-9 stops the script.

Outputs (aggregates only): <arc>/arc_compare/compare.json and <arc>/reports/arc_compare.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 20261108
BLOCK = 250
LOWER, UPPER = 0.80, 1.25
LEVEL = 0.90
MIN_GAIN = 0.005
EUCLID_TOL = 1e-9
MIN_CELL = 10


def _pct(b: np.ndarray, level: float) -> list[float]:
    b = b[np.isfinite(b)]
    q = 100 * (1 - level) / 2
    return [float(np.percentile(b, q)), float(np.percentile(b, 100 - q))] if b.size else [float("nan")] * 2


def load(work: Path) -> dict[str, dict]:
    out = {}
    for f in sorted((work / "arc_compare").glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            out[f.stem] = {k: z[k] for k in z.files}
    return out


def pair(v: dict, a: dict, unit: str) -> None:
    """Stops unless both arms evaluated the same rows, patients and labels, and the same Euclidean kNN."""
    for k in ("ids", "subjects", "y"):
        if v[k].shape != a[k].shape or not np.array_equal(v[k], a[k]):
            raise SystemExit(f"[compare] {unit}: the arms do not evaluate the same {k}; the contrast is not paired")
    d = float(np.max(np.abs(v["ll_euclidean"] - a["ll_euclidean"]))) if v["y"].size else 0.0
    if d > EUCLID_TOL:
        raise SystemExit(f"[compare] {unit}: the Euclidean kNN differs between the arms (max {d:.3g} nats)")


def contrast(e, tv, ta, lv, la, subjects, seed: int, n_boot: int = N_BOOT) -> dict:
    """Every statistic of a unit on the same patient-level bootstrap replicates."""
    terms = {"g_v": e - tv, "g_a": e - ta, "d_t": ta - tv, "d_l": la - lv, "gl_v": e - lv, "gl_a": e - la}
    _, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    P = int(inv.max()) + 1
    S = {k: np.bincount(inv, weights=v, minlength=P) for k, v in terms.items()}
    cnt = np.bincount(inv, minlength=P).astype(float)
    rng = np.random.default_rng(seed)
    B = {k: [] for k in ("Q", "Q_local", "D", "D_local", "gain_value", "gain_local_value", "gain_local_arc",
                         "R_value", "R_arc", "dR")}
    for start in range(0, n_boot, BLOCK):
        W = rng.multinomial(P, np.full(P, 1.0 / P), size=min(BLOCK, n_boot - start)).astype(np.float64)
        s = {k: W @ v for k, v in S.items()}
        n = W @ cnt
        with np.errstate(divide="ignore", invalid="ignore"):
            rv, ra = s["g_v"] / s["gl_v"], s["g_a"] / s["gl_a"]
            B["Q"].append(s["g_a"] / s["g_v"])
            B["Q_local"].append(s["gl_a"] / s["gl_v"])
            B["R_value"].append(rv)
            B["R_arc"].append(ra)
            B["dR"].append(ra - rv)
        B["D"].append(s["d_t"] / n)
        B["D_local"].append(s["d_l"] / n)
        B["gain_value"].append(s["g_v"] / n)
        B["gain_local_value"].append(s["gl_v"] / n)
        B["gain_local_arc"].append(s["gl_a"] / n)
    B = {k: np.concatenate(v) for k, v in B.items()}
    m = {k: float(v.sum() / cnt.sum()) for k, v in S.items()}
    est = {"Q": m["g_a"] / m["g_v"], "Q_local": m["gl_a"] / m["gl_v"], "D": m["d_t"], "D_local": m["d_l"],
           "gain_value": m["g_v"], "gain_local_value": m["gl_v"], "gain_local_arc": m["gl_a"],
           "R_value": m["g_v"] / m["gl_v"], "R_arc": m["g_a"] / m["gl_a"]}
    est["dR"] = est["R_arc"] - est["R_value"]
    res = {k: {"est": float(est[k]), "ci90": _pct(B[k], LEVEL), "ci95": _pct(B[k], 0.95)} for k in B}
    res["n"], res["patients"] = int(e.size), P
    res["logloss"] = {"euclidean": float(e.mean()), "value": float(tv.mean()), "arc": float(ta.mean()),
                      "local_value": float(lv.mean()), "local_arc": float(la.mean())}
    return res


def _evaluable(g: dict) -> bool:
    return g["ci95"][0] > 0 and g["est"] >= MIN_GAIN


def readings(r: dict) -> dict:
    q = r["Q"]["ci90"]
    if not _evaluable(r["gain_value"]):
        qv = "not evaluable"
    elif q[0] >= LOWER and q[1] <= UPPER:
        qv = "equivalent"
    else:
        qv = "not equivalent (" + ("arc worse" if q[0] < LOWER else "arc better") + ")"
    d = r["D"]["ci95"]
    dv = "arc better" if d[1] < 0 else ("arc worse" if d[0] > 0 else "no difference")
    rv = {}
    for arm, g in (("value", "gain_local_value"), ("arc", "gain_local_arc")):
        rv[arm] = ("not evaluable" if not _evaluable(r[g]) else
                   "equivalent" if r[f"R_{arm}"]["ci90"][0] >= LOWER else "not equivalent")
    dr = r["dR"]["ci95"]
    return {"Q": qv, "D": dv, "R_value": rv["value"], "R_arc": rv["arc"],
            "dR": "higher with arc" if dr[0] > 0 else ("lower with arc" if dr[1] < 0 else "no difference")}


def shapes(value: Path, arc: Path) -> dict:
    out = {}
    for fa in sorted((arc / "arc_compare").glob("shapes_*.json")):
        fv = value / "arc_compare" / fa.name
        if not fv.is_file():
            continue
        va = json.loads(fa.read_text(encoding="utf-8"))["variables"]
        vv = json.loads(fv.read_text(encoding="utf-8"))["variables"]
        if set(va) != set(vv):
            raise SystemExit(f"[compare] {fa.stem}: the arms shaped different variables")
        rows, dmax = {}, 0.0
        for n in sorted(va, key=lambda k: -va[k]["backtracking"]):
            d = float(np.max(np.abs(np.asarray(va[n]["g"]) - np.asarray(vv[n]["g"]))))
            dmax = max(dmax, d)
            rows[n] = {"backtracking": va[n]["backtracking"], "arc": va[n]["arc"], "net": va[n]["net"],
                       "weight_share_value": vv[n]["weight_share"], "weight_share_arc": va[n]["weight_share"]}
        if dmax > 1e-6:          # g is rounded to 1e-6 in the dump
            raise SystemExit(f"[compare] {fa.stem}: the consensus curves differ between the arms (max {dmax:.3g})")
        out[fa.stem.replace("shapes_", "")] = {"g_max_diff": dmax, "variables": rows}
    return out


def compare(value: Path, arc: Path) -> dict:
    V, A = load(value), load(arc)
    units = sorted(set(V) & set(A))
    res = {"units": {}, "only_value": sorted(set(V) - set(A)), "only_arc": sorted(set(A) - set(V)),
           "rules": {"Q": f"90 % CI inside [{LOWER}, {UPPER}]", "D": "95 % CI", "min_gain": MIN_GAIN,
                     "n_boot": N_BOOT, "seed": SEED}}
    for i, u in enumerate(units):
        v, a = V[u], A[u]
        pair(v, a, u)
        if v["y"].size < MIN_CELL:
            continue
        r = contrast(v["ll_euclidean"], v["ll_transported"], a["ll_transported"], v["ll_local"], a["ll_local"],
                     v["subjects"], SEED + 10 * i)
        r["readings"] = readings(r)
        r["euclidean_max_diff"] = float(np.max(np.abs(v["ll_euclidean"] - a["ll_euclidean"])))
        res["units"][u] = r
    res["shapes"] = shapes(value, arc)
    return res


def render(res: dict) -> str:
    def ci(x, k="ci95", f="{:+.4f}"):
        return f"{f.format(x['est'])} ({f.format(x[k][0])} to {f.format(x[k][1])})"

    L = ["# Arc vs value placement: paired contrast (docs/prereg_arc.md §6)", "",
         f"Patient bootstrap, {N_BOOT} resamples. Q: 90 % CI, equivalent inside [{LOWER}, {UPPER}]. D, ΔR: 95 % CI. "
         "Aggregates only.", "",
         "| unit | n (patients) | L value | L arc | D = L arc − L value | Q (90 % CI) | reading Q | reading D |",
         "|---|---|---|---|---|---|---|---|"]
    for u, r in res["units"].items():
        L.append(f"| {u} | {r['n']:,} ({r['patients']:,}) | {r['logloss']['value']:.4f} | {r['logloss']['arc']:.4f} | "
                 f"{ci(r['D'])} | {ci(r['Q'], 'ci90', '{:.3f}')} | {r['readings']['Q']} | {r['readings']['D']} |")
    L += ["", "## Transport within each arm (secondary)", "",
          "| unit | R value (90 % CI) | R arc (90 % CI) | ΔR = R arc − R value (95 % CI) | readings |", "|---|---|---|---|---|"]
    for u, r in res["units"].items():
        rd = r["readings"]
        L.append(f"| {u} | {ci(r['R_value'], 'ci90', '{:.3f}')} | {ci(r['R_arc'], 'ci90', '{:.3f}')} | "
                 f"{ci(r['dR'], 'ci95', '{:+.3f}')} | value {rd['R_value']}; arc {rd['R_arc']}; ΔR {rd['dR']} |")
    L += ["", "## Locally learned metrics (secondary)", "", "| unit | D local (95 % CI) | Q local (90 % CI) |", "|---|---|---|"]
    for u, r in res["units"].items():
        L.append(f"| {u} | {ci(r['D_local'])} | {ci(r['Q_local'], 'ci90', '{:.3f}')} |")
    for rot, t in res.get("shapes", {}).items():
        L += ["", f"## Shapes of `hypo`'s frozen metric, rotation external-{rot} (descriptive)", "",
              "| variable | backtracking 1 − net/arc | weight value | weight arc |", "|---|---|---|---|"]
        for n, v in t["variables"].items():
            L.append(f"| {n} | {v['backtracking']:.2f} | {v['weight_share_value']:.3f} | {v['weight_share_arc']:.3f} |")
    if res["only_value"] or res["only_arc"]:
        L += ["", f"Units in one arm only: value {res['only_value']}, arc {res['only_arc']}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--value", required=True, help="work folder of arm value")
    ap.add_argument("--arc", required=True, help="work folder of arm arc")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from perisim.arc_guard import marker_arm

    value, arc = Path(a.value).resolve(), Path(a.arc).resolve()
    if marker_arm(value) != "value" or marker_arm(arc) != "arc":
        raise SystemExit("[compare] --value must be arm value's work folder and --arc arm arc's (ARC_METRIC_WORKDIR)")
    res = compare(value, arc)
    (arc / "arc_compare" / "compare.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (arc / "reports").mkdir(exist_ok=True)
    (arc / "reports" / "arc_compare.md").write_text(render(res), encoding="utf-8")
    print(f"[compare] {len(res['units'])} units → {arc / 'reports' / 'arc_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
