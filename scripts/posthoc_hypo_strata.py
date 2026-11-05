"""Post-hoc, EXPLORATORY follow-up of `hypo`: where MIMIC-IV's small transport cost sits (not pre-registered; plan
fixed before running, docs/DECISIONS.md 2026-11-04). Reads only saved predictions and the landmarks; the freeze record
is checked at the start and the end. Nothing is refitted.

A. The transport cost by stratum, on the external stage's own predictions over the five folds: Δ log-loss
   transported − local (positive = cost), the stratum's share of the total cost, and the share of the local metric's
   gain over the Euclidean kNN (Phase 2's comparator) that the transported one keeps. Strata: vasopressors running now
   (`vaso_on`) and time since ICU admission (L bins). MIMIC-IV, and eICU as a descriptive contrast.
B. The weight swaps of scripts/posthoc_hypo.py §B (fold 0, S4's test sample) by stratum, from their saved
   per-landmark log-loss: the `vaso_on` swap by `vaso_on`, the L swap and all weights local by L bin.

Usage:  python scripts/posthoc_hypo_strata.py [--sites mimic eicu]

Outputs (aggregates only): work/experiments/hypo/posthoc_strata/strata.json and reports/posthoc_strata.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _frozen  # noqa: E402

_frozen.use_src()      # v0.6: $PERISIM_SRC (a frozen experiment's src) first on sys.path, else ROOT/src

from hypo.external import Stage  # noqa: E402
from hypo.external_report import _on, _phase2, _units  # noqa: E402
from hypo.frozen import check, external_dir, load_pickle  # noqa: E402
from hypo.report import auc_prep, auc_weighted  # noqa: E402
from hypo.stats import PatientBootstrap, ci, logloss_rows  # noqa: E402

LOG = logging.getLogger("hypo.strata")
MIN_DEN = 0.005     # nats per landmark: the smallest denominator for which a share is read (as posthoc_equivalence)
L_BINS = [(0, 6), (6, 12), (12, 24), (24, 48)]      # hours since ICU admission; the last bin includes 48
SWAPS = {"one:vaso_on": "vaso", "one:L": "L", "all": "L"}


def out_dir() -> Path:
    return external_dir().with_name("posthoc_strata")


def strata(vaso: np.ndarray, L: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    out = {"vaso": {"vaso_on = 0": vaso == 0, "vaso_on = 1": vaso == 1}, "L": {}}
    for lo, hi in L_BINS:
        m = (L >= lo) & ((L <= hi) if hi == L_BINS[-1][1] else (L < hi))
        out["L"][f"L {lo}–{hi} h"] = m
    return out


def evaluable(est: float, b: np.ndarray, positive: bool = False) -> bool:
    """A ratio's denominator (mean per landmark `est`, bootstrap replicates `b`) is readable only when its 95 % CI
    excludes 0 and |est| ≥ MIN_DEN (`positive`: and it is > 0). v0.6 (software audit item 9): shares of a cost or
    gain whose CI includes 0 were published (e.g. eICU's −1.68 to 1.09)."""
    lo, hi = ci(b)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    if positive:
        return lo > 0 and est >= MIN_DEN
    return (lo > 0 or hi < 0) and abs(est) >= MIN_DEN


def ratio(num: float, den: float, b: np.ndarray, ok: bool, den_est: float, den_b: np.ndarray) -> dict:
    """{est, ci} of a share, or "not evaluable" (with the denominator) when its denominator is not readable."""
    if not ok:
        return {"est": None, "ci": None, "verdict": "not evaluable",
                "denominator": {"est": float(den_est), "ci": ci(den_b)}}
    return {"est": float(num / den), "ci": ci(b)}


def _r(x: dict, d: int = 2) -> str:
    return "not evaluable" if x.get("est") is None else f"{x['est']:.{d}f} ({x['ci'][0]:.{d}f}–{x['ci'][1]:.{d}f})"


class Sums:
    """Bootstrap replicates of masked sums through per-patient totals (resamples × patients, not × landmarks)."""

    def __init__(self, boot: PatientBootstrap):
        self.inv, self.P = boot.inv, boot.uniq.size
        self.Wp = np.array([wp for wp in boot])

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.Wp @ np.bincount(self.inv[mask], weights=np.asarray(x, float)[mask], minlength=self.P)



def cost_by_stratum(st: Stage, X: str) -> dict:
    """Part A on site X: the stage's transported and local predictions and Phase 2's Euclidean kNN, five folds."""
    lm = st.S.lm(X)
    T = _units(f"transported_{X}_f*.parquet")
    rows = T["row"].to_numpy()
    p = {"transported": T["p"].to_numpy(), "local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
         "euclidean": _on(_phase2(X, "euclidean"), rows, "p")}
    ok = np.all([np.isfinite(v) for v in p.values()], axis=0)
    rows = rows[ok]
    p = {k: v[ok] for k, v in p.items()}
    y = lm.y[rows]
    ll = {k: logloss_rows(v, y) for k, v in p.items()}
    d = ll["transported"] - ll["local"]
    gain_l, gain_t = ll["euclidean"] - ll["local"], ll["euclidean"] - ll["transported"]
    boot = PatientBootstrap(lm.ft.subjects[rows], int(st.x["n_boot"]), st.seed + 83 + st.ci(X))
    sums = Sums(boot)
    S = strata(lm.ft.num["vaso_on"][rows], lm.ft.num["L"][rows])
    allm = np.ones(rows.size, bool)
    one_ = np.ones(rows.size)
    tot_b = sums(d, allm)
    n_b = sums(one_, allm)
    # v0.6: the share of the cost is read only when the total cost (its denominator) is readable
    cost_ok = evaluable(float(d.mean()), tot_b / n_b)

    def one(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0}
        sd = sums(d, mask)
        db = sd / sums(one_, mask)
        nm = sums(one_, mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh_b = sd / tot_b
            gl_b = sums(gain_l, mask)
            ret_b = sums(gain_t, mask) / gl_b
        yy = y[mask]
        # v0.6: kept is read only when the stratum's local gain over the Euclidean kNN is readable (> 0)
        kept_ok = evaluable(float(gain_l[mask].mean()), gl_b / nm, positive=True)
        return {"n": int(mask.sum()), "share_of_landmarks": float(mask.mean()), "prevalence": float(yy.mean()),
                "delta": {"est": float(d[mask].mean()), "ci": ci(db)},
                "share_of_cost": ratio(d[mask].sum(), d.sum(), sh_b, cost_ok, d.mean(), tot_b / n_b),
                "kept": ratio(gain_t[mask].sum(), gain_l[mask].sum(), ret_b, kept_ok, gain_l[mask].mean(), gl_b / nm),
                "logloss": {k: float(v[mask].mean()) for k, v in ll.items()},
                "auroc": {k: auc_weighted(yy, np.ones(yy.size), auc_prep(v[mask])) for k, v in p.items()},
                "_b": db}

    res = {"site": X, "n": int(rows.size), "all": one(allm), "strata": {}}
    for g, ms in S.items():
        res["strata"][g] = {name: one(m) for name, m in ms.items()}
    v0, v1 = res["strata"]["vaso"]["vaso_on = 0"], res["strata"]["vaso"]["vaso_on = 1"]
    Lb = list(res["strata"]["L"].values())
    res["contrasts"] = {
        "vaso_on 1 − 0": {"est": v1["delta"]["est"] - v0["delta"]["est"], "ci": ci(v1["_b"] - v0["_b"])},
        "last − first L bin": {"est": Lb[-1]["delta"]["est"] - Lb[0]["delta"]["est"], "ci": ci(Lb[-1]["_b"] - Lb[0]["_b"])}}
    res["unclassified_vaso"] = int(rows.size - sum(m.sum() for m in S["vaso"].values()))
    for r in [res["all"], *[x for g in res["strata"].values() for x in g.values()]]:
        r.pop("_b", None)
    return res


def swaps_by_stratum(st: Stage, X: str) -> dict:
    """Part B on site X: the §13.1 swaps' per-landmark log-loss on S4's test sample (fold 0), by stratum."""
    xd = st.x["drift"]
    lm = st.S.lm(X)
    _, te = st.S.split(X, int(xd["fold"]))
    rng = np.random.default_rng([st.seed, 17, st.ci(X)])               # hypo.drift._Site's test sample
    te = np.sort(rng.choice(te, size=min(int(xd["test"]), te.size), replace=False))
    fits = load_pickle(external_dir().with_name("posthoc") / f"posthoc_B_ext-{X}.pkl")
    saved = json.loads((external_dir().with_name("posthoc") / f"posthoc_B_ext-{X}.json").read_text(encoding="utf-8"))
    if fits["frozen"].size != te.size or abs(float(fits["frozen"].mean()) - saved["frozen_logloss"]) > 1e-12:
        raise RuntimeError("strata: the saved swaps do not belong to S4's test sample")
    boot = PatientBootstrap(lm.ft.subjects[te], int(st.x["n_boot"]), st.seed + 41 + st.ci(X))   # §13.1's resamples
    S = strata(lm.ft.num["vaso_on"][te], lm.ft.num["L"][te])
    res = {"site": X, "fold": int(xd["fold"]), "n": int(te.size), "swaps": {}}
    sums = Sums(boot)
    one_ = np.ones(te.size)
    for key, g in SWAPS.items():
        if key not in fits:
            continue
        gain = fits[key] - fits["frozen"]
        allm = np.ones(te.size, bool)
        allb = sums(gain, allm) / sums(one_, allm)
        r = {"all": {"est": float(gain.mean()), "ci": ci(allb)}, "strata": {}}
        gain_ok = evaluable(float(gain.mean()), allb)          # v0.6: the share's denominator must be readable
        tot = sums(gain, allm)
        for name, m in S[g].items():
            b = sums(gain, m) / sums(one_, m)
            with np.errstate(divide="ignore", invalid="ignore"):
                sh = ratio(gain[m].sum(), gain.sum(), sums(gain, m) / tot, gain_ok, gain.mean(), allb)
            r["strata"][name] = {"n": int(m.sum()), "est": float(gain[m].mean()), "ci": ci(b),
                                 "share_of_gain": sh["est"], "share_of_gain_ci": sh["ci"],
                                 **({"share_of_gain_verdict": sh["verdict"]} if "verdict" in sh else {})}
        res["swaps"][key] = r
    return res


def _e(r: dict, d: int = 4) -> str:
    return f"{r['est']:+.{d}f} ({r['ci'][0]:+.{d}f} to {r['ci'][1]:+.{d}f})"


def render(A: dict, B: dict) -> str:
    L = ["# `hypo` post hoc: where MIMIC-IV's transport cost sits (exploratory)", "",
         "Plan: docs/DECISIONS.md 2026-11-04. Δ = log-loss transported − local (positive = transport cost); kept = the "
         "share of the local metric's gain over the Euclidean kNN that the transported metric keeps. 95 % CIs from 500 "
         "patient-level resamples. A share is **not evaluable** when its denominator (the total cost; the stratum's local "
         f"gain; the swap's total gain) has a 95 % CI including 0 or is below {MIN_DEN} nats per landmark (v0.6).", ""]
    for X, r in A.items():
        L += [f"## {X}: the five folds ({r['n']:,} landmarks)", "",
              "| stratum | landmarks | prevalence | Δ | share of the cost | kept | AUROC transported / local |",
              "|---|---|---|---|---|---|---|"]
        rows = [("all", r["all"])] + [(n, x) for g in r["strata"].values() for n, x in g.items()]
        for n, x in rows:
            if not x.get("n"):
                continue
            sh = _r(x["share_of_cost"]) if n != "all" else ("1" if x["share_of_cost"].get("est") is not None
                                                             else "not evaluable")
            L.append(f"| {n} | {x['n']:,} ({x['share_of_landmarks']:.1%}) | {x['prevalence']:.1%} | {_e(x['delta'])} | "
                     f"{sh} | {_r(x['kept'])} | "
                     f"{x['auroc']['transported']:.3f} / {x['auroc']['local']:.3f} |")
        L += [""] + [f"- {k}: {_e(v)}" for k, v in r["contrasts"].items()] + [""]
    for X, r in B.items():
        L += [f"## {X}: the §13.1 swaps by stratum (fold 0, {r['n']:,} landmarks; gain = swap − frozen, negative = "
              "better)", "", "| swap | stratum | landmarks | gain | share of the gain |", "|---|---|---|---|---|"]
        for key, s in r["swaps"].items():
            L.append(f"| {key} | all | {r['n']:,} | {_e(s['all'])} | 1 |")
            for n, x in s["strata"].items():
                shg = "not evaluable" if x["share_of_gain"] is None else \
                    f"{x['share_of_gain']:.2f} ({x['share_of_gain_ci'][0]:.2f}–{x['share_of_gain_ci'][1]:.2f})"
                L.append(f"| {key} | {n} | {x['n']:,} | {_e(x)} | {shg} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["mimic", "eicu"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    rec = check()
    st = Stage(threads=4)
    A = {X: cost_by_stratum(st, X) for X in a.sites}
    B = {"mimic": swaps_by_stratum(st, "mimic")} if "mimic" in a.sites else {}
    check()
    out_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / "strata.json").write_text(json.dumps({"freeze": rec["tag"], "A": A, "B": B}, indent=1, default=float),
                                           encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_strata.md"
    rep.write_text(render(A, B), encoding="utf-8")
    LOG.info("strata report: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
