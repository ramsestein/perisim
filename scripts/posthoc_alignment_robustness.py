"""Post-hoc, PRE-SPECIFIED robustness of the neighbour-alignment analysis (`hypo` REPORT §15.3; plan:
docs/DECISIONS.md 2026-11-07).

Two questions raised by the 2026-11-07 novelty review:
  R1  Do §15.3's readings depend on k or on the size of the set the neighbours are drawn from (Koepke et al., 2026)?
  R2  Is "other neighbours, the same vote" (transported–local: 14 % of neighbours shared, logits r = 0.91) specific
      to transport, or what any two metrics with that little overlap do?

Inputs: only the per-metric caches of scripts/posthoc_alignment.py (fold 0; one landmark per test patient as the
query; the k = 50 case-base neighbours, the queries' predictions and their weighted embeddings). Numpy only; nothing
is refitted, no frozen artefact is touched. Aggregates only in the outputs.

Usage:  python scripts/posthoc_alignment_robustness.py [--workdir PATH] [--sites inspire mimic eicu]
Outputs: <workdir>/experiments/posthoc_alignment/robustness.json and robustness.md
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 20261107
N_BOOT = 1000
KS_CB = (5, 10, 20, 50)                 # case-base k (the caches hold the first 50)
KS_HUH = (5, 10, 20, 50, 100)           # within-sample k at 1,000 items
N_ITEMS = (500, 1000, 2000, 4000)       # within-sample set sizes (k fixed at 10, and k = n/100)
DRAWS = 5
SIZES = ("n500", "n1000", "n2000", "n4000", "nhalf")
SITES = ("inspire", "mimic", "eicu")


# --------------------------------------------------------------------------- measures
def first_k(I: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    return np.where(valid[:, :k], I[:, :k], -1)


def overlap(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per query, the share of A's k neighbours among B's, averaged over both directions (−1 never matches)."""
    k = A.shape[1]

    def one(X, Y):
        return ((X[:, :, None] == Y[:, None, :]).any(2) & (X >= 0)).sum(1) / k
    return (one(A, B) + one(B, A)) / 2


def knn_self(E: np.ndarray, k: int) -> np.ndarray:
    """Each item's k nearest other items of the same set (squared Euclidean on the weighted embedding)."""
    E = E.astype(np.float64)
    sq = (E * E).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * E @ E.T
    np.fill_diagonal(D, np.inf)
    I = np.argpartition(D, k, axis=1)[:, :k]
    return I


def huh(Ea: np.ndarray, Eb: np.ndarray, n: int, k: int, draws: list[np.ndarray]) -> dict:
    obs = [float(overlap(knn_self(Ea[d], k), knn_self(Eb[d], k)).mean()) for d in draws]
    chance = k / (n - 1)
    o = float(np.mean(obs))
    return {"est": o, "sd_draws": float(np.std(obs, ddof=1)), "chance": chance, "calibrated": (o - chance) / (1 - chance)}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Boot:
    """Query resamples (one query per patient, so a query bootstrap is a patient bootstrap)."""

    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float64)

    def mean(self, v: np.ndarray) -> np.ndarray:
        return self.W @ v / self.W.sum(1)


def ci(b: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(b, a)), float(np.quantile(b, 1 - a))]


# --------------------------------------------------------------------------- one site
def load_units(cache: Path, X: str) -> dict:
    out = {}
    for f in sorted(cache.glob(f"hypo_{X}_*.npz")):
        name = f.stem[len(f"hypo_{X}_"):]
        z = np.load(f)
        out[name] = {"I": z["I"], "valid": z["valid"], "p": z["p_q"], "E": z["Ewq"]}
    need = {"transported", "local", "zscore+uniform"} | {f"{s}_{d}" for s in SIZES for d in "ab"}
    miss = need - set(out)
    if miss:
        raise RuntimeError(f"robustness {X}: missing caches {sorted(miss)}")
    n = {u["p"].size for u in out.values()}
    if len(n) != 1:
        raise RuntimeError(f"robustness {X}: the caches do not share their queries")
    return out


def site(cache: Path, X: str, si: int) -> dict:
    U = load_units(cache, X)
    nq = next(iter(U.values()))["p"].size
    boot = Boot(nq, SEED + si)
    key_pairs = [("transported", "local"), ("nhalf_a", "nhalf_b"), ("zscore+uniform", "local")] + \
                [(f"{s}_a", f"{s}_b") for s in SIZES if s != "nhalf"]
    res = {"n_queries": int(nq), "case_base": {}, "within": {}, "gallery": {}, "checks": {}}

    # R1a: case-base alignment at several k, with the paired differences that carry §15.3's readings
    for k in KS_CB:
        N = {m: first_k(U[m]["I"], U[m]["valid"], k) for m in U}
        ov = {p: overlap(N[p[0]], N[p[1]]) for p in key_pairs}
        r = {f"{a}|{b}": {"est": float(v.mean()), "ci95": ci(boot.mean(v))} for (a, b), v in ov.items()}
        d1 = ov[("transported", "local")] - ov[("nhalf_a", "nhalf_b")]
        d2 = ov[("transported", "local")] - ov[("zscore+uniform", "local")]
        r["transported|local − nhalf_a|nhalf_b"] = {"est": float(d1.mean()), "ci95": ci(boot.mean(d1))}
        r["transported|local − zscore+uniform|local"] = {"est": float(d2.mean()), "ci95": ci(boot.mean(d2))}
        res["case_base"][f"k{k}"] = r

    # R1b: within-sample alignment (Huh et al.) at 1,000 items and several k; R1c: set size with k fixed / k ∝ n
    rng = np.random.default_rng(SEED + 100 + si)
    pick = {n: [np.sort(rng.choice(nq, min(n, nq), replace=False)) for _ in range(DRAWS)] for n in N_ITEMS}
    for k in KS_HUH:
        res["within"][f"k{k}"] = {f"{a}|{b}": huh(U[a]["E"], U[b]["E"], min(1000, nq), k, pick[1000])
                                  for a, b in key_pairs}
    for n in N_ITEMS:
        if n > nq:
            continue
        for mode, k in (("k10", 10), ("k_prop", max(5, n // 100))):
            res["gallery"][f"n{n}_{mode}"] = {"k": k, **{f"{a}|{b}": huh(U[a]["E"], U[b]["E"], n, k, pick[n])
                                                        for a, b in key_pairs}}

    # the pre-specified checks at every setting
    def conv(tab):         # C1: nhalf > n2000 > n500
        g = lambda s: tab[f"{s}_a|{s}_b"]["est"]  # noqa: E731
        return g("nhalf") > g("n2000") > g("n500")
    settings = {**{f"case_base_{k}": v for k, v in res["case_base"].items()},
                **{f"within_{k}": v for k, v in res["within"].items()},
                **{f"gallery_{k}": v for k, v in res["gallery"].items()}}
    res["checks"]["C1_convergence"] = {s: bool(conv(t)) for s, t in settings.items()}
    res["checks"]["C2_transported_below_halves"] = {
        s: bool(t["transported|local − nhalf_a|nhalf_b"]["ci95"][1] < 0) for s, t in
        {f"case_base_{k}": v for k, v in res["case_base"].items()}.items()}
    res["checks"]["C2_within_point"] = {
        s: bool(t["transported|local"]["est"] < t["nhalf_a|nhalf_b"]["est"]) for s, t in settings.items()
        if not s.startswith("case_base")}
    res["checks"]["C3_vs_euclidean"] = {
        s: float(t["transported|local − zscore+uniform|local"]["est"]) for s, t in
        {f"case_base_{k}": v for k, v in res["case_base"].items()}.items()}

    # R2: overlap (case base, k = 50) against the logits' correlation, over every pair of metrics
    N50 = {m: first_k(U[m]["I"], U[m]["valid"], 50) for m in U}
    L = {m: logit(U[m]["p"]) for m in U}
    pairs = []
    for a, b in combinations(sorted(U), 2):
        pairs.append({"a": a, "b": b, "overlap": float(overlap(N50[a], N50[b]).mean()),
                      "r_logit": float(np.corrcoef(L[a], L[b])[0, 1])})
    ref = [p for p in pairs if "transported" not in (p["a"], p["b"])]
    x = np.array([p["overlap"] for p in ref])
    y = np.array([p["r_logit"] for p in ref])
    A = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    s = float(np.sqrt((resid ** 2).sum() / (x.size - 2)))
    tl = next(p for p in pairs if {p["a"], p["b"]} == {"transported", "local"})
    pred = float(beta[0] + beta[1] * tl["overlap"])
    xbar, sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    half = 1.96 * s * np.sqrt(1 + 1 / x.size + (tl["overlap"] - xbar) ** 2 / sxx)
    reading = ("generic: within what non-transported pairs with that overlap do" if abs(tl["r_logit"] - pred) <= half
               else "transport agrees more than its overlap predicts" if tl["r_logit"] > pred
               else "transport agrees less than its overlap predicts")
    loc = {f"{s_}_a|{s_}_b": next(p for p in pairs if {p["a"], p["b"]} == {f"{s_}_a", f"{s_}_b"}) for s_ in SIZES}
    res["R2"] = {"n_pairs_reference": int(x.size), "intercept": float(beta[0]), "slope": float(beta[1]),
                 "resid_sd": s, "transported|local": tl, "predicted_r": pred, "pi95": [pred - half, pred + half],
                 "reading": reading,
                 "local_local": {k_: {"overlap": v["overlap"], "r_logit": v["r_logit"]} for k_, v in loc.items()},
                 "min_overlap_reference": float(x.min()), "max_overlap_reference": float(x.max())}
    return res


def rule(S: dict) -> dict:
    """The pre-specified reading (DECISIONS 2026-11-07)."""
    c1 = all(all(v["checks"]["C1_convergence"].values()) for v in S.values())
    c2 = all(all(v["checks"]["C2_transported_below_halves"].values()) for v in S.values())
    c2w = all(all(v["checks"]["C2_within_point"].values()) for v in S.values())
    c3 = {X: {"inspire_not_closer" if X == "inspire" else "closer":
              (all(d <= 0 for d in v["checks"]["C3_vs_euclidean"].values()) if X == "inspire"
               else all(d > 0 for d in v["checks"]["C3_vs_euclidean"].values()))} for X, v in S.items()}
    return {"C1_convergence_at_every_setting": c1, "C2_transported_below_halves_case_base_every_k": c2,
            "C2_transported_below_halves_within_every_setting": c2w, "C3": c3,
            "section_15_3_robust": bool(c1 and c2 and c2w),
            "R2": {X: v["R2"]["reading"] for X, v in S.items()}}


# --------------------------------------------------------------------------- report
def render(S: dict, R: dict) -> str:
    L = ["# Robustness of the neighbour-alignment analysis (hypo §15.3)", "",
         "Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Fold 0; one query per test patient; "
         f"{N_BOOT} query (= patient) resamples. Inputs: posthoc_alignment's caches only.", ""]
    for X, s in S.items():
        L += [f"## {X} ({s['n_queries']:,} queries)", "", "### Case base, by k", "",
              "| k | transported–local | halves | n500 pair | Euclidean–local | transported–local − halves (95 % CI) |",
              "|---|---|---|---|---|---|"]
        for k, t in s["case_base"].items():
            d = t["transported|local − nhalf_a|nhalf_b"]
            L.append(f"| {k[1:]} | {t['transported|local']['est']:.3f} | {t['nhalf_a|nhalf_b']['est']:.3f} | "
                     f"{t['n500_a|n500_b']['est']:.3f} | {t['zscore+uniform|local']['est']:.3f} | "
                     f"{d['est']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) |")
        L += ["", "### Within a sample (Huh et al.), 1,000 items, by k (raw / chance-calibrated)", "",
              "| k | chance | transported–local | halves | n500 pair | Euclidean–local |", "|---|---|---|---|---|---|"]
        for k, t in s["within"].items():
            c = t["transported|local"]["chance"]
            L.append(f"| {k[1:]} | {c:.3f} | " + " | ".join(
                f"{t[p]['est']:.3f} / {t[p]['calibrated']:.3f}" for p in
                ("transported|local", "nhalf_a|nhalf_b", "n500_a|n500_b", "zscore+uniform|local")) + " |")
        L += ["", "### Set size (within a sample)", "", "| items | k | transported–local | halves | n500 pair |",
              "|---|---|---|---|---|"]
        for g, t in s["gallery"].items():
            L.append(f"| {g.split('_')[0][1:]} | {t['k']} | " + " | ".join(
                f"{t[p]['est']:.3f}" for p in ("transported|local", "nhalf_a|nhalf_b", "n500_a|n500_b")) + " |")
        r2 = s["R2"]
        L += ["", "### Neighbours shared against agreement of the predictions (case base, k = 50)", "",
              f"- reference: {r2['n_pairs_reference']} metric pairs without the transported metric; "
              f"r_logit = {r2['intercept']:.3f} + {r2['slope']:.3f} × overlap (residual SD {r2['resid_sd']:.3f}; "
              f"overlaps {r2['min_overlap_reference']:.2f}–{r2['max_overlap_reference']:.2f})",
              f"- transported–local: overlap {r2['transported|local']['overlap']:.3f}, r_logit "
              f"{r2['transported|local']['r_logit']:.3f}; predicted {r2['predicted_r']:.3f} "
              f"(95 % PI {r2['pi95'][0]:.3f}–{r2['pi95'][1]:.3f}) → **{r2['reading']}**",
              "- local–local pairs: " + "; ".join(f"{k} overlap {v['overlap']:.3f}, r {v['r_logit']:.3f}"
                                                 for k, v in r2["local_local"].items()), ""]
    L += ["## Rule (pre-specified)", "", "```", json.dumps(R, indent=1), "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default=os.environ.get("PERISIM_WORKDIR") or f"work_{os.environ.get('PERISIM_ARM') or 'arc'}")
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    a = ap.parse_args(argv)
    base = Path(a.workdir) / "experiments" / "posthoc_alignment"
    cache = base / "cache" / "units"
    S = {X: site(cache, X, i) for i, X in enumerate(a.sites)}
    R = rule(S)
    (base / "robustness.json").write_text(json.dumps({"sites": S, "rule": R}, indent=1, default=float), encoding="utf-8")
    (base / "robustness.md").write_text(render(S, R), encoding="utf-8")
    print(f"robustness: {base / 'robustness.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
