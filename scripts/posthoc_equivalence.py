"""Post-hoc, EXPLORATORY: transport equivalence of the learned metric, by the bioequivalence convention (criterion
fixed with the point estimates already known, not pre-registered; docs/DECISIONS.md 2026-11-05). Nothing is refitted;
the frozen artefacts are only read and checked at the start and the end.

The estimand, on the same test rows, the same local case base and the same k:

    R = (L_euclidean − L_transported) / (L_euclidean − L_local)

the share of the local metric's gain over the Euclidean kNN that the transported metric keeps (1 = the local metric's
whole gain, 0 = none). Criterion (bioequivalence: 90 % CI of the ratio, lower limit 0.80; the upper limit 1.25 does not
apply, R > 1 means the transported metric does better than the local one):

- **equivalent** when the lower limit of R's 90 % CI is ≥ 0.80;
- **not evaluable** when the denominator (the local metric's gain over the Euclidean kNN) has a 95 % CI that includes
  0 or a point estimate below 0.005 nats, since R is then unstable;
- otherwise **not equivalent**.

95 % CIs are reported too, with the absolute cost (Δ log-loss transported − local). 2,000 patient-level resamples,
the same for every quantity of a unit.

A. `hypo`, the external stage's own predictions over the five folds (transported and local) and Phase 2's Euclidean
   kNN, as scripts/posthoc_hypo_strata.py: INSPIRE, MIMIC-IV and eICU, all landmarks and by `vaso_on`.
B. T2, the transport stage's split (test set, local pool, own labels): A(i) the frozen INSPIRE metric with the local
   case base, A(iii) the local PERISIM refit (its saved model), and the Euclidean kNN with the same case base. The
   three are rebuilt from the frozen artefacts, and their log-loss must reproduce the stage's (transport.json) to
   1e-6 before anything is computed.

Usage:  python scripts/posthoc_equivalence.py [--part all|hypo|t2] [--profile local] [--workdir <work>]

Outputs (aggregates only): hypo → work/experiments/hypo/posthoc_equivalence/equivalence.json and
reports/posthoc_equivalence.md; T2 → <workdir>/artifacts/main/posthoc_equivalence.json and
reports/posthoc_equivalence.md.
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
import _frozen  # noqa: E402

_frozen.use_src()      # v0.6: $PERISIM_SRC (a frozen experiment's src) first on sys.path, else ROOT/src

LOG = logging.getLogger("posthoc.equivalence")
N_BOOT = 2000
SEED = 20261105
LOWER = 0.80                  # bioequivalence lower limit of the ratio
LEVEL = 0.90                  # bioequivalence CI level (two one-sided tests at α = 0.05)
MIN_GAIN = 0.005              # nats: the smallest denominator for which R is read
BLOCK = 250                   # replicates per block (memory)


# --------------------------------------------------------------------------- the statistic
def _pct(b: np.ndarray, level: float) -> list[float]:
    b = b[np.isfinite(b)]
    q = 100 * (1 - level) / 2
    return [float(np.percentile(b, q)), float(np.percentile(b, 100 - q))] if b.size else [float("nan")] * 2


def ratio_bootstrap(ll_e: np.ndarray, ll_t: np.ndarray, ll_l: np.ndarray, subjects: np.ndarray,
                    n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """R, the denominator and the absolute cost, with patient-level bootstrap replicates (multinomial patient weights,
    the same replicates for the three quantities)."""
    num, den = ll_e - ll_t, ll_e - ll_l
    _, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    P = int(inv.max()) + 1
    s_num = np.bincount(inv, weights=num, minlength=P)
    s_den = np.bincount(inv, weights=den, minlength=P)
    s_cnt = np.bincount(inv, minlength=P).astype(float)
    rng = np.random.default_rng(seed)
    R, D, C = [], [], []
    for start in range(0, n_boot, BLOCK):
        W = rng.multinomial(P, np.full(P, 1.0 / P), size=min(BLOCK, n_boot - start)).astype(np.float64)
        a, b, n = W @ s_num, W @ s_den, W @ s_cnt
        with np.errstate(divide="ignore", invalid="ignore"):
            R.append(a / b)
        D.append(b / n)
        C.append((b - a) / n)
    R, D, C = (np.concatenate(x) for x in (R, D, C))
    est = float(num.sum() / den.sum())
    return {"n": int(num.size), "patients": P,
            "R": {"est": est, "ci90": _pct(R, LEVEL), "ci95": _pct(R, 0.95)},
            "gain_local": {"est": float(den.mean()), "ci95": _pct(D, 0.95)},
            "gain_transported": {"est": float(num.mean())},
            "cost": {"est": float((den - num).mean()), "ci95": _pct(C, 0.95)},
            "logloss": {"euclidean": float(ll_e.mean()), "transported": float(ll_t.mean()), "local": float(ll_l.mean())}}


def verdict(r: dict) -> str:
    g = r["gain_local"]
    if not (g["ci95"][0] > 0 and g["est"] >= MIN_GAIN):
        return "not evaluable"
    return "equivalent" if r["R"]["ci90"][0] >= LOWER else "not equivalent"


def unit(ll_e, ll_t, ll_l, subjects, seed) -> dict:
    r = ratio_bootstrap(ll_e, ll_t, ll_l, subjects, seed=seed)
    r["verdict"] = verdict(r)
    return r


# --------------------------------------------------------------------------- A. hypo
def part_hypo(sites: list[str]) -> dict:
    from hypo.external import Stage
    from hypo.external_report import _on, _phase2, _units
    from hypo.frozen import check, external_dir
    from hypo.stats import logloss_rows

    rec = check()
    st = Stage(threads=4)
    res = {"freeze": rec["tag"], "sites": {}}
    for i, X in enumerate(sites):
        lm = st.S.lm(X)
        T = _units(f"transported_{X}_f*.parquet")
        rows = T["row"].to_numpy()
        p = {"transported": T["p"].to_numpy(), "local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
             "euclidean": _on(_phase2(X, "euclidean"), rows, "p")}
        ok = np.all([np.isfinite(v) for v in p.values()], axis=0)
        rows, p = rows[ok], {k: v[ok] for k, v in p.items()}
        y = lm.y[rows]
        ll = {k: logloss_rows(v, y) for k, v in p.items()}
        subj = lm.ft.subjects[rows]
        vaso = lm.ft.num["vaso_on"][rows]
        out = {"all": unit(ll["euclidean"], ll["transported"], ll["local"], subj, SEED + 10 * i)}
        for v in (0, 1):
            m = vaso == v
            if m.sum() and m.sum() < m.size:
                out[f"vaso_on = {v}"] = unit(ll["euclidean"][m], ll["transported"][m], ll["local"][m], subj[m],
                                             SEED + 10 * i + 1 + v)
                out[f"vaso_on = {v}"]["share_of_landmarks"] = float(m.mean())
        res["sites"][X] = out
        LOG.info("equivalence hypo %s: R %.3f, 90%% CI %s -> %s", X, out["all"]["R"]["est"],
                 np.round(out["all"]["R"]["ci90"], 3).tolist(), out["all"]["verdict"])
    check()
    od = external_dir().with_name("posthoc_equivalence")
    od.mkdir(parents=True, exist_ok=True)
    (od / "equivalence.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_equivalence.md"
    rep.write_text(render("`hypo`: hypotension in the next 6 h (external stage, five folds)", res["sites"]),
                   encoding="utf-8")
    LOG.info("equivalence hypo: %s", rep)
    return res


# --------------------------------------------------------------------------- B. T2
def t2_unit(cfg, ws, target: str, c: str, args: dict) -> tuple[dict, dict]:
    """The three predictions of transport analysis A / B on the test set, rebuilt as src/perisim/transport/analyses.py
    builds them (frozen metric, saved local refit, neutral Euclidean geometry)."""
    from perisim.evaluation import metrics as M
    from perisim.transport.analyses import _load, _with_local, context
    from perisim.transport.core import case_base, predict_gammas

    ctx = context(cfg, ws, target, c, args)
    C, X, split = ctx["C"], ctx["X"], ctx["split"]
    y = ctx["lab"]["own"]
    kcfg = cfg.experiment["knn"]
    kp, kg = int(kcfg["k_primary"]), [int(k) for k in kcfg["k_grid"]]
    fm = _load(ws.predictions("external", target) / "final_model.pkl")
    enc, alpha = fm["enc"], dict(fm["alpha"])
    wd = enc.weights_dims(fm["w"])
    te, pool = split["test"], split["pool"]
    q = te[y[te] >= 0]
    sX = X.ft.subjects.astype(str)
    E_X, N_X = enc.transform(X.ft), enc.transform_neutral(X.ft)
    nd = enc.neutral_weights_dims(np.ones(enc.n_features))
    Eq, sq, yq = E_X[q], sX[q], y[q]
    one = {"1": 1.0}
    P_T = predict_gammas(case_base([(E_X[pool], y[pool], sX[pool], True)], wd, C, kcfg, kg, alpha), Eq, sq, kp,
                         one)[0]["1"]
    P_E = predict_gammas(case_base([(N_X[pool], y[pool], sX[pool], True)], nd, C, kcfg, [kp]), N_X[q], sq, kp,
                         one)[0]["1"]
    fpL = _load(ws.predictions("transport", target) / f"{c}_local_model.pkl")
    P_L = predict_gammas(_with_local(fpL["knn"]), fpL["enc"].transform(X.ft.take(q)), sq, kp, one)[0]["1"]
    ll = {"euclidean": M.row_logloss(P_E, yq), "transported": M.row_logloss(P_T, yq), "local": M.row_logloss(P_L, yq)}
    return ll, {"subjects": sq, "seed": ctx["seed"]}


def _published(tj: dict, target: str, c: str) -> dict:
    co = tj["targets"][target]["cohorts"][c]
    A = {a["variant"]: a["logloss"] for a in co["A"] if "logloss" in a}
    e = [b["logloss"] for b in co["B"] if b.get("composition") == "local_euclidean" and b.get("labels") == "own"]
    return {"transported": A["A(i) frozen INSPIRE metric"], "local": A["A(iii) local PERISIM"], "euclidean": e[0]}


def part_t2(profile: str | None, workdir: str | None) -> dict:
    from perisim.config import load_config
    from perisim.pipeline import Pipeline
    from perisim.transport.core import check_hashes, frozen_hashes

    over = {"experiment": {"paths": {"workdir": str(Path(workdir).resolve())}}} if workdir else {}
    pipe = Pipeline(load_config(None, profile, over))
    cfg, ws = pipe.cfg, pipe.ws
    target = "t2"
    tj = json.loads((ws.artifacts() / "transport.json").read_text(encoding="utf-8"))
    # v0.6: the live code and configuration must be T2's frozen ones (not only unchanged during the run)
    _frozen.check_t2_freeze(cfg, ws)
    before = frozen_hashes(ws, [target])
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    args = {"names": pipe.feature_names(), "dev": dev, "externals": externals}
    res = {"target": target, "sites": {}}
    for i, c in enumerate(externals):
        ll, info = t2_unit(cfg, ws, target, c, args)
        pub = _published(tj, target, c)
        for k, v in pub.items():
            if abs(float(ll[k].mean()) - float(v)) > 1e-6:
                raise RuntimeError(f"equivalence T2 {c}: {k} log-loss {ll[k].mean():.6f} does not reproduce the "
                                   f"transport stage's {v:.6f}")
        res["sites"][c] = {"all": unit(ll["euclidean"], ll["transported"], ll["local"], info["subjects"],
                                       SEED + 500 + 10 * i)}
        r = res["sites"][c]["all"]
        LOG.info("equivalence T2 %s: R %.3f, 90%% CI %s -> %s", c, r["R"]["est"], np.round(r["R"]["ci90"], 3).tolist(),
                 r["verdict"])
    after = frozen_hashes(ws, [target])
    check_hashes(before, after)
    res["frozen_hashes"] = {"start": before, "end": after, "unchanged": True}
    (ws.artifacts() / "posthoc_equivalence.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = ws.reports / "posthoc_equivalence.md"
    rep.write_text(render("T2: post-ICU stay (transport stage, fixed test set)", res["sites"]), encoding="utf-8")
    LOG.info("equivalence T2: %s", rep)
    return res


# --------------------------------------------------------------------------- report
def render(title: str, sites: dict) -> str:
    L = [f"# Transport equivalence — {title} (exploratory)", "",
         "Criterion fixed with the point estimates already known (DECISIONS 2026-11-05), by the bioequivalence "
         f"convention: equivalent when the lower limit of the 90 % CI of R is ≥ {LOWER:.2f}. R = (L_euclidean − "
         "L_transported) / (L_euclidean − L_local). Not evaluable when the local gain over the Euclidean kNN has a 95 % "
         f"CI including 0 or is below {MIN_GAIN} nats. {N_BOOT:,} patient-level resamples.", "",
         "| site | rows | R | 90 % CI | 95 % CI | local gain (nats) | cost (nats) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for s, units in sites.items():
        for name, r in units.items():
            lab = s if name == "all" else f"{s}, {name} ({r['share_of_landmarks']:.0%})"
            g, c, R = r["gain_local"], r["cost"], r["R"]
            L.append(f"| {lab} | {r['n']:,} | {R['est']:.3f} | {R['ci90'][0]:.3f}–{R['ci90'][1]:.3f} | "
                     f"{R['ci95'][0]:.3f}–{R['ci95'][1]:.3f} | {g['est']:.4f} ({g['ci95'][0]:.4f}–{g['ci95'][1]:.4f}) | "
                     f"{c['est']:+.4f} ({c['ci95'][0]:+.4f} to {c['ci95'][1]:+.4f}) | {r['verdict']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=["all", "hypo", "t2"], default="all")
    ap.add_argument("--sites", nargs="*", default=["inspire", "mimic", "eicu"], help="hypo's external sites")
    ap.add_argument("--profile", default=None, help="T2's configuration profile (run_local.ps1 passes local)")
    ap.add_argument("--workdir", default=None, help="T2's work folder")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    os.environ.setdefault("PERISIM_EXPERIMENT", "t2")
    if a.part in ("all", "hypo"):
        part_hypo(a.sites)
    if a.part in ("all", "t2"):
        part_t2(a.profile, a.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
