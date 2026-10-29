"""Phase 1 smoke run: the whole model on the development sites of one rotation, with a plain patient split.

The metric is learned on the learning sample of 80 % of each development site's patients; each site's case base is
every landmark of those patients; the other 20 % are queried. Only mechanics, cost and sanity: no comparators, no
external site (that is Phases 2–4). Aggregates only (<workdir>/reports/smoke_<rotation>.md / .json).
"""

from __future__ import annotations

import copy
import json
import logging
import time

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .casebase import CaseBase
from .config import REPO, code_hash, load, workdir
from .data import concat, learning_sample
from .data import load as load_lm
from .metric import fit_metric

LOG = logging.getLogger("hypo.run")
QUICK = {"shapes": {"n_models": 8}, "nca": {"epochs": 10}}


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def smoke(external: str, *, quick: bool, threads: int) -> dict:
    cfg = load()
    mcfg = copy.deepcopy(cfg["metric"])
    if quick:
        for k, v in QUICK.items():
            mcfg[k].update(v)
    with open(REPO / "config" / "model_space.yaml", encoding="utf-8") as fh:
        space = yaml.safe_load(fh)
    dev = [c for c in cfg["cohorts"] if c != external]
    seed = int(cfg["seed"])
    t0 = time.perf_counter()
    sites, train, test = {}, {}, {}
    for c in dev:
        lm = load_lm(c, cfg)
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(lm.y, groups=lm.ft.subjects))
        sites[c], train[c], test[c] = lm, lm.take(tr), lm.take(te)
        LOG.info("%s: %d landmarks (%d train, %d test)", c, len(lm), tr.size, te.size)
    pool = concat([train[c] for c in dev])
    rows = learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)
    LOG.info("learning sample: %d landmarks", rows.size)
    metric = fit_metric(pool.take(rows), mcfg, space, seed=seed, threads=threads)
    out = {"rotation": f"external-{external}", "development": dev, "quick": quick, "code_hash": code_hash(),
           "metric": {k: metric.info[k] for k in ("n_learn", "learn_per_site", "dropped_site_constant", "n_features",
                                                   "n_dims", "n_shaped", "n_accepted", "n_competitive", "n_attempts",
                                                   "seconds")},
           "weights": metric.weights_table(),
           "shaped": {n: d["shaped"] for n, d in metric.info["shape_diag"].items()},
           "delta_last": metric.info["delta"][-3:], "nca_loss": metric.info["nca_history"][::max(1, len(
               metric.info["nca_history"]) // 5)], "sites": {}}
    for c in dev:
        t = time.perf_counter()
        cb = CaseBase(metric, train[c], mcfg)
        a = cb.fit_alpha(seed)
        pr = cb.predict(test[c])
        y, p = test[c].y, pr["p"]
        prev = float(train[c].y.mean())
        out["sites"][c] = {"case_base_landmarks": len(train[c]), "test_landmarks": len(y), **a,
                           "auroc": float(roc_auc_score(y, p)), "logloss": _ll(y, p),
                           "logloss_prevalence": _ll(y, np.full_like(p, prev)), "brier": float(np.mean((p - y) ** 2)),
                           "mean_p": float(p.mean()), "prevalence_test": float(y.mean()),
                           "ood_share": float(pr["ood"].mean()), "seconds": time.perf_counter() - t}
        LOG.info("%s: AUROC %.3f, log-loss %.4f (prevalence %.4f)", c, out["sites"][c]["auroc"],
                 out["sites"][c]["logloss"], out["sites"][c]["logloss_prevalence"])
    out["seconds"] = time.perf_counter() - t0
    rep = workdir() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    name = f"smoke_external-{external}{'_quick' if quick else ''}"
    (rep / f"{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (rep / f"{name}.md").write_text(render(out), encoding="utf-8")
    LOG.info("smoke: %s (%.0f s)", rep / f"{name}.md", out["seconds"])
    return out


def render(o: dict) -> str:
    m = o["metric"]
    L = [f"# `hypo` Phase 1 smoke run: development {' + '.join(o['development'])}" + (" (quick)" if o["quick"] else ""),
         "", f"Code {o['code_hash']}. Learning sample {m['n_learn']:,} landmarks ({m['learn_per_site']}); "
         f"{m['n_features']} metric variables, {m['n_shaped']} shaped; {m['n_accepted']} accepted models "
         f"({m['n_competitive']} competitive, {m['n_attempts']} attempts). Left out (constant within every site): "
         f"{', '.join(m['dropped_site_constant']) or 'none'}. Time: shapes {m['seconds']['shapes']:.0f} s, NCA "
         f"{m['seconds']['nca']:.0f} s, total {o['seconds']:.0f} s.", "",
         "| site | case base | test | α | AUROC | log-loss | prevalence log-loss | mean p / prevalence | OOD |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c, s in o["sites"].items():
        L.append(f"| {c} | {s['case_base_landmarks']:,} | {s['test_landmarks']:,} | {s['alpha']:g} | {s['auroc']:.3f} | "
                 f"{s['logloss']:.4f} | {s['logloss_prevalence']:.4f} | {s['mean_p']:.3f} / "
                 f"{s['prevalence_test']:.3f} | {s['ood_share']:.1%} |")
    L += ["", "| variable | weight | shaped |", "|---|---|---|"]
    for n, w in sorted(o["weights"].items(), key=lambda x: -x[1]):
        L.append(f"| {n} | {w:.3f} | {'yes' if o['shaped'].get(n) else ''} |")
    return "\n".join(L) + "\n"
