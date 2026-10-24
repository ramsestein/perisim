"""Outcome-definition sensitivity variants by RELABELLING (decided 2026-10-24).

A variant that only changes the definition of the outcome (T1 without antibiotics, without AKI, alternative
KDIGO thresholds, ward hypoxaemia, T2 absolute categories…) is evaluated with the MAIN metric of each outer fold:
the same embedding and weights, the case base relabelled with the variant's definition, α refitted by patient-
level LOO on the relabelled case base. It answers "do the results change if the outcome is defined another way?"
in minutes instead of a whole CV. Whether the learned metric itself would differ is not assessed (declared).
Variants that change the INPUTS (cutoff margin, diagnosis window) are refitted in full instead.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import polars as pl

from perisim.config import stable_hash
from perisim.evaluation.html import box, page
from perisim.evaluation.report import eval_sections, evaluate_predictions
from perisim.experiment import TARGET_CLASSES, knn_predict, labels_for, load_cohort_data
from perisim.knn.predict import KNNModel
from perisim.utils import read_json, write_json, write_parquet

LOG = logging.getLogger("perisim.sensitivity")


def _frame(ids, subj, y, method, k, P) -> pl.DataFrame:
    d = {"stay_id": ids.astype(str), "subject_id": subj.astype(str), "y": y.astype(np.int64),
         "method": [method] * len(y), "k": np.full(len(y), int(k), dtype=np.int64)}
    for c in range(P.shape[1]):
        d[f"p{c}"] = P[:, c].astype(float)
    return pl.DataFrame(d)


def relabel_variant(main, sub, n_outer: int) -> None:
    """Evaluate the variant `sub` (a Pipeline with the variant's config, targets already built) on the main folds."""
    cfg = sub.cfg
    dev = cfg.experiment["cohorts"]["development"]
    kcfg = cfg.experiment["knn"]
    kp, k_grid = int(kcfg["k_primary"]), [int(k) for k in kcfg["k_grid"]]
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    names = main.feature_names()
    data = load_cohort_data(main.ws.common(dev), main.ws.derived(dev), names, targets_dir=sub.ws.derived(dev))
    pos = {s: i for i, s in enumerate(data.ft.ids.astype(str))}
    report = {}
    for t in cfg.targets_to_model:
        key = stable_hash({"main_cv": main.key("cv"), "variant": cfg.to_dict()["targets"], "t": t})
        out = sub.ws.artifacts() / f"relabel_{t}.json"
        prev = read_json(out)
        if prev and prev.get("_key") == key and not main.force:
            report[t] = prev
            continue
        C = TARGET_CLASSES[t]
        frames = []
        for f in range(n_outer):
            mp = main.ws.predictions("cv", t) / f"fold{f}_model.pkl"
            if not mp.exists():
                LOG.warning("relabel %s: no fold model %s (main CV incomplete) → skipped", sub.variant, mp.name)
                continue
            with open(mp, "rb") as fh:
                model = pickle.load(fh)
            tr_all = np.array([pos[s] for s in model["train_ids"] if s in pos])
            te_all = np.array([pos[s] for s in model["test_ids"] if s in pos])
            y, _ = labels_for(t, data, tr_all, cfg.targets)
            tr, te = tr_all[y[tr_all] >= 0], te_all[y[te_all] >= 0]
            if tr.size < 20 or te.size == 0 or np.unique(y[tr]).size < 2:
                continue
            ft_tr, ft_te = data.ft.take(tr), data.ft.take(te)
            enc, w = model["enc"], model["w"]
            kn = KNNModel(enc.transform(ft_tr), enc.weights_dims(w), y[tr], C, ft_tr.subjects, k_grid)
            kn.fit_alpha_and_ood(kcfg["alpha_grid"], kp, kcfg["ood_percentile"])
            pr, _ = knn_predict(kn, enc.transform(ft_te), ft_te.subjects, k_grid)
            for k in k_grid:
                frames.append(_frame(ft_te.ids, ft_te.subjects, y[te], "perisim", k, pr[k]))
            ke = KNNModel(enc.transform_neutral(ft_tr), enc.neutral_weights_dims(np.ones(enc.n_features)), y[tr], C,
                          ft_tr.subjects, [kp])
            ke.fit_alpha_and_ood(kcfg["alpha_grid"], kp, kcfg["ood_percentile"])
            pe, _ = knn_predict(ke, enc.transform_neutral(ft_te), ft_te.subjects, [kp])
            frames.append(_frame(ft_te.ids, ft_te.subjects, y[te], "euclidean", kp, pe[kp]))
            pri = np.bincount(y[tr], minlength=C) / tr.size
            frames.append(_frame(ft_te.ids, ft_te.subjects, y[te], "prevalence", 0, np.tile(pri, (te.size, 1))))
        if not frames:
            continue
        pred = pl.concat(frames, how="diagonal_relaxed")
        write_parquet(pred, sub.ws.predictions("relabel", t) / "pred.parquet")
        ev = evaluate_predictions(pred, t, kp, nb, cfg.seed)
        res = {"_key": key, "target": t, "mode": "relabel", "evaluation": ev}
        write_json(out, res)
        report[t] = res
    if report:
        sec = [box(f"Sensitivity variant <b>{sub.variant}</b> (outcome definition): evaluated with the MAIN metric of "
                   "each outer fold and the case base relabelled with the variant's definition; α refitted by "
                   "patient-level LOO. Whether the metric itself would differ is not assessed (declared).")]
        for t, r in report.items():
            sec.append(f"<h2>{t.upper()}</h2>")
            sec.extend(eval_sections(r["evaluation"], nb))
        (sub.ws.reports / f"results_{sub.variant}.html").write_text(
            page(f"PERISIM sensitivity — {sub.variant} (relabelled)", sec, "PERISIM · aggregates only"), encoding="utf-8")
