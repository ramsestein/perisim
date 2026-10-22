"""Validación externa (§6.2) — solo tras la congelación (§6.3).

Modelo final: PERISIM ajustado en TODO INSPIRE con el mismo procedimiento anidado.
- Modo A (transporte completo): pesos, tablas VDM y base de casos de INSPIRE; consultas desde la cohorte destino.
- Modo B (transporte de la métrica): pesos y escalas de INSPIRE, base de casos de la cohorte destino, LOO por
  paciente dentro del destino. Variante B-VDM: tablas VDM recalculadas en el destino con CV por paciente.
- eICU: T1 reducido a grave / no grave (se declara como target reducido). T2 con terciles de cada cohorte.
- Sensibilidad: discretización por percentiles de INSPIRE aplicada a MIMIC-IV / eICU (§4.1).
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from perisim.baselines import parametric
from perisim.evaluation.html import box, page, table
from perisim.evaluation.report import evaluate_predictions, p4_metrics
from perisim.experiment import (SEVERE, TARGET_CLASSES, CohortData, fit_perisim, knn_predict, labels_for,
                                load_cohort_data, swift_inputs)
from perisim.knn.predict import KNNModel
from perisim.utils import set_threads, write_json

LOG = logging.getLogger("perisim.external")


def _pred_frame(ids, subj, y, method, k, P) -> pl.DataFrame:
    d = {"stay_id": ids.astype(str), "subject_id": subj.astype(str), "y": y.astype(np.int64),
         "method": [method] * len(y), "k": np.full(len(y), int(k), dtype=np.int64)}
    for c in range(P.shape[1]):
        d[f"p{c}"] = P[:, c].astype(float)
    return pl.DataFrame(d)


def discretize_like(ft_ext, ft_dev):
    """Aplica a la cohorte destino la discretización por percentiles (pasos de 5, recorte p2,5–p97,5) de INSPIRE."""
    grid_q = np.array([0.025] + [i / 20 for i in range(1, 20)] + [0.975])
    out = copy.copy(ft_ext)
    out.num = dict(ft_ext.num)
    for s in ft_ext.specs:
        if s.type == "continuous" and (s.name.startswith("last_") or s.name.startswith("preop_")):
            ref = ft_dev.num[s.name]
            ref = ref[np.isfinite(ref)]
            if ref.size < 20:
                continue
            g = np.unique(np.quantile(ref, grid_q))
            x = ft_ext.num[s.name]
            ok = np.isfinite(x)
            idx = np.abs(x[ok, None] - g[None, :]).argmin(1)
            v = x.copy()
            v[ok] = g[idx]
            out.num[s.name] = v
    return out


def _eval_target_for(target: str, cohort: str, cfg) -> str:
    reduced = cohort in set(cfg.targets["t1"].get("reduced_cohorts") or [])
    if target == "t1" and reduced:
        return "t1_bin"
    return target


def run_external(pipe) -> None:
    cfg, ws = pipe.cfg, pipe.ws
    set_threads(pipe.threads())
    names = pipe.feature_names()
    d7 = pipe.d7()
    st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
    dev = cfg.experiment["cohorts"]["development"]
    kp = int(cfg.experiment["knn"]["k_primary"])
    k_grid = [int(k) for k in cfg.experiment["knn"]["k_grid"]]
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    D = load_cohort_data(ws.common(dev), ws.derived(dev), names)
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    ext_data = {c: load_cohort_data(ws.common(c), ws.derived(c), names) for c in externals}
    out: dict = {"targets": {}, "note": "Validación externa con pipeline congelado (§6.3)."}
    for target in cfg.experiment["targets_to_model"]:
        idx = np.arange(len(D))
        y, thr = labels_for(target, D, idx, cfg.targets)
        tr = idx[y >= 0]
        C = TARGET_CLASSES[target]
        LOG.info("external: modelo final %s en %s (n=%d)", target, dev, len(tr))
        ft_dev = D.ft.take(tr)
        fp = fit_perisim(cfg, ft_dev, y[tr], C, seed=cfg.seed, threads=pipe.threads(), stopping_override=st,
                         exclude_families=set(d7.get("exclude_families", [])), early_stop=True)
        euc = KNNModel(fp.enc.transform(ft_dev), fp.enc.weights_dims(np.ones(fp.enc.n_features)), y[tr], C,
                       ft_dev.subjects, k_grid)
        euc.fit_alpha_and_ood(cfg.experiment["knn"]["alpha_grid"], kp, cfg.experiment["knn"]["ood_percentile"])
        Xm_dev, _ = fp.enc.transform_model(ft_dev)
        lr_f, _ = parametric.fit_logreg(Xm_dev, y[tr], cfg.model_space["baselines"]["logreg_C_grid"],
                                        [(a, b) for a, b in GroupKFold(3).split(Xm_dev, y[tr], ft_dev.subjects)], C)
        sev = SEVERE[target]
        sw = None
        if target == "t1":
            sw, _ = parametric.fit_swift(swift_inputs(ft_dev), (y[tr] == sev).astype(int))
        tres: dict = {"final_model": {"lambda": fp.info["lambda"], "m": fp.info["m"], "alpha": fp.info["alpha"],
                                      "weights": dict(zip(fp.enc.feature_names, np.round(fp.w, 8).tolist())),
                                      "vdm_tables": fp.enc.vdm_tables(int(cfg.experiment["evaluation"]["small_cell_threshold"])),
                                      "n_accepted": fp.loop["n_accepted"], "stop_reason": fp.loop["stop_reason"],
                                      "thresholds": thr}, "cohorts": {}}
        for c, E in ext_data.items():
            et = _eval_target_for(target, c, cfg)
            if target == "t3" and et == "t3" and (E.t1 < 0).all():
                continue
            eidx = np.arange(len(E))
            if et == "t1_bin":
                ye = E.t1_grave.copy()
            else:
                ye, _ = labels_for(target, E, eidx, cfg.targets)   # terciles propios de la cohorte destino
            te = eidx[ye >= 0]
            if te.size < 20:
                continue
            ft_e = E.ft.take(te)
            yt = ye[te]
            frames = []
            res_c: dict = {"n": int(te.size), "eval_target": et}
            variants = {"": ft_e}
            if True:
                variants["_discretizado"] = discretize_like(ft_e, ft_dev)
            for suf, fte in variants.items():
                Ee = fp.enc.transform(fte)
                # ---- modo A
                probs, (Dd, I, valid) = knn_predict(fp.knn, Ee, fte.subjects, k_grid)
                kth, pct, ood = fp.knn.kth_info(Dd, valid, kp)
                res_c[f"ood_rate_modeA{suf}"] = float(ood.mean())
                for k in k_grid:
                    P = probs[k]
                    if et == "t1_bin":
                        P = np.column_stack([1 - P[:, sev], P[:, sev]])
                    frames.append(_pred_frame(fte.ids, fte.subjects, yt, f"A_perisim{suf}", k, P))
                # ---- modo B: base de casos del destino, LOO por paciente, pesos de INSPIRE
                Cb = 2 if et == "t1_bin" else C
                yb = yt if et != "t1_bin" else yt
                kb = KNNModel(Ee, fp.enc.weights_dims(fp.w), yb, Cb, fte.subjects, k_grid)
                kb.alpha = dict(fp.knn.alpha)
                Db, Ib, vb = kb.neighbors(None, loo=True)
                for k in k_grid:
                    frames.append(_pred_frame(fte.ids, fte.subjects, yt, f"B_perisim{suf}", k,
                                              kb.probs(Ib, vb, min(k, kb.k_max))))
                if suf:
                    continue
                # ---- B-VDM: tablas VDM recalculadas en el destino (CV por paciente)
                Pv = _mode_b_vdm(fp, fte, yb, Cb, k_grid, cfg)
                for k in k_grid:
                    frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_vdm_destino", k, Pv[k]))
                # ---- comparadores
                Pe, (_, Ie, ve) = knn_predict(euc, Ee, fte.subjects, [kp])
                Pa = Pe[kp] if et != "t1_bin" else np.column_stack([1 - Pe[kp][:, sev], Pe[kp][:, sev]])
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_euclidean", kp, Pa))
                kbe = KNNModel(Ee, fp.enc.weights_dims(np.ones(fp.enc.n_features)), yb, Cb, fte.subjects, [kp])
                kbe.alpha = dict(euc.alpha)
                _, Ibe, vbe = kbe.neighbors(None, loo=True)
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_euclidean", kp, kbe.probs(Ibe, vbe, min(kp, kbe.k_max))))
                Xe, _ = fp.enc.transform_model(fte)
                Pl = lr_f(Xe)
                if et == "t1_bin":
                    Pl = np.column_stack([1 - Pl[:, sev], Pl[:, sev]])
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_logreg", 0, Pl))
                if sw is not None:
                    pg = sw(swift_inputs(fte))
                    frames.append(_pred_frame(fte.ids, fte.subjects, yt if et == "t1_bin" else (yt == sev).astype(int) + 0,
                                              "swift_bin", 0, np.column_stack([1 - pg, pg])))
                # ---- P4 en el destino (modo B: vecinos del propio destino)
                car = {k_: v[te] for k_, v in E.carried.items() if np.isfinite(v[te]).any()}
                rows = []
                for meth, (II, vv) in {"B_perisim": (Ib, vb), "B_euclidean": (Ibe, vbe)}.items():
                    for var, pr in kb.carried_summary(II, vv, kp, car).items():
                        rows.append(pl.DataFrame({"stay_id": fte.ids.astype(str), "method": meth, "variable": var,
                                                  "pred": pr.astype(float), "obs": car[var].astype(float)}))
                res_c["p4"] = p4_metrics(pl.concat(rows)) if rows else {}
            pred = pl.concat(frames, how="diagonal_relaxed")
            swift = pred.filter(pl.col("method") == "swift_bin")
            pred = pred.filter(pl.col("method") != "swift_bin")
            res_c["evaluation"] = evaluate_predictions(pred.with_columns(
                pl.when(pl.col("method").str.starts_with("A_perisim") & (pl.col("k") == kp) & (pl.col("method") == "A_perisim"))
                .then(pl.lit("perisim")).otherwise(pl.col("method")).alias("method")), et, kp, nb, cfg.seed)
            if swift.height:
                from perisim.evaluation.report import _binary_block

                res_c["swift"] = _binary_block(swift["p1"].to_numpy(), swift["y"].to_numpy(), swift["subject_id"].to_numpy(),
                                               nb, cfg.seed)
            if c == "eicu" and target == "t2":
                st_e = pl.read_parquet(ws.common(c) / "stays_index.parquet").select("stay_id", "apache_pred_hosp_los")
                j = pl.DataFrame({"stay_id": E.ft.ids.astype(str), "los": E.los}).join(st_e, on="stay_id", how="left") \
                    .drop_nulls().filter(pl.col("los").is_not_nan())
                if j.height >= 20:
                    rho = spearmanr(j["apache_pred_hosp_los"].to_numpy(), j["los"].to_numpy()).correlation
                    res_c["apache_iv_los_descriptive"] = {
                        "spearman_rho_vs_post_icu_los": float(rho), "n": int(j.height),
                        "note": "Estimando distinto (estancia hospitalaria total predicha al ingreso en UCI vs estancia "
                                "post-UCI observada): solo referencia descriptiva (§6.4.10)."}
            pred_dir = ws.predictions("external", target)
            from perisim.utils import write_parquet

            write_parquet(pred, pred_dir / f"{c}_pred.parquet")
            tres["cohorts"][c] = res_c
        out["targets"][target] = tres
    write_json(ws.artifacts() / "external.json", out)
    (ws.reports / "external.html").write_text(render_external(out, kp), encoding="utf-8")


def _mode_b_vdm(fp, ft, y, C, k_grid, cfg, n_splits: int = 5) -> dict[int, np.ndarray]:
    """Modo B con tablas VDM recalculadas en el destino: para cada fold f, las tablas se ajustan con los
    demás folds (por paciente) y las consultas de f se resuelven contra el destino excluyendo su paciente."""
    from perisim.metric.encoding import OTHER, _norm_cat

    P = {k: np.zeros((len(y), C)) for k in k_grid}
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(ft.subjects))))
    prior = np.bincount(y, minlength=C) / len(y)
    for tr, te in gkf.split(np.zeros(len(y)), y, ft.subjects):
        enc = copy.deepcopy(fp.enc)
        for b in enc.blocks:
            if b.kind != "nominal":
                continue
            v = _norm_cat(ft.cat[b.source])[tr]
            vv = np.where(np.isin(v, b.params["keep"]), v, OTHER)
            m = enc.m
            table_ = {}
            for cat in np.unique(vv):
                sel = vv == cat
                nc = sel.sum()
                ph = np.bincount(y[tr][sel], minlength=C) / max(nc, 1)
                table_[cat] = ((nc * ph + m * prior) / (nc + m), int(nc))
            b.params = {**b.params, "table": table_, "unseen": prior}
            b.dims = C
            raw = enc._raw_block(b, ft.take(tr))          # δ_j renormalizado en la parte de ajuste del destino
            b.scale = float(np.sqrt(2.0 * raw.var(axis=0).sum())) or 1.0
        enc._index()
        Ee = enc.transform(ft)
        kn = KNNModel(Ee, enc.weights_dims(fp.w), y, C, ft.subjects, k_grid)
        kn.alpha = {k: fp.knn.alpha.get(k, 1.0) for k in k_grid}
        Dq, Iq, vq = kn.neighbors(Ee[te], ft.subjects[te])
        for k in k_grid:
            P[k][te] = kn.probs(Iq, vq, min(k, kn.k_max))
    return P


def render_external(out: dict, kp: int) -> str:
    sec = [box("Modo A: base de casos de INSPIRE. Modo B: pesos de INSPIRE con la base de casos del destino (LOO por "
               "paciente). La diferencia A − B responde a P3. eICU: target reducido grave / no grave.")]
    for t, r in out["targets"].items():
        sec.append(f"<h2>{t.upper()}</h2>")
        fm = r["final_model"]
        sec.append(table(["Modelo final (INSPIRE)", "valor"], [["λ", fm["lambda"]], ["m (VDM)", fm["m"]],
                                                               ["modelos aceptados", fm["n_accepted"]],
                                                               ["parada", fm["stop_reason"]]]))
        for c, rc in r["cohorts"].items():
            ev = rc["evaluation"]
            sec.append(f"<h3>{c} (n={rc['n']}, target {rc['eval_target']})</h3>")
            rows = [[k, s["n"], s["ci"]["logloss"]["mean"], f"{s['ci']['logloss']['ci_low']:.3f}–{s['ci']['logloss']['ci_high']:.3f}",
                     s["ci"]["brier"]["mean"], s.get("auroc_severe"), s.get("auprc_severe")]
                    for k, s in ev["methods"].items() if k.endswith(f"@{kp}") or "@" not in k or k == "perisim@%d" % kp]
            sec.append(table(["Método@k", "n", "Log-loss", "IC 95 %", "Brier", "AUROC grave", "AUPRC grave"], rows))
            sec.append(table(["OOD modo A", "OOD modo A (discretizado)"],
                             [[rc.get("ood_rate_modeA"), rc.get("ood_rate_modeA_discretizado")]]))
            if rc.get("swift"):
                s = rc["swift"]
                sec.append(table(["SWIFT (recalibrado en INSPIRE)", "Log-loss", "Brier", "AUROC"],
                                 [["grave / no grave", s["logloss"], s["brier"], s.get("auroc", {}).get("mean")]]))
            if rc.get("apache_iv_los_descriptive"):
                a = rc["apache_iv_los_descriptive"]
                sec.append(box(f"APACHE IV (eICU): ρ de Spearman con la estancia post-UCI = {a['spearman_rho_vs_post_icu_los']:.3f} "
                               f"(n={a['n']}). {a['note']}", "warn"))
    return page("Validación externa PERISIM (MIMIC-IV, eICU)", sec, "PERISIM · §6.2 · solo agregados")
