"""A PERISIM metric trained outside fnxsql and deployed as one SQL query over FakeNex (the "external model" check of
ML_sqlization, as with its external XGBoost): train → export → `fnxsql evaluar` → compare with PERISIM in Python.

Usage (from the perisim repository; ML_sqlization's fnxsql on the Python path, FakeNex reachable over HTTPS):
    python scripts/fnxsql_fakenex.py --fnxsql <ML_sqlization> [--out <folder>] [--threads 8] [--muestra 3] [--lotes 1,10,50]

1. Data: the practice dataset of fnxsql (`salidas/practica/dataset.csv`, `fnxsql practicar`): every HOSP / HOSP_IQ
   admission of the original FakeNex base, 30 variables with DataNex names computed on the server, death during the
   admission as the label, and its 80 / 20 split by patient (`practica.json`). 100 % synthetic data.
2. PERISIM: the metric (`hypo.metric.fit_metric`, `hypo`'s settings, placement `value`) on the training patients; the
   case base of the training admissions gives α (leave-one-patient-out). Exported with `deploy.fnxsql_export` to
   `<out>/perisim_fakenex.json` (aggregates only).
3. fnxsql `evaluar` on that file with test patients: names, variables server = local, SQL within Athena, SQL = if-then
   on the same bag (the whole FakeNex base) to 1e-9, times per batch. Report: `<out>/perisim_fakenex/informe.md`.
4. PERISIM in Python, same bag (downloaded from FakeNex as the SQL sees it), same α: its predictions for the evaluated
   admissions against the SQL's (`<out>/comparacion_perisim.json`). Only near-ties can differ (PERISIM searches in
   float32, the SQL in double).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BINARY_PREFIX = ("ESTADO_CONCIENC", "sex_")


def landmarks(df, names):
    """hypo Landmarks of a DataFrame of admissions (one site, each patient a subject)."""
    import polars as pl

    from hypo.data import Landmarks
    from perisim.features.build import FeatureSpec
    from perisim.features.table import FeatureTable

    specs = [FeatureSpec(name=n, type="binary" if n.startswith(BINARY_PREFIX) else "continuous",
                         group=n.split("_")[0], block="inputs") for n in names]
    num = {n: df[n].to_numpy(dtype=np.float64) for n in names}
    ids = df["episode_ref"].astype("int64").astype(str).to_numpy().astype(object)
    subj = df["patient_ref"].astype("int64").astype(str).to_numpy().astype(object)
    y = df["muere"].to_numpy(dtype=np.int64) if "muere" in df else df["y"].to_numpy(dtype=np.int64)
    n = len(df)
    return Landmarks(FeatureTable(ids, subj, specs, num, {}), y, np.full(n, -1), np.full(n, "fakenex", dtype=object),
                     np.full(n, None, dtype=object), pl.DataFrame({"row": np.arange(n)}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fnxsql", type=Path, required=True, help="the ML_sqlization repository (branch perisim-vecinos)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--muestra", type=int, default=3)
    ap.add_argument("--lotes", default="1,10,50")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--rank-nodes", type=int, default=101, help="see deploy.fnxsql_export.export")
    ap.add_argument("--local-queries", type=int, default=2000)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.fnxsql.resolve()))
    os.environ.setdefault("PERISIM_ARM", "value")
    import pandas as pd

    from deploy.fnxsql_export import export
    from fnxsql import ir, validar
    from fnxsql import contrato as K
    from hypo.casebase import CaseBase
    from hypo.cv import settings
    from hypo.metric import fit_metric

    out = (a.out or a.fnxsql / "salidas" / "perisim").resolve()
    out.mkdir(parents=True, exist_ok=True)
    prac = json.loads((a.fnxsql / "salidas" / "practica" / "practica.json").read_text(encoding="utf-8"))
    names = list(prac["variables"])
    df = pd.read_csv(a.fnxsql / "salidas" / "practica" / "dataset.csv")
    test = set(int(p) for p in prac["pacientes_test"])
    tr = df[~df.patient_ref.isin(test)].reset_index(drop=True)
    print(f"training admissions {len(tr)} ({tr.patient_ref.nunique()} patients), deaths {tr.muere.mean():.1%}", flush=True)

    # 2. PERISIM metric + α on the training case base
    cfg, mcfg, space = settings(False)
    mcfg["shapes"]["embedding"] = "value"
    lm = landmarks(tr, names)
    from hypo.data import learning_sample

    rows = learning_sample(lm, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), a.seed)
    t = time.perf_counter()
    metric = fit_metric(lm.take(rows), mcfg, space, seed=a.seed, threads=a.threads)
    cb = CaseBase(metric, lm, mcfg)
    cb.fit_alpha(a.seed)
    alpha = cb.alpha_for(cb.k)
    print(f"metric fitted in {time.perf_counter() - t:.0f} s; k = {cb.k}, alpha = {alpha:.3f}", flush=True)
    art = export(metric, k=cb.k, alpha=alpha, distinct=cb.distinct, rank_nodes=a.rank_nodes,
                 info={"datos": "fnxsql practica (FakeNex, sintéticos)", "semilla": a.seed,
                       "ingresos_entrenamiento": int(len(tr)), "pesos_por_variable": metric.weights_table()})
    path = out / "perisim_fakenex.json"
    path.write_text(json.dumps(art, ensure_ascii=False), encoding="utf-8")

    # 2b. locally, no server: the artefact (fnxsql's intérprete) against PERISIM, training case base, test admissions
    from deploy.fnxsql_export import embed_check, rows
    from sklearn.metrics import roc_auc_score

    te = df[df.patient_ref.isin(test)].reset_index(drop=True)
    te = te.sample(n=min(a.local_queries, len(te)), random_state=a.seed).reset_index(drop=True)
    lq = landmarks(te, names)
    d_emb = embed_check(art, metric, lq.ft)
    p_per = cb.predict(lq)["p"]
    dl = {"variables": [{"nombre": v} for v in art["entradas"]], "preproceso": art["preproceso"],
          "entrada_nucleo": [x["salida"] for x in art["preproceso"]],
          "nucleo": {"tipo": "vecinos", "k": art["k"], "ponderacion": "suavizada", "p": 2, "pesos": art["pesos"],
                     "alfa": art["alfa"], "prior": None,
                     "bolsa": {"excluir_mismo_paciente": True, "un_vecino_por_paciente": art["un_vecino_por_paciente"]}}}
    p_art = ir.vecinos(dl, ir.entrada(dl, rows(lq.ft, names)), ir.entrada(dl, rows(lm.ft, names)), lm.y.astype(float),
                       pac_q=lq.ft.subjects.astype(str), pac_b=lm.ft.subjects.astype(str))
    dloc = np.abs(p_art - p_per)
    local = {"queries": int(len(te)), "bag": int(len(tr)), "max_embedding_diff": d_emb,
             "approximated_rank_axes": int(sum(1 for x in art["preproceso"] if x.get("aproximado"))),
             "max_abs_diff": float(dloc.max()), "mean_abs_diff": float(dloc.mean()),
             "share_identical_1e9": float((dloc <= 1e-9).mean()),
             "auroc_perisim": float(roc_auc_score(lq.y, p_per)), "auroc_artefact": float(roc_auc_score(lq.y, p_art))}
    print("local check:", json.dumps(local), flush=True)

    # 3. fnxsql evaluar (test patients, bag = FakeNex)
    lotes = tuple(int(x) for x in a.lotes.split(",") if x)
    inf = validar.evaluar(path, None, muestra=a.muestra, lotes=lotes, salida=str(out), solo_pacientes=sorted(test),
                          semilla=a.seed)
    rep = out / "perisim_fakenex"

    # 4. PERISIM in Python with the same bag and α
    obj, origen = ir.cargar(path)
    contrato = K.Contrato()
    variables, _ = K.variables_del_modelo(names, contrato)
    doc = json.loads((rep / "if_then.json").read_text(encoding="utf-8"))
    bag, _ = validar.descargar_bolsa(doc, names)
    bag = bag.rename(columns={"y": "muere"})
    comp = pd.read_csv(rep / "comparacion.csv")
    pids = inf.d["datos"]["sql"]["pacientes"]
    q, _, _ = K.descargar_variables(variables, contrato, pids)
    q = q[q.episode_ref.astype("int64").isin(comp.episode_ref.astype("int64"))].reset_index(drop=True)
    q["muere"] = 0
    cbb = CaseBase(metric, landmarks(bag.reset_index(drop=True), names), mcfg)
    cbb.alpha[cbb.k] = alpha
    p_py = cbb.predict(landmarks(q, names))["p"]
    m = comp.assign(episode_ref=comp.episode_ref.astype("int64")).set_index("episode_ref")
    p_sql = m.loc[q.episode_ref.astype("int64"), "p_sql"].to_numpy(dtype=float)
    d = np.abs(p_py - p_sql)
    step = 1.0 / (cbb.k + 2 * alpha)
    res = {"admissions": int(len(q)), "patients": len(pids), "bag_admissions": int(len(bag)),
           "bag_patients": int(bag.patient_ref.nunique()), "k": cbb.k, "alpha": alpha,
           "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "share_identical_1e9": float((d <= 1e-9).mean()),
           "one_neighbour_step": step, "fnxsql_passes": bool(inf.ok), "local": local}
    (out / "comparacion_perisim.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    ok = inf.ok  # the conversion's own checks (SQL = if-then); the comparisons with PERISIM are reported, not gated
    print(("PASS" if ok else "FAIL") + f": {rep / 'informe.md'}; {out / 'comparacion_perisim.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
