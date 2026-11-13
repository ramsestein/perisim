"""A frozen PERISIM metric exported to fnxsql (src/deploy/fnxsql_export.py + fnxsql.perisim): the artefact's steps give
PERISIM's embedding, and fnxsql's KNN (smoothed vote, one neighbour per patient, the case base as the bag) gives the
case base's predictions. Synthetic data only."""

import copy

import numpy as np
import pytest

fnxsql = pytest.importorskip("fnxsql")
from fnxsql import ir  # noqa: E402

from deploy import fnxsql_export as X  # noqa: E402
from hypo.data import concat  # noqa: E402
from hypo.data import load as load_lm  # noqa: E402
from tests.test_hypo_model import CFG, SPACE, _frame  # noqa: E402

CODES = {"a": 1, "b": 2, "c": 3}


@pytest.fixture(scope="module", params=["value", "arc"])
def fitted(request):
    from hypo.metric import fit_metric

    lm = concat([load_lm("inspire", CFG, _frame("inspire", 300, 5, 1)), load_lm("mimic", CFG, _frame("mimic", 400, 5, 2))])
    m = copy.deepcopy(CFG["metric"])
    m["shapes"].update(n_models=4, max_train=2000, max_val=1000, n_explain=600, subsample=300, n_reference_pairs=2000)
    m["nca"].update(epochs=4, n_candidates=100)
    m.update(alpha_queries=500)
    m["shapes"]["embedding"] = request.param
    return lm, m, fit_metric(lm, m, SPACE, seed=3, threads=1)


def _split(lm):
    subj = lm.ft.subjects.astype(str)
    u = np.unique(subj)
    q = np.isin(subj, np.random.default_rng(0).choice(u, size=u.size // 4, replace=False))
    return np.where(~q)[0], np.where(q)[0]


def test_artefact_steps_reproduce_the_embedding(fitted):
    lm, mcfg, metric = fitted
    art = X.export(metric, k=50, alpha=1.0, codes=CODES)
    assert X.embed_check(art, metric, lm.ft, CODES) <= 2e-6       # PERISIM embeds in float32
    kinds = {s["tipo"] for s in art["preproceso"]}
    assert ("interpolar" in kinds) == (mcfg["shapes"]["embedding"] == "arc") and "categoria" in kinds


def test_fnxsql_knn_reproduces_the_case_base(fitted):
    from hypo.casebase import CaseBase

    lm, mcfg, metric = fitted
    tr, te = _split(lm)
    cb = CaseBase(metric, lm.take(tr), mcfg)
    cb.fit_alpha(7)
    ref = cb.predict(lm.take(te))["p"]
    art = X.export(metric, k=cb.k, alpha=cb.alpha_for(cb.k), distinct=cb.distinct, codes=CODES)
    from fnxsql.contrato import Contrato
    from fnxsql.perisim import documento

    names = {e: f"x{i}" for i, e in enumerate(art["entradas"])}
    doc = {"variables": [{"nombre": v} for v in art["entradas"]], "preproceso": art["preproceso"],
           "entrada_nucleo": [s["salida"] for s in art["preproceso"]],
           "nucleo": {"tipo": "vecinos", "k": art["k"], "ponderacion": "suavizada", "p": 2, "pesos": art["pesos"],
                      "alfa": art["alfa"], "prior": None,
                      "bolsa": {"excluir_mismo_paciente": True, "un_vecino_por_paciente": art["un_vecino_por_paciente"]}}}
    Zb = ir.entrada(doc, X.rows(lm.ft.take(tr), art["entradas"], CODES))
    Zq = ir.entrada(doc, X.rows(lm.ft.take(te), art["entradas"], CODES))
    subj = lm.ft.subjects.astype(str)
    p = ir.vecinos(doc, Zq, Zb, lm.y[tr].astype(float), pac_q=subj[te], pac_b=subj[tr])
    diff = np.abs(p - ref)
    # float32 search (faiss) vs double: only near-ties can pick another neighbour
    assert np.median(diff) == 0.0 and (diff <= 1e-9).mean() >= 0.98 and diff.max() <= 1.0 / (cb.k + 2 * art["alfa"]) + 1e-9
    with pytest.raises(ValueError, match="contrato"):          # the document needs contract variables
        documento(art, names, Contrato())
