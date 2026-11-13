"""A frozen PERISIM metric (+ the case-base settings fitted at the target) as an fnxsql artefact.

The artefact (`export`) is JSON with only the metric's aggregates: per embedding dimension one fnxsql preprocessing
step of one input (`tabla`: a step function on the bin edges, `interpolar`: the arc placement, `categoria`: a nominal
input), already divided by the block's scale; the NCA weight of each dimension; k, the smoothing α and the vote
rules. `fnxsql.perisim.documento` turns it into an fnxsql document for a contract; fnxsql's intérprete and SQL then
give PERISIM's prediction with the hospital's own patients as the case base:

    distance² = Σ_d w_d (φ_d(x) − φ_d(y))²,   φ_d = the step of dimension d,  w_d = NCA weight of its input
    k nearest patients (other patients only; each patient once, at its closest row)
    p = (n₁ + 2 α π) / (n + 2 α),   π = share of y = 1 in the case base

Supported blocks (geometry "shape"): continuous / ordinal (shaped by bins or by arc, or the neutral rank axis),
nominal (shaped, or the neutral one-hot), binary and missingness indicators. Hierarchical (PCS) inputs are refused.
The neutral rank axis of an unshaped continuous input is a step function on the training values themselves: the
artefact then flags `valores_de_entrenamiento` (with a metric distilled from a synthetic cohort they are synthetic).
"""

from __future__ import annotations

import numpy as np

FORMATO = "perisim_fnxsql_v1"


def _f(x) -> float:
    return float(x)


def _num_code(c: str, codes: dict | None):
    if codes and c in codes:
        return float(codes[c])
    try:
        return float(c)
    except ValueError:
        raise ValueError(f"nominal category {c!r} has no numeric code: pass codes={{category: code}}") from None


def _steps(enc, b, codes: dict | None) -> tuple[list[dict], bool]:
    """fnxsql steps of one block (one per embedding dimension, divided by the block's scale), and whether they carry
    training values (neutral rank axis)."""
    from perisim.metric.encoding import MISSING, OTHER

    s, p, v = float(b.scale), b.params, b.source
    if enc.geometry != "shape":
        raise ValueError("only geometry 'shape' metrics are exported")
    if b.kind == "hier":
        raise ValueError(f"input {v}: hierarchical (PCS) blocks are not exported")
    if b.kind in ("continuous", "ordinal"):
        med = float(p["median"])
        if b.shape is not None and enc._arc(b):
            xn, phi = enc.arc_nodes(b)
            return [{"tipo": "interpolar", "variable": v, "nodos_x": [_f(x) for x in xn],
                     "nodos_y": [_f(y / s) for y in phi], "nulo": _f(np.interp(med, xn, phi) / s)}], False
        if b.shape is not None:
            e = np.asarray(p["edges"], dtype=np.float64)
            g = np.asarray(b.shape, dtype=np.float64)
            i_med = int(np.clip(np.searchsorted(e, med, side="right") - 1, 0, e.size - 2))
            return [{"tipo": "tabla", "variable": v, "cortes": [_f(c) for c in e[1:-1]],
                     "valores": [_f(x / s) for x in g[:, d]], "en_corte": None, "nulo": _f(g[i_med, d] / s)}
                    for d in range(g.shape[1])], False
        srt = np.asarray(p["sorted"], dtype=np.float64)       # neutral rank: (#{t < x} + #{t <= x}) / 2n
        u, cnt = np.unique(srt, return_counts=True)
        n = srt.size
        le = np.cumsum(cnt)                                     # #{t <= u_i}
        lt = le - cnt                                           # #{t <  u_i}
        between = np.concatenate([[0.0], le]) / n               # x in (u_{i-1}, u_i): #{t <= x} = #{t < x} = le_{i-1}
        at = (lt + le) / (2.0 * n)
        r_med = (np.searchsorted(srt, med, "left") + np.searchsorted(srt, med, "right")) / (2.0 * n)
        return [{"tipo": "tabla", "variable": v, "cortes": [_f(x) for x in u], "valores": [_f(x / s) for x in between],
                 "en_corte": [_f(x / s) for x in at], "nulo": _f(r_med / s)}], True
    if b.kind == "binary":
        prev = float(p["prevalence"])                          # x > 0.5 → 1; x = 0.5 → 0
        return [{"tipo": "tabla", "variable": v, "cortes": [0.5], "valores": [0.0, 1.0 / s], "en_corte": [0.0],
                 "nulo": _f(prev / s)}], False
    if b.kind == "missing":
        return [{"tipo": "tabla", "variable": v, "cortes": [], "valores": [0.0], "en_corte": None,
                 "nulo": _f(1.0 / s)}], False
    if b.kind == "nominal":
        if b.shape is not None:                                 # value_index: kept → its row, seen → OTHER, unseen → 0
            g = np.asarray(b.shape, dtype=np.float64)
            vals = list(p["values"])
            pos = {c: i for i, c in enumerate(vals)}
            oth = pos.get(OTHER)

            def row(c):
                if c in pos:
                    return g[pos[c]]
                if c in set(p["seen"]) and oth is not None:
                    return g[oth]
                return np.zeros(g.shape[1])
            cats = [c for c in sorted(set(vals) | set(p["seen"])) if c not in (OTHER, MISSING)]
            return [{"tipo": "categoria", "variable": v,
                     "valores": {repr(_num_code(c, codes)): _f(row(c)[d] / s) for c in cats},
                     "otro": 0.0, "nulo": _f(row(MISSING)[d] / s)} for d in range(g.shape[1])], False
        oh = list(p["onehot"])                                  # neutral: one-hot / √2 (kept categories, else OTHER)
        keep = [c for c in p["keep"] if c not in (OTHER, MISSING)]
        val = _f(1.0 / np.sqrt(2.0) / s)
        miss_to = MISSING if MISSING in set(p["keep"]) else OTHER
        out = []
        for c in oh:
            if c == OTHER:
                out.append({"tipo": "categoria", "variable": v, "valores": {repr(_num_code(x, codes)): 0.0 for x in keep},
                            "otro": val, "nulo": val if miss_to == OTHER else 0.0})
            elif c == MISSING:
                out.append({"tipo": "categoria", "variable": v, "valores": {}, "otro": 0.0, "nulo": val})
            else:
                out.append({"tipo": "categoria", "variable": v, "valores": {repr(_num_code(c, codes)): val},
                            "otro": 0.0, "nulo": 0.0})
        return out, False
    raise ValueError(f"block kind {b.kind!r} not exported")


def export(metric, *, k: int, alpha: float, distinct: bool = True, codes: dict | None = None, info: dict | None = None) -> dict:
    """The artefact of a frozen `hypo.metric.Metric` with the case-base settings (k, α fitted at the target)."""
    enc = metric.enc
    steps, weights, train_values = [], [], False
    w_feat = np.asarray(metric.w, dtype=np.float64)
    for j, b in enumerate(enc.blocks):
        st, tv = _steps(enc, b, codes)
        if len(st) != b.dims:
            raise AssertionError(f"block {b.source}: {len(st)} steps for {b.dims} dimensions")
        train_values |= tv
        for d, x in enumerate(st):
            x["salida"] = f"phi({b.source})" if len(st) == 1 else f"phi{d}({b.source})"
            steps.append(x)
            weights.append(float(w_feat[j]))
    return {"formato": FORMATO, "entradas": sorted({s["variable"] for s in steps}), "preproceso": steps,
            "pesos": weights, "k": int(k), "alfa": float(alpha), "un_vecino_por_paciente": bool(distinct),
            "excluir_mismo_paciente": True, "valores_de_entrenamiento": bool(train_values), "info": info or {}}


def embed_check(artefact: dict, metric, ft, codes: dict | None = None) -> float:
    """Largest |φ of the artefact (fnxsql's intérprete) − PERISIM's embedding| over the rows of `ft`."""
    from fnxsql import ir

    doc = {"variables": [{"nombre": v} for v in artefact["entradas"]], "preproceso": artefact["preproceso"],
           "entrada_nucleo": [x["salida"] for x in artefact["preproceso"]]}
    return float(np.max(np.abs(ir.entrada(doc, rows(ft, artefact["entradas"], codes)) - metric.enc.transform(ft))))


def rows(ft, inputs: list[str], codes: dict | None = None) -> list[dict]:
    """fnxsql rows (input → number, None = missing) of a FeatureTable; nominal inputs through their numeric codes."""
    cols = {}
    for v in inputs:
        c = ft.column(v)
        if v in ft.cat:
            cols[v] = [None if (x is None or (isinstance(x, float) and x != x) or x == "") else _num_code(str(x), codes)
                       for x in c]
        else:
            cols[v] = [None if not np.isfinite(x) else float(x) for x in np.asarray(c, dtype=np.float64)]
    return [{v: cols[v][i] for v in inputs} for i in range(len(ft))]
