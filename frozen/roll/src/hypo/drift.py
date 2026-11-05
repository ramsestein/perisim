"""S4–S6 of the pre-registration (fold 0): drift of the transported metric per site and per eICU hospital, and the
shapes per site and per hospital.

T2's post hoc drift analyses (scripts/posthoc_drift.py, posthoc_drift_missing.py), pre-registered here and
re-implemented for `hypo` (binary outcome; neighbours are distinct patients; LOO anchors drawn from the case base).
Every variable of the metric is a lookup: one point per value (a bin of the metric's grid, a category, 0 / 1 for
binaries and missingness indicators), the block of the weighted embedding being √w_j · g_j(value) / s_j; continuous
variables get their own "missing" point, starting at the median bin's (where the encoder puts a missing value), and
binaries an "unobserved" point at the training prevalence.

- ICE: test landmarks moved along every value of each variable, predicted by the real kNN with the site's case base;
  centred logit of p (centred over the site's value counts), reported as the PDP and the 10–90 % band.
- Re-deformation of one variable: its points freed (case base and queries alike) and fitted to the site, every other
  variable frozen: L-BFGS on a smoothed kNN (each of an anchor's distinct-patient candidates votes σ((r − d²)/τ)
  around the anchor's k-th distance r), patient-level LOO over anchors of the case base, ridge towards the frozen
  points. Candidates come from the frozen metric (one search shared by every variable).
- All variables at once (candidates refreshed each round), and the missing points only.
- Reported: the change in test log-loss of the real kNN (paired, patient bootstrap), the effective weight with the
  frozen and the adapted points, the shape change (weighted Procrustes) and the split-half agreement of the
  deformation (anchors cut into two patient halves, each fitted alone). A variable with (near-)zero weight has
  (near-)zero gradient and cannot gain importance here; the per-site shapes (S5) cover that case.

Outputs, aggregates only (values with < `min_cell` landmarks at the site are not exported): drift_ext-<X>_<site>.json,
drift-hospitals_ext-<X>.json, shapes_<site>.json; fits cached in .pkl beside them (patient level, never exported).
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch

from .casebase import CaseBase, search_distinct
from .config import code_hash
from .data import learning_sample
from .frozen import curve_table, dump, load_pickle, logodds_curve, thresholds
from .metric import fit_encoder, shape_loop
from .stats import PatientBootstrap, ci, logloss_rows

LOG = logging.getLogger("hypo.drift")
MISSING_LABELS = ("missing", "unobserved")


# --------------------------------------------------------------------------- the metric as a lookup
def lookup(enc, w) -> list[dict]:
    """Per metric variable: its embedding columns, value labels, frozen points (metric coordinates) and the factor
    f = √w_j / s_j that turns points back into g_j units. `exact` is False only for a neutral rank axis, quantised
    to each bin's mid-rank."""
    sw = np.sqrt(np.asarray(w, float))
    out = []
    for j, b in enumerate(enc.blocks):
        f = sw[j] / b.scale
        exact = True
        if b.kind in ("continuous", "ordinal"):
            e = b.params["edges"]
            if b.shape is not None:
                pts = np.asarray(b.shape, float)
            else:
                srt = b.params["sorted"]
                mid = (np.searchsorted(srt, e[:-1], "left") + np.searchsorted(srt, e[1:], "right")) / (2.0 * srt.size)
                pts, exact = mid[:, None], False
            med = int(np.clip(np.searchsorted(e, b.params["median"], side="right") - 1, 0, e.size - 2))
            pts = np.vstack([pts, pts[med]])
            labels = [f"[{e[i]:.4g}, {e[i + 1]:.4g})" for i in range(e.size - 1)] + ["missing"]
        elif b.kind == "nominal" and b.shape is not None:
            pts = np.vstack([b.shape, np.zeros((1, b.shape.shape[1]))])
            labels = [str(v) for v in b.params["values"]] + ["unseen"]
        elif b.kind == "nominal":
            pts, labels = np.eye(len(b.params["onehot"])) / np.sqrt(2.0), [str(v) for v in b.params["onehot"]]
        elif b.kind == "binary":
            pts, labels = np.array([[0.0], [1.0], [b.params["prevalence"]]]), ["0", "1", "unobserved"]
        elif b.kind == "missing":
            pts, labels = np.array([[0.0], [1.0]]), ["observed", "missing"]
        else:
            raise ValueError(b.kind)
        out.append({"name": b.name, "kind": b.kind, "cols": np.where(enc.dim_feature == j)[0], "f": float(f),
                    "points": np.asarray(pts, float) * f, "labels": labels, "shaped": b.shape is not None,
                    "exact": exact})
    return out


def value_codes(enc, ft, look: list[dict]) -> np.ndarray:
    """Each landmark's value in every metric variable (n × F): the row of its point in `lookup`."""
    from perisim.metric.encoding import OTHER, _norm_cat

    codes = np.zeros((len(ft), len(enc.blocks)), dtype=np.int64)
    for j, b in enumerate(enc.blocks):
        V = look[j]["points"].shape[0]
        if b.kind in ("continuous", "ordinal"):
            x = ft.num[b.source]
            codes[:, j] = np.where(np.isfinite(x), enc._bin_index(b, ft), V - 1)
        elif b.kind == "nominal" and b.shape is not None:
            idx = enc.value_index(b, ft)
            codes[:, j] = np.where(idx >= 0, idx, V - 1)        # never seen in training → zero contribution
        elif b.kind == "nominal":
            v = _norm_cat(ft.cat[b.source])
            pos = {c: i for i, c in enumerate(b.params["onehot"])}
            codes[:, j] = [pos[c if c in b.params["keep"] else OTHER] for c in v]
        elif b.kind == "binary":
            x = ft.num[b.source]
            codes[:, j] = np.where(np.isfinite(x), (x > 0.5).astype(np.int64), 2)
        else:
            codes[:, j] = (~np.isfinite(ft.num[b.source])).astype(np.int64)
    return codes


def embed(points: list[np.ndarray], codes: np.ndarray) -> np.ndarray:
    return np.hstack([points[j][codes[:, j]] for j in range(len(points))]).astype(np.float32)


def missing_rows(look: list[dict]) -> list[np.ndarray]:
    """Rows of the missing-data points of each variable (the "missing" / "unobserved" values, and the "missing"
    value of each missingness indicator)."""
    return [np.array([i for i, lab in enumerate(lk["labels"]) if lab in MISSING_LABELS or
                      (lk["kind"] == "missing" and lab == "missing")], dtype=np.int64) for lk in look]


# --------------------------------------------------------------------------- smoothed kNN and the fits
def smoothed_p1(d2: torch.Tensor, valid: torch.Tensor, yc: torch.Tensor, k: int, alpha: float, prior1: float,
                smooth: int) -> torch.Tensor:
    """Differentiable P(hypotension) of the kNN over each anchor's candidates: a candidate votes σ((r − d²)/τ), r the
    anchor's k-th distance and τ a quarter of the spread between its (k − smooth)-th and (k + smooth)-th (both held
    fixed within a step), so the votes add up to ≈ k as in the real kNN."""
    big = torch.where(valid, d2.detach(), torch.full_like(d2, torch.finfo(d2.dtype).max))
    srt = torch.sort(big, dim=1).values
    K = d2.shape[1]
    r = srt[:, min(k, K) - 1]
    hi = srt[:, min(k - 1 + smooth, K - 1)]
    hi = torch.where(hi < torch.finfo(d2.dtype).max / 2, hi, r)
    tau = ((hi - srt[:, max(k - 1 - smooth, 0)]) / 4.0).clamp_min(1e-6)
    s = torch.sigmoid((r[:, None] - d2) / tau[:, None]) * valid
    n = s.sum(1)
    n1 = (s * (yc == 1)).sum(1)
    return ((n1 + alpha * prior1 * 2) / (n + alpha * 2)).clamp(1e-9, 1 - 1e-9)


def _nll(p1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(y * torch.log(p1) + (1 - y) * torch.log(1 - p1))


def fit_variable(points: list[np.ndarray], j: int, codes_ref: np.ndarray, a_rows: np.ndarray, y_ref: np.ndarray,
                 cand: tuple, k: int, alpha: float, prior1: float, xd: dict) -> np.ndarray:
    """Points of variable j fitted to the case base (patient-level LOO over the anchors `a_rows`, candidates `cand`
    from the frozen metric), every other variable frozen: only V_j × d_j numbers move."""
    D, I, valid = cand
    P0 = np.asarray(points[j], float)
    P = torch.tensor(P0, dtype=torch.float64, requires_grad=True)
    P0t = torch.tensor(P0, dtype=torch.float64)
    n = len(a_rows)
    va, vc = torch.as_tensor(codes_ref[a_rows, j]), torch.as_tensor(codes_ref[I, j])
    own = ((P0[codes_ref[a_rows, j]][:, None, :] - P0[codes_ref[I, j]]) ** 2).sum(-1)
    rest = torch.as_tensor(np.maximum(D.astype(np.float64) - own, 0.0))
    yc, vt = torch.as_tensor(y_ref[I]), torch.as_tensor(valid)
    ya = torch.as_tensor(y_ref[a_rows].astype(np.float64))
    ridge, smooth = float(xd["ridge"]), int(xd["smooth"])
    opt = torch.optim.LBFGS([P], lr=1.0, max_iter=int(xd["iters"]), line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        diff = P[va][:, None, :] - P[vc]
        p1 = smoothed_p1(rest + (diff * diff).sum(-1), vt, yc, k, alpha, prior1, smooth)
        loss = _nll(p1, ya).mean() + ridge / n * ((P - P0t) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return P.detach().numpy().copy()


def fit_joint(points: list[np.ndarray], free: list[np.ndarray], codes_ref: np.ndarray, code_ref: np.ndarray,
              a_rows: np.ndarray, y_ref: np.ndarray, k: int, alpha: float, prior1: float, fetch: int,
              xd: dict) -> list[np.ndarray]:
    """The rows `free[j]` of every variable's points at once (float32, anchors in chunks with the gradient
    accumulated); candidates refreshed at each round from the current points."""
    F, n = len(points), len(a_rows)
    P0 = [torch.tensor(p, dtype=torch.float32) for p in points]
    fr = [torch.as_tensor(np.asarray(r, dtype=np.int64)) for r in free]
    dl = [torch.zeros((len(r), p.shape[1]), dtype=torch.float32, requires_grad=True) for r, p in zip(fr, P0)]
    params = [d for d, r in zip(dl, fr) if len(r)]
    if not params:
        return [np.asarray(p, float) for p in points]

    def P(j: int) -> torch.Tensor:
        return P0[j].index_add(0, fr[j], dl[j]) if len(fr[j]) else P0[j]

    ca = torch.as_tensor(codes_ref[a_rows])
    ya = torch.as_tensor(y_ref[a_rows].astype(np.float32))
    ridge, smooth, chunk = float(xd["ridge"]), int(xd["smooth"]), int(xd["chunk"])
    for _ in range(int(xd["joint_rounds"])):
        cur = [P(j).detach().numpy() for j in range(F)]
        Er = embed(cur, codes_ref)
        _, I, valid = search_distinct(Er, code_ref, Er[a_rows], code_ref[a_rows], int(xd["k_candidates"]), fetch)
        ci_ = torch.as_tensor(codes_ref[I])                             # n × K × F
        vt, yc = torch.as_tensor(valid), torch.as_tensor(y_ref[I])
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=int(xd["joint_iters"]), line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Ps = [P(j) for j in range(F)]
            tot = 0.0
            for s in range(0, n, chunk):
                a = slice(s, min(s + chunk, n))
                Ea = torch.cat([Ps[j][ca[a, j]] for j in range(F)], 1)
                Ec = torch.cat([Ps[j][ci_[a, :, j]] for j in range(F)], 2)
                p1 = smoothed_p1(((Ea[:, None, :] - Ec) ** 2).sum(-1), vt[a], yc[a], k, alpha, prior1, smooth)
                part = _nll(p1, ya[a]).sum() / n
                part.backward(retain_graph=True)
                tot += float(part.detach())
            reg = ridge / n * sum((d ** 2).sum() for d in params)
            reg.backward()
            return torch.tensor(tot + float(reg.detach()))

        opt.step(closure)
    return [P(j).detach().numpy().astype(float) for j in range(F)]


def knn_p1(points: list[np.ndarray], codes_ref: np.ndarray, code_ref: np.ndarray, y_ref: np.ndarray,
           codes_q: np.ndarray, k: int, alpha: float, prior1: float, fetch: int,
           Er: np.ndarray | None = None) -> np.ndarray:
    """The real kNN (distinct patients; the queries are not in the case base) with the given points."""
    Er = embed(points, codes_ref) if Er is None else Er
    _, I, valid = search_distinct(Er, code_ref, embed(points, codes_q), np.full(len(codes_q), -1), k, fetch)
    n1 = ((y_ref[I] == 1) & valid).sum(1)
    nv = valid.sum(1)
    return (n1 + alpha * prior1 * 2) / (nv + alpha * 2)


def ice(points: list[np.ndarray], look_j: dict, j: int, Er: np.ndarray, code_ref: np.ndarray, y_ref: np.ndarray,
        codes_q: np.ndarray, counts: np.ndarray, k: int, alpha: float, prior1: float, fetch: int) -> dict:
    """ICE of the real kNN along variable j (the case base as it is): centred logit per landmark, centred over the
    values with the site's value counts."""
    Eq = embed(points, codes_q)
    V, n = points[j].shape[0], len(codes_q)
    Q = np.repeat(Eq[None], V, 0)
    Q[:, :, look_j["cols"]] = points[j][:, None, :]
    _, I, valid = search_distinct(Er, code_ref, Q.reshape(V * n, -1), np.full(V * n, -1), k, fetch)
    n1 = ((y_ref[I] == 1) & valid).sum(1)
    p = np.clip((n1 + alpha * prior1 * 2) / (valid.sum(1) + alpha * 2), 1e-6, 1 - 1e-6)
    cur = np.log(p / (1 - p)).reshape(V, n).T                          # n × V
    wv = counts / max(counts.sum(), 1e-12)
    cur -= (cur * wv[None, :]).sum(1, keepdims=True)
    return {"pdp": cur.mean(0), "p10": np.percentile(cur, 10, axis=0), "p90": np.percentile(cur, 90, axis=0)}


def effective_weight(P: np.ndarray, pi: np.ndarray) -> float:
    """Mean squared distance between the points of two random landmarks drawn from the value distribution π."""
    pi = pi / max(pi.sum(), 1e-12)
    mu = (pi[:, None] * P).sum(0)
    return float(2.0 * (pi * ((P - mu) ** 2).sum(1)).sum())


def shape_change(P0: np.ndarray, P1: np.ndarray, pi: np.ndarray) -> float:
    """Share of the adapted points' spread (weighted by π) that a scaled, rotated and shifted copy of the frozen
    points cannot reproduce (weighted Procrustes): 0 = the same shape, only stretched; 1 = unrelated."""
    ok = pi > 0
    if ok.sum() < 3:
        return float("nan")
    p = pi[ok] / pi[ok].sum()
    X = P0[ok] - (p[:, None] * P0[ok]).sum(0)
    Y = P1[ok] - (p[:, None] * P1[ok]).sum(0)
    sx, sy = float((p * (X ** 2).sum(1)).sum()), float((p * (Y ** 2).sum(1)).sum())
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    sv = np.linalg.svd((X * p[:, None]).T @ Y, compute_uv=False)
    return float(np.clip(1.0 - sv.sum() ** 2 / (sx * sy), 0.0, 1.0))


def split_half_r(P0: np.ndarray, PA: np.ndarray, PB: np.ndarray, nA: np.ndarray, nB: np.ndarray,
                 min_n: int) -> float:
    ok = (nA >= min_n) & (nB >= min_n)
    a, b = (PA - P0)[ok].ravel(), (PB - P0)[ok].ravel()
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def g_units(P: np.ndarray, lk: dict) -> list[float]:
    """Points back in g_j units: log-odds (column 1 − column 0) for a shape, the axis value otherwise."""
    g = P / lk["f"] if lk["f"] > 0 else np.zeros_like(P)
    return (g[:, 1] - g[:, 0]).tolist() if g.shape[1] == 2 and lk["shaped"] else g[:, 0].tolist() \
        if g.shape[1] == 1 else g.tolist()


# --------------------------------------------------------------------------- per site
class _Site:
    """The site's case base and test sample in rotation X (fold `xd['fold']`), as lookups."""

    def __init__(self, stage, X: str, c: str, rows_ref: np.ndarray | None = None, rows_te: np.ndarray | None = None):
        xd = stage.x["drift"]
        self.f = int(xd["fold"])
        lm = stage.S.lm(c)
        tr, te = stage.S.split(c, self.f)
        tr = tr if rows_ref is None else rows_ref
        te = te if rows_te is None else rows_te
        self.metric = stage.metric(X)
        self.look = lookup(self.metric.enc, self.metric.w)
        self.frozen = [lk["points"] for lk in self.look]
        codes = value_codes(self.metric.enc, lm.ft, self.look)
        seed = stage.seed + 100 * self.f
        rng = np.random.default_rng([stage.seed, 17, stage.ci(c)])
        self.te = np.sort(rng.choice(te, size=min(int(xd["test"]), te.size), replace=False))
        self.ice_rows = np.sort(rng.choice(self.te, size=min(int(xd["n_ice"]), self.te.size), replace=False))
        self.codes_ref, self.y_ref = codes[tr], lm.y[tr]
        self.codes_te, self.y_te = codes[self.te], lm.y[self.te]
        self.codes_ice = codes[self.ice_rows]
        subj = lm.ft.subjects.astype(str)
        self.subj_te = subj[self.te]
        _, self.code_ref = np.unique(subj[tr], return_inverse=True)
        cb = CaseBase(self.metric, lm.take(tr), stage.mcfg, E=stage.embed(X, c)[tr])
        self.alpha = cb.fit_alpha(seed)["alpha"]
        self.prior1 = float(self.y_ref.mean())
        self.k, self.fetch = int(stage.mcfg["k"]), int(stage.mcfg.get("fetch_factor", 6))
        self.a_rows = learning_sample(lm.take(tr), int(stage.mcfg["learn"]["per_patient"]), int(xd["anchors"]), seed)
        self.check = float(np.abs(embed(self.frozen, self.codes_te[:200])[:, self._exact_cols()]
                                  - (stage.embed(X, c)[self.te[:200]] * self.metric.sqrt_w_dims)[:, self._exact_cols()]
                                  ).max()) if self.te.size else 0.0
        self.counts = [np.bincount(self.codes_ref[:, j], minlength=lk["points"].shape[0]).astype(float)
                       for j, lk in enumerate(self.look)]
        self.n_ref = len(self.y_ref)

    def _exact_cols(self) -> np.ndarray:
        return np.concatenate([lk["cols"] for lk in self.look if lk["exact"]] or [np.zeros(0, dtype=np.int64)])

    def ll(self, points: list[np.ndarray]) -> np.ndarray:
        p = knn_p1(points, self.codes_ref, self.code_ref, self.y_ref, self.codes_te, self.k, self.alpha, self.prior1,
                   self.fetch)
        return logloss_rows(p, self.y_te)


def _delta(ll1: np.ndarray, ll0: np.ndarray, boot: PatientBootstrap) -> dict:
    d = ll1 - ll0
    bs = [float((boot.rows(wp) * d).sum() / boot.rows(wp).sum()) for wp in boot]
    return {"delta": float(d.mean()), "ci": ci(np.asarray(bs))}


def site_drift(stage, X: str, c: str) -> None:
    out = stage.out / f"drift_ext-{X}_{c}.json"
    if out.exists():
        return
    t = time.perf_counter()
    xd, min_cell = stage.x["drift"], int(stage.x["min_cell"])
    s = _Site(stage, X, c)
    cache = out.with_suffix(".pkl")
    fits = load_pickle(cache) if cache.exists() else {"code_hash": code_hash()}
    boot = PatientBootstrap(s.subj_te, int(stage.x["n_boot"]), stage.seed + 31 * stage.ci(c))
    ll0 = s.ll(s.frozen)
    Er = embed(s.frozen, s.codes_ref)
    cand = search_distinct(Er, s.code_ref, Er[s.a_rows], s.code_ref[s.a_rows], int(xd["k_candidates"]), s.fetch)
    # two patient halves of the anchors
    subj_a = s.code_ref[s.a_rows]
    half = np.random.default_rng([stage.seed, 19, stage.ci(c)]).random(subj_a.max() + 1) < 0.5
    hA, hB = np.where(half[subj_a])[0], np.where(~half[subj_a])[0]
    variables = []
    for j, lk in enumerate(s.look):
        key = f"var:{lk['name']}"
        if key not in fits:
            P1 = fit_variable(s.frozen, j, s.codes_ref, s.a_rows, s.y_ref, cand, s.k, s.alpha, s.prior1, xd)
            sub = lambda h: (cand[0][h], cand[1][h], cand[2][h])  # noqa: E731
            PA = fit_variable(s.frozen, j, s.codes_ref, s.a_rows[hA], s.y_ref, sub(hA), s.k, s.alpha, s.prior1, xd)
            PB = fit_variable(s.frozen, j, s.codes_ref, s.a_rows[hB], s.y_ref, sub(hB), s.k, s.alpha, s.prior1, xd)
            pts = list(s.frozen)
            pts[j] = P1
            fits[key] = {"P1": P1, "PA": PA, "PB": PB, "ll": s.ll(pts),
                         "ice": ice(s.frozen, lk, j, Er, s.code_ref, s.y_ref, s.codes_ice, s.counts[j], s.k, s.alpha,
                                    s.prior1, s.fetch)}
            dump(fits, cache)
        r = fits[key]
        cnt = s.counts[j]
        nA = np.bincount(s.codes_ref[s.a_rows[hA], j], minlength=cnt.size)
        nB = np.bincount(s.codes_ref[s.a_rows[hB], j], minlength=cnt.size)
        keep = cnt >= min_cell
        P0 = s.frozen[j]
        variables.append({
            "name": lk["name"], "kind": lk["kind"], "shaped": lk["shaped"],
            "values": [{"label": lab, "n": int(round(float(n), -1)), "g_frozen": g0, "g_adapted": g1,
                        "ice_pdp": float(pd), "ice_p10": float(a), "ice_p90": float(b)}
                       for lab, n, g0, g1, pd, a, b, k_ in zip(lk["labels"], cnt, g_units(P0, lk), g_units(r["P1"], lk),
                                                               r["ice"]["pdp"], r["ice"]["p10"], r["ice"]["p90"], keep)
                       if k_],
            **_delta(r["ll"], ll0, boot),
            "eff_weight_frozen": effective_weight(P0, cnt), "eff_weight_adapted": effective_weight(r["P1"], cnt),
            "shape_change": shape_change(P0, r["P1"], cnt),
            "split_half_r": split_half_r(P0, r["PA"], r["PB"], nA, nB, min_cell)})
    role = "dry run (in-sample)" if stage.dry else "external" if c == X else "development (control)"
    res = {"rotation": f"external-{X}", "site": c, "role": role,
           "fold": s.f, "n_case_base": s.n_ref, "n_anchors": int(s.a_rows.size), "n_test": int(s.te.size),
           "alpha": s.alpha, "lookup_max_abs_error": s.check, "frozen_logloss": float(ll0.mean()),
           "variables": variables}
    for name, free in (("joint", [np.arange(p.shape[0]) for p in s.frozen]), ("missing", missing_rows(s.look))):
        if name not in fits:
            P = fit_joint(s.frozen, free, s.codes_ref, s.code_ref, s.a_rows, s.y_ref, s.k, s.alpha, s.prior1, s.fetch,
                          xd)
            fits[name] = {"P": P, "ll": s.ll(P)}
            dump(fits, cache)
        P = fits[name]["P"]
        res[name] = {**_delta(fits[name]["ll"], ll0, boot),
                     "variables": {lk["name"]: {"eff_weight_ratio": effective_weight(P[j], s.counts[j]) /
                                                max(effective_weight(s.frozen[j], s.counts[j]), 1e-12),
                                                "shape_change": shape_change(s.frozen[j], P[j], s.counts[j])}
                                   for j, lk in enumerate(s.look)}}
        if name == "missing":
            res[name]["moves"] = {lk["name"]: [{"label": lk["labels"][i], "n": int(round(float(s.counts[j][i]), -1)),
                                                "g_frozen": g_units(s.frozen[j][i:i + 1], lk)[0],
                                                "g_adapted": g_units(P[j][i:i + 1], lk)[0]}
                                               for i in free[j] if s.counts[j][i] >= min_cell]
                                  for j, lk in enumerate(s.look) if len(free[j])}
    res.update(seconds=time.perf_counter() - t, code_hash=code_hash())
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    LOG.info("drift external-%s, %s: %.0f s", X, c, res["seconds"])


def hospital_drift(stage, X: str) -> None:
    """S6: the joint re-deformation with each eligible hospital's own patients as the case base (fold 0)."""
    from .external import hospital_labels

    lm = stage.S.lm(X)
    out = stage.out / f"drift-hospitals_ext-{X}.json"
    if out.exists() or lm.hospital[0] is None:
        return
    t = time.perf_counter()
    xd = stage.x["drift"]
    tr, te = stage.S.split(X, int(xd["fold"]))
    hosp = lm.hospital.astype(str)
    res = {"rotation": f"external-{X}", "site": X, "fold": int(xd["fold"]), "hospitals": {}}
    cache = out.with_suffix(".pkl")
    fits = load_pickle(cache) if cache.exists() else {}
    for gi, (h, lab) in enumerate(hospital_labels(lm, int(stage.x["hospitals"]["min_patients"])).items()):
        rt, rq = tr[hosp[tr] == h], te[hosp[te] == h]
        if rq.size == 0:
            continue
        s = _Site(stage, X, X, rows_ref=rt, rows_te=rq)
        if lab not in fits:
            free = [np.arange(p.shape[0]) for p in s.frozen]
            P = fit_joint(s.frozen, free, s.codes_ref, s.code_ref, s.a_rows, s.y_ref, s.k, s.alpha, s.prior1, s.fetch,
                          xd)
            fits[lab] = {"ll0": s.ll(s.frozen), "ll1": s.ll(P), "P": P}
            dump(fits, cache)
        boot = PatientBootstrap(s.subj_te, int(stage.x["n_boot"]), stage.seed + 37 + gi)
        r = fits[lab]
        res["hospitals"][lab] = {"n_case_base": s.n_ref, "n_test": int(s.te.size), "alpha": s.alpha,
                                 "frozen_logloss": float(r["ll0"].mean()), **_delta(r["ll1"], r["ll0"], boot),
                                 "eff_weight_ratio": {lk["name"]: effective_weight(r["P"][j], s.counts[j]) /
                                                      max(effective_weight(s.frozen[j], s.counts[j]), 1e-12)
                                                      for j, lk in enumerate(s.look)}}
    res.update(seconds=time.perf_counter() - t, code_hash=code_hash())
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    LOG.info("drift per hospital external-%s: %.0f s", X, res["seconds"])


# --------------------------------------------------------------------------- S5: shapes per site and hospital
def _shapes(stage, lm_s, seed: int) -> tuple:
    scfg = stage.mcfg["shapes"]
    enc, dropped = fit_encoder(lm_s, scfg)
    loop = shape_loop(enc, lm_s, scfg, stage.space, seed=seed, threads=stage.threads)
    return enc.with_shapes(loop["shapes"], lm_s.ft), loop, dropped


def _compare(enc_s, enc_t, min_cell: int) -> dict:
    """Per continuous variable shaped in both: weighted correlation and WLS slope of the site's log-odds curve on
    the transported one (interpolated at the site's bin mid-points, inside the transported range)."""
    out = {}
    fi_t = enc_t.feature_index()
    for b in enc_s.blocks:
        if b.kind != "continuous" or b.shape is None or b.name not in fi_t:
            continue
        bt = enc_t.blocks[fi_t[b.name]]
        if bt.shape is None:
            continue
        xs, ls = logodds_curve(enc_s, b.name, b.shape)
        xt, lt = logodds_curve(enc_t, b.name, bt.shape)
        w = np.asarray(b.params["value_counts"], float)
        ok = (xs >= xt[0]) & (xs <= xt[-1]) & (w >= min_cell)
        if ok.sum() < 3:
            continue
        a, c, ww = np.interp(xs[ok], xt, lt), ls[ok], w[ok] / w[ok].sum()
        a0, c0 = a - (ww * a).sum(), c - (ww * c).sum()
        va, vc = float((ww * a0 * a0).sum()), float((ww * c0 * c0).sum())
        out[b.name] = {"r": float((ww * a0 * c0).sum() / np.sqrt(va * vc)) if va > 0 and vc > 0 else float("nan"),
                       "slope": float((ww * a0 * c0).sum() / va) if va > 0 else float("nan"), "bins": int(ok.sum())}
    return out


def site_shapes(stage, c: str) -> None:
    from .external import hospital_labels

    out = stage.out / f"shapes_{c}.json"
    if out.exists():
        return
    t = time.perf_counter()
    xd, min_cell, tcfg = stage.x["drift"], int(stage.x["min_cell"]), stage.x["threshold"]
    f = int(xd["fold"])
    lm = stage.S.lm(c)
    tr, _ = stage.S.split(c, f)
    seed = stage.seed + 100 * f
    pool = lm.take(tr)
    rows = learning_sample(pool, int(stage.mcfg["learn"]["per_patient"]), int(stage.mcfg["learn"]["per_site"]), seed)
    enc, loop, dropped = _shapes(stage, pool.take(rows), seed)
    enc_t = stage.metric(c).enc                         # the metric transported to this site
    res = {"site": c, "fold": f, "n_learn": int(rows.size), "dropped_constant": dropped,
           "n_accepted": loop["n_accepted"], "shaped": {n: d["shaped"] for n, d in loop["shape_diag"].items()},
           "shapes": {b.name: curve_table(enc, b.name, b.shape, min_cell) for b in enc.blocks if b.shape is not None},
           "thresholds": thresholds(enc, loop["curves"], None, tcfg),
           "vs_transported": _compare(enc, enc_t, min_cell), "hospitals": {}}
    if lm.hospital[0] is not None:
        hosp = lm.hospital.astype(str)
        for gi, (h, lab) in enumerate(hospital_labels(lm, int(stage.x["hospitals"]["min_patients"])).items()):
            rh = tr[hosp[tr] == h]
            enc_h, loop_h, _ = _shapes(stage, lm.take(rh), seed + gi + 1)
            res["hospitals"][lab] = {
                "n_learn": int(rh.size), "n_accepted": loop_h["n_accepted"],
                "thresholds": thresholds(enc_h, loop_h["curves"], None, tcfg),
                "shapes": {v: curve_table(enc_h, v, enc_h.blocks[enc_h.feature_index()[v]].shape, min_cell)
                           for v in tcfg["variables"] if v in enc_h.feature_index()
                           and enc_h.blocks[enc_h.feature_index()[v]].shape is not None},
                "vs_transported": _compare(enc_h, enc_t, min_cell)}
    res.update(seconds=time.perf_counter() - t, code_hash=code_hash())
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    LOG.info("shapes %s: %.0f s", c, res["seconds"])
