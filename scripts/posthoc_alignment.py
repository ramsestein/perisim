"""Post-hoc, EXPLORATORY: do metrics learned apart retrieve the same neighbours? The mutual k-nearest-neighbour
alignment of Huh et al. (2024), measured on `hypo` and T2 (not pre-registered; plan fixed before running,
docs/DECISIONS.md 2026-11-04). The frozen artefacts are only read; `hypo`'s and T2's freeze records are checked at the
start and the end.

The measure, for two metrics A and B and a set of queries: the share of each query's k nearest neighbours that A and
B have in common, averaged over the queries (1 = the same neighbours). Two versions:
- in the case base, the neighbours CE-kNN votes with (k = 50, the method's, and 10), a neighbour identified by its
  row (primary) or its patient; CIs from patient resamples of the queries, the same for every pair;
- within the sample (Huh et al.'s recipe): N_HUH queries, one per patient, neighbours among themselves, k = 10, over
  DRAWS_HUH random draws, with a permutation null (Gröger et al., 2026): the correspondence between the two metrics'
  items shuffled, PERMS_PER_DRAW times per draw; calibrated score (alignment − null) / (1 − null).
Beside each pair, the agreement of the predictions on the same queries.

H. `hypo`, fold 0 of each external site (the external stage's case base, one random landmark per test patient as
   queries, distinct-patient neighbours). Metrics: the transported (frozen) one; the H1 arm's local fold-0 metric,
   refitted with its seed; the ablation's 14 variants (scripts/posthoc_hypo_ablation.py); local metrics learned on n
   training patients (SIZES), two disjoint draws each. Every metric that has stage or ablation predictions must
   reproduce them exactly on the queries, but for one vote at a float32 tie in the k-th neighbour (near_ties).
T. T2, MIMIC-IV and eICU (the transport stage's split and k; no distinct-patient rule, as T2). Metrics: the frozen
   INSPIRE metric (A(i)), the local refit (A(iii)), the drift analyses' joint and missing-only re-deformations (their
   cached fits), and the frozen metric through the drift lookup (a check: it must align with the frozen one).
N. T2's local–local reference (DECISIONS 2026-11-05): T's metrics again (their pairs must equal T's exactly), local
   metrics learned on n pool patients (SIZES), two disjoint draws each, and the Euclidean kNN's neutral axes as the
   floor. Each draw is fitted by the transport stage's A(iii) procedure (`_fit`) on its patients' labelled pool
   stays, WORKERS processes side by side with the stage's threads each; its α is chosen by patient-level
   leave-one-out on the whole pool, its case base. The Euclidean kNN uses the frozen α. Each metric's competence is
   its relative log-loss gain over the case base's prevalence, for T2 and for `hypo`'s draws.

Usage:  python scripts/posthoc_alignment.py [--parts H T N] [--sites inspire mimic eicu] [--threads 16]
                                            [--workers 5] [--report-only] [--dry-run]
--dry-run fits N's draws at DRY_SIZES into posthoc_alignment/dryrun/ (a check of the mechanics; H and T are read
from the last run, never recomputed).

Outside src/ on purpose (the frozen code hashes cover it). Caches (patient level, never exported) in
work/experiments/posthoc_alignment/cache/; outputs (aggregates only): alignment_hypo_<site>.json,
alignment_t2_<site>.json, alignment_t2n_<site>.json, posthoc_alignment.json, posthoc_alignment.md,
posthoc_alignment_n.png and posthoc_alignment_competence.png in work/experiments/posthoc_alignment/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _frozen  # noqa: E402

_frozen.use_src()      # v0.6: $PERISIM_SRC (a frozen experiment's src) first on sys.path, else ROOT/src

from hypo.data import learning_sample  # noqa: E402
from hypo.external import Stage  # noqa: E402
from hypo.external_report import _on  # noqa: E402
from hypo.frozen import check, dump, external_dir, load_pickle  # noqa: E402
from hypo.metric import fit_metric  # noqa: E402
from hypo.report import auc_prep, auc_weighted  # noqa: E402
from hypo.stats import PatientBootstrap, ci, logit, logloss_rows  # noqa: E402
from perisim.knn.index import search  # noqa: E402

LOG = logging.getLogger("posthoc.alignment")
FOLD = 0
KS = (50, 10)                 # case-base alignment: the method's k first (primary), then Huh et al.'s
K_HUH, N_HUH, DRAWS_HUH, PERMS_PER_DRAW = 10, 1000, 20, 10
SIZES = [500, 1000, 2000, 4000, "half"]
DRY_SIZES = [100, 200]
N_BOOT = 500
T2_SITES = ["mimic", "eicu"]
WORKERS = 5
DRY = False


def base_dir() -> Path:
    return external_dir().parents[1] / "posthoc_alignment"


def out_dir() -> Path:
    return base_dir() / "dryrun" if DRY else base_dir()


_FREEZE = {"key": None}       # v0.6: hypo's freeze record hashes (set by main): caches are keyed by them


def freeze_key(rec: dict) -> str:
    """Short key of a freeze record's hashes (code, configuration, folds, comparators, final artefacts)."""
    return hashlib.sha256(json.dumps(rec.get("hashes", rec), sort_keys=True, default=str).encode()).hexdigest()[:16]


def cache_dir() -> Path:
    """Patient-level caches, in a folder per freeze key, so that a cache of another frozen state is never reused."""
    return out_dir() / "cache" / (_FREEZE["key"] or "unkeyed")


def reproduction(ref: np.ndarray | None, p: np.ndarray) -> float | None:
    """Largest |ref − p| over the queries; +inf when any query is missing (NaN) on either side, so that the
    reproduction gate cannot pass on missing rows. v0.6: np.nanmax ignored missing rows (and is NaN, which passes
    the `> 1e-9` gate, when every row is missing)."""
    if ref is None:
        return None
    ref, p = np.asarray(ref, float), np.asarray(p, float)
    if ref.shape != p.shape or ref.size == 0 or not (np.isfinite(ref).all() and np.isfinite(p).all()):
        return float("inf")
    return float(np.abs(ref - p).max())


def reproduces(x: float | None, tol: float = 1e-9) -> bool:
    """The gate: None (nothing to reproduce) passes; NaN or inf fails."""
    return x is None or bool(np.isfinite(x) and x <= tol)


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


# --------------------------------------------------------------------------- the measure (pure helpers)
def first_k(I: np.ndarray, valid: np.ndarray, k: int, ident: np.ndarray | None = None) -> np.ndarray:
    """The first k valid neighbours of each query, as row positions or as `ident[row]` (−1 where there are fewer)."""
    J = I[:, :k] if ident is None else ident[I[:, :k]]
    return np.where(valid[:, :k], J, -1)


def overlap(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """Per query, the share of A's k neighbours that are among B's, averaged over both directions (a patient that
    appears twice in one list, possible in T2, counts once per appearance; −1 never matches)."""
    def one(X, Y):
        return ((X[:, :, None] == Y[:, None, :]).any(2) & (X >= 0)).sum(1) / k
    return (one(A, B) + one(B, A)) / 2


def knn_self(E: np.ndarray, k: int) -> np.ndarray:
    """Each item's k nearest other items of the same set (Huh et al.: the item itself excluded)."""
    _, I = search(E, E, k + 1)
    me = I == np.arange(len(E))[:, None]
    me[~me.any(1), -1] = True                    # not retrieved because of ties: drop the farthest instead
    return I[~me].reshape(len(E), k)


class Huh:
    """Huh et al.'s mutual-kNN alignment within the sample, on fixed draws and permutations shared by every pair."""

    def __init__(self, n_items: int, seed: int):
        rng = np.random.default_rng(seed)
        n = min(N_HUH, n_items)
        self.draws = [np.sort(rng.choice(n_items, n, replace=False)) for _ in range(DRAWS_HUH)]
        self.perms = [[rng.permutation(n) for _ in range(PERMS_PER_DRAW)] for _ in range(DRAWS_HUH)]
        self.cache: dict = {}

    def nn(self, name: str, Ew: np.ndarray) -> list[np.ndarray]:
        if name not in self.cache:
            self.cache[name] = [knn_self(np.ascontiguousarray(Ew[d]), K_HUH) for d in self.draws]
        return self.cache[name]

    def pair(self, a: str, b: str) -> dict:
        obs, null = [], []
        for NA, NB, perms in zip(self.cache[a], self.cache[b], self.perms):
            obs.append(float(overlap(NA, NB, K_HUH).mean()))
            for pi in perms:                     # item i under A against item π(i) under B
                inv = np.argsort(pi)
                null.append(float(overlap(NA, inv[NB[pi]], K_HUH).mean()))
        o, nm = float(np.mean(obs)), float(np.mean(null))
        return {"est": o, "sd_draws": float(np.std(obs, ddof=1)), "null_mean": nm,
                "null_p95": float(np.percentile(null, 95)), "calibrated": (o - nm) / (1 - nm),
                "n_items": int(self.draws[0].size), "k": K_HUH}


class CaseBaseAlignment:
    """Case-base alignment of every pair on the same queries and the same patient resamples."""

    def __init__(self, subjects_q: np.ndarray, ident_patient: np.ndarray, seed: int):
        self.boot = PatientBootstrap(subjects_q, N_BOOT, seed)
        self.W = np.array([self.boot.rows(wp) for wp in self.boot])        # N_BOOT × queries
        self.pat = ident_patient
        self.units: dict = {}

    def add(self, name: str, I: np.ndarray, valid: np.ndarray) -> None:
        self.units[name] = {(k, "row"): first_k(I, valid, k) for k in KS}
        self.units[name].update({(k, "patient"): first_k(I, valid, k, self.pat) for k in KS})

    def per_query(self, a: str, b: str) -> dict:
        return {key: overlap(self.units[a][key], self.units[b][key], key[0]) for key in self.units[a]}

    def summary(self, ov: np.ndarray) -> dict:
        B = self.W @ ov / self.W.sum(1)
        return {"est": float(ov.mean()), "ci": ci(B), "boot": B}


def agreement(pa: np.ndarray, pb: np.ndarray) -> dict:
    """Prediction agreement: total-variation distance (|Δp| for two classes); for two classes also the correlation of
    the logits, for more the share of queries with the same most probable class."""
    pa, pb = np.atleast_2d(pa.T).T, np.atleast_2d(pb.T).T
    out = {"mean_tv": float(0.5 * np.abs(pa - pb).sum(1).mean()) if pa.shape[1] > 1 else float(np.abs(pa - pb).mean())}
    if pa.shape[1] == 1:
        out["r_logit"] = float(np.corrcoef(logit(pa[:, 0]), logit(pb[:, 0]))[0, 1])
    else:
        out["same_argmax"] = float((pa.argmax(1) == pb.argmax(1)).mean())
    return out


def pair_table(names: list[str], cba: CaseBaseAlignment, huh: Huh, preds: dict) -> dict:
    out = {}
    for a, b in combinations(names, 2):
        pq = cba.per_query(a, b)
        r = {f"k{k}_{ident}": {kk: v for kk, v in cba.summary(ov).items() if kk != "boot"} for (k, ident), ov in pq.items()}
        r["huh"] = huh.pair(a, b)
        r["pred"] = agreement(preds[a], preds[b])
        out[f"{a}|{b}"] = r
    return out


def diff_ci(cba: CaseBaseAlignment, p1: tuple[str, str], p2: tuple[str, str], key=(50, "row")) -> dict:
    """Paired difference of two pairs' case-base alignment (p1 − p2), on the same resamples."""
    a = cba.summary(cba.per_query(*p1)[key])
    b = cba.summary(cba.per_query(*p2)[key])
    return {"est": a["est"] - b["est"], "ci": ci(a["boot"] - b["boot"])}


# --------------------------------------------------------------------------- H: hypo
def hypo_queries(st: Stage, X: str, te: np.ndarray) -> np.ndarray:
    """One landmark per test patient, at random (positions in the site's landmarks, sorted)."""
    s = st.S.lm(X).ft.subjects[te].astype(str)
    o = np.random.default_rng([st.seed, 43, st.ci(X)]).permutation(te.size)
    _, first = np.unique(s[o], return_index=True)
    return np.sort(te[o[first]])


def size_draws(st: Stage, X: str, tr: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
    """(name, rows of the site, seed) of the local metrics learned on n training patients, two disjoint draws per n."""
    subj = st.S.lm(X).ft.subjects[tr].astype(str)
    pats = np.unique(subj)
    out = []
    for i, n in enumerate(SIZES):
        m = pats.size // 2 if n == "half" else int(n)
        perm = np.random.default_rng([st.seed, 47, st.ci(X), m]).permutation(pats)
        for d, part in enumerate((perm[:m], perm[m:2 * m])):
            out.append((f"n{n}_{'ab'[d]}", tr[np.isin(subj, part)], st.seed + 1000 + 10 * i + d))
    return out


def hypo_metric(st: Stage, X: str, name: str, rows: np.ndarray, seed: int):
    """A local metric learned on the site's rows `rows` with the external stage's learning sample (cached)."""
    path = cache_dir() / "metrics" / f"hypo_{X}_{name}.pkl"
    if path.exists():
        return load_pickle(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pool = st.S.lm(X).take(rows)
    learn = learning_sample(pool, int(st.mcfg["learn"]["per_patient"]), int(st.mcfg["learn"]["per_site"]), seed)
    m = fit_metric(pool.take(learn), st.mcfg, st.space, seed=seed, threads=st.threads)
    m.info["n_patients_learn"] = int(np.unique(pool.ft.subjects[learn].astype(str)).size)
    dump(m, path)
    return m


def hypo_unit(st: Stage, X: str, name: str, metric, tr: np.ndarray, te: np.ndarray, q: np.ndarray,
              full_test: bool) -> dict:
    """Neighbours and predictions of the queries under one metric (and, with `full_test`, its predictions on every
    test landmark), cached."""
    path = cache_dir() / "units" / f"hypo_{X}_{name}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    path.parent.mkdir(parents=True, exist_ok=True)
    lm = st.S.lm(X)
    E = metric.enc.transform(lm.ft)
    cb, a = st.case_base(metric, lm, tr, E, st.seed + 100 * FOLD)
    assert cb.k == KS[0], "the method's k changed"
    _, I, valid = cb.search(E[q], lm.ft.subjects[q])
    u = {"I": I, "valid": valid, "p_q": cb.probs(I, valid, cb.k)[:, 1], "Ewq": (E[q] * cb.sw).astype(np.float32),
         "alpha": np.array(a["alpha"])}
    if full_test:
        u["p_te"] = cb.predict(lm.take(te), E=E[te])["p"]
    np.savez(path, **u)
    return u


def near_ties(st: Stage, X: str, metric, tr: np.ndarray, q: np.ndarray, dp: np.ndarray, alpha: float) -> dict:
    """Where a refit's predictions differ from the stage's: is each difference one vote, at a query whose k-th and
    (k+1)-th neighbours are tied to float32 precision? (BLAS rounds the distances differently with the batch size;
    DECISIONS 2026-11-04.)"""
    lm = st.S.lm(X)
    E = metric.enc.transform(lm.ft)
    cb, _ = st.case_base(metric, lm, tr, E, st.seed + 100 * FOLD)
    qd = q[dp > 1e-9]
    D, _, valid = cb.search(E[qd], lm.ft.subjects[qd], k=cb.k + 1)
    gap = np.array([(d[v][cb.k] - d[v][cb.k - 1]) / d[v][cb.k - 1] for d, v in zip(D, valid)])
    one_vote = 1 / (cb.k + 2 * alpha) + 1e-9
    return {"n_queries_differ": int(qd.size), "max_rel_gap_k": float(gap.max()),
            "ties_only": bool(dp.max() <= one_vote and gap.max() < 1e-5)}


def hypo_site(st: Stage, X: str) -> dict:
    import posthoc_hypo_ablation as AB

    out = out_dir() / f"alignment_hypo_{X}.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("freeze_key") == _FREEZE["key"]:            # v0.6: never a result of another frozen state
            return old
        LOG.info("H %s: %s belongs to another freeze state; recomputed", X, out.name)
    t = time.perf_counter()
    lm = st.S.lm(X)
    tr, te = st.S.split(X, FOLD)
    q = hypo_queries(st, X, te)
    W = AB.weights_part(st, X)                      # cached by the ablation (it raises nothing new)
    stage = {"transported": pl.read_parquet(external_dir() / f"transported_{X}_f{FOLD}.parquet"),
             "local": pl.read_parquet(external_dir() / f"local_{X}_f{FOLD}.parquet")}
    metrics: dict = {"transported": (st.metric(X), None)}
    seed_local = st.seed + 100 * FOLD
    metrics["local"] = (hypo_metric(st, X, "local", tr, seed_local), None)
    geo: dict = {}
    for v in AB.VARIANTS:
        if v != AB.FROZEN:
            metrics[AB.vname(v)] = (AB.variant_metric(v, st, X, W, geo), v)
    sizes = size_draws(st, X, tr)
    for name, rows, seed in sizes:
        LOG.info("H %s: metric %s (%d patients)", X, name, np.unique(lm.ft.subjects[rows].astype(str)).size)
        metrics[name] = (hypo_metric(st, X, name, rows, seed), None)
    units, info = {}, {}
    for name, (m, v) in metrics.items():
        units[name] = hypo_unit(st, X, name, m, tr, te, q, full_test=name.startswith("n"))
        u = units[name]
        if name in stage:
            ref = _on(stage[name], q, "p")
        elif v is not None:
            ref = _on(pl.read_parquet(AB.out_dir() / "units" / f"{AB.vname(v)}_{X}_f{FOLD}.parquet"), q, "p")
        else:
            ref = None
        # competence on every test landmark: the stage's and the ablation's own predictions where they exist
        if name in stage:
            p_te = _on(stage[name], te, "p")
        elif v is not None:
            p_te = _on(pl.read_parquet(AB.out_dir() / "units" / f"{AB.vname(v)}_{X}_f{FOLD}.parquet"), te, "p")
        else:
            p_te = u["p_te"]
        y = lm.y[te]
        info[name] = {"reproduces": reproduction(ref, u["p_q"]),
                      "alpha": float(u["alpha"]), "logloss": float(logloss_rows(p_te, y).mean()),
                      "auroc": auc_weighted(y, np.ones(y.size), auc_prep(p_te)),
                      "n_learn": m.info.get("n_learn"), "n_patients_learn": m.info.get("n_patients_learn")}
        if ref is not None and np.isfinite(info[name]["reproduces"]) and info[name]["reproduces"] > 1e-9:
            info[name].update(near_ties(st, X, m, tr, q, np.abs(ref - u["p_q"]), float(u["alpha"])))
        LOG.info("H %s: %s done (reproduces %s)", X, name, info[name]["reproduces"])
    bad = {n: r["reproduces"] for n, r in info.items()
           if not reproduces(r["reproduces"]) and not (np.isfinite(r["reproduces"]) and r.get("ties_only"))}
    if bad:
        raise RuntimeError(f"alignment: these metrics do not reproduce their predictions on {X}: {bad}")
    _, pcode = np.unique(lm.ft.subjects[tr].astype(str), return_inverse=True)
    cba = CaseBaseAlignment(lm.ft.subjects[q], pcode, st.seed + 59 * st.ci(X))
    huh = Huh(q.size, st.seed + 61 * st.ci(X))
    names = list(metrics)
    for n in names:
        cba.add(n, units[n]["I"], units[n]["valid"])
        huh.nn(n, units[n]["Ewq"])
    pairs = pair_table(names, cba, huh, {n: units[n]["p_q"] for n in names})
    n_pat_cb = int(np.unique(pcode).size)
    variants = ["transported"] + [AB.vname(v) for v in AB.VARIANTS if v != AB.FROZEN]
    diffs = {"transported|local − nhalf_a|nhalf_b": diff_ci(cba, ("transported", "local"), ("nhalf_a", "nhalf_b")),
             "transported|local − zscore+uniform|local": diff_ci(cba, ("transported", "local"),
                                                                ("zscore+uniform", "local"))}
    res = {"site": X, "fold": FOLD, "freeze_key": _FREEZE["key"], "n_queries": int(q.size), "n_case_base_rows": int(tr.size),
           "n_case_base_patients": n_pat_cb, "chance": {f"k{k}": k / n_pat_cb for k in KS},
           "metrics": info, "sizes": [{"name": n, "n_patients": int(np.unique(lm.ft.subjects[r].astype(str)).size)}
                                       for n, r, _ in sizes],
           "variants": variants, "pairs": pairs, "differences": diffs,
           "architecture": architecture(pairs, info, variants), "seconds": time.perf_counter() - t}
    _write(out, res)
    return res


def _pair(pairs: dict, a: str, b: str) -> dict:
    return pairs.get(f"{a}|{b}") or pairs[f"{b}|{a}"]


def architecture(pairs: dict, info: dict, variants: list[str]) -> dict:
    """Huh et al.'s competence–alignment relation over the variants: each one's mean alignment with the others
    against its log-loss (Spearman; negative = the more competent align more)."""
    mean_al = {v: float(np.mean([_pair(pairs, v, o)["k50_row"]["est"] for o in variants if o != v])) for v in variants}
    mean_huh = {v: float(np.mean([_pair(pairs, v, o)["huh"]["est"] for o in variants if o != v])) for v in variants}
    ll = np.array([info[v]["logloss"] for v in variants])

    def rho(x):
        rx, ry = np.argsort(np.argsort(ll)), np.argsort(np.argsort(np.array([x[v] for v in variants])))
        return float(np.corrcoef(rx, ry)[0, 1])
    return {"mean_alignment_k50_row": mean_al, "mean_alignment_huh": mean_huh,
            "spearman_logloss_vs_alignment_k50_row": rho(mean_al), "spearman_logloss_vs_alignment_huh": rho(mean_huh)}


def hypo_prevalence(st: Stage, X: str) -> float:
    """Log-loss of the case base's prevalence on the fold's test landmarks: the baseline of a metric's competence."""
    lm = st.S.lm(X)
    tr, te = st.S.split(X, FOLD)
    return float(logloss_rows(np.full(te.size, float(lm.y[tr].mean())), lm.y[te]).mean())


# --------------------------------------------------------------------------- T: T2
def t2_args(pipe) -> dict:
    cfg = pipe.cfg
    return {"names": pipe.feature_names(), "dev": cfg.experiment["cohorts"]["development"],
            "externals": [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()],
            "d7": pipe.d7(), "threads": pipe.threads()}


def ids_sha(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(ids).astype(str)).encode()).hexdigest()[:16]


def t2_size_draws(subjects: np.ndarray, pool: np.ndarray, seed: int, si: int) -> list[tuple[str, np.ndarray, int]]:
    """As size_draws, for T2: (name, pool stays, seed) of the local metrics learned on n pool patients, two disjoint
    draws per n (`subjects` are the pool stays' patients)."""
    subj = np.asarray(subjects).astype(str)
    pats = np.unique(subj)
    out = []
    for i, n in enumerate(DRY_SIZES if DRY else SIZES):
        m = pats.size // 2 if n == "half" else int(n)
        perm = np.random.default_rng([seed, 47, si, m]).permutation(pats)
        for d, part in enumerate((perm[:m], perm[m:2 * m])):
            out.append((f"n{n}_{'ab'[d]}", pool[np.isin(subj, part)], seed + 1000 + 10 * i + d))
    return out


def _t2_fit_job(job: dict) -> dict:
    """One T2 draw's local metric, in its own process: the transport stage's A(iii) fit on the draw's stays."""
    from perisim.config import load_config
    from perisim.pipeline import Pipeline
    from perisim.transport import analyses as A
    from perisim.utils import set_threads

    set_threads(int(job["threads"]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    t = time.perf_counter()
    pipe = Pipeline(load_config(None, "local", {}))
    args = t2_args(pipe)
    args["threads"] = int(job["threads"])
    ctx = A.context(pipe.cfg, pipe.ws, "t2", job["site"], args)
    rows = np.asarray(job["rows"])
    if ids_sha(ctx["X"].ft.ids[rows]) != job["ids_sha"]:
        raise RuntimeError(f"alignment: the stays of T2 {job['site']} {job['name']} changed")
    fp = A._fit(pipe.cfg, ctx["X"].ft.take(rows), ctx["lab"]["own"][rows], ctx["C"], int(job["seed"]), args,
                Path(job["ckpt"]))
    s = time.perf_counter() - t
    A._dump({"enc": fp.enc, "w": fp.w, "lambda": fp.info["lambda"], "n_accepted": fp.loop["n_accepted"],
             "stop_reason": fp.loop["stop_reason"], "n_rows": int(rows.size), "ids_sha": job["ids_sha"],
             "seed": int(job["seed"]), "threads": int(job["threads"]), "seconds": s}, Path(job["out"]))
    return {"seconds": s}


def t2_fit_draws(pipe, args: dict, sites: list[str], workers: int) -> tuple[dict, dict]:
    """The draws of every site, and the fits still missing, the largest first, `workers` processes side by side with
    the transport stage's threads each (cached; a failed fit is reported, not replaced)."""
    import concurrent.futures as cf
    import multiprocessing as mp

    from perisim.pipeline import _child_threads
    from perisim.transport.analyses import context

    draws, jobs = {}, []
    for s in sites:
        ctx = context(pipe.cfg, pipe.ws, "t2", s, args)
        X, y, sp = ctx["X"], ctx["lab"]["own"], ctx["split"]
        pool = sp["pool"][y[sp["pool"]] >= 0]
        draws[s] = []
        for name, rows, seed in t2_size_draws(X.ft.subjects[pool], pool, int(pipe.cfg.seed), int(s == "eicu")):
            d = {"name": name, "rows": rows, "seed": seed, "ids_sha": ids_sha(X.ft.ids[rows]),
                 "path": cache_dir() / "metrics" / f"t2_{s}_{name}.pkl",
                 "n_patients": int(np.unique(X.ft.subjects[rows].astype(str)).size), "n_rows": int(rows.size)}
            draws[s].append(d)
            if not d["path"].exists():
                jobs.append({"site": s, "name": name, "rows": rows.tolist(), "seed": seed, "ids_sha": d["ids_sha"],
                             "threads": int(args["threads"]), "out": str(d["path"]),
                             "ckpt": str(cache_dir() / "ckpt" / f"t2_{s}_{name}")})
    jobs.sort(key=lambda j: -len(j["rows"]))
    failed: dict = {}
    if not jobs:
        return draws, failed
    par = min(workers, len(jobs))
    LOG.info("N: %d T2 draw fit(s) in %d process(es) × %d threads", len(jobs), par, int(args["threads"]))
    (cache_dir() / "metrics").mkdir(parents=True, exist_ok=True)
    with _child_threads(int(args["threads"])), \
            cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_t2_fit_job, j): j for j in jobs}
        for f in cf.as_completed(futs):
            j = futs[f]
            try:
                LOG.info("N %s: %s fitted (%d stays, %.0f s)", j["site"], j["name"], len(j["rows"]),
                         f.result()["seconds"])
            except Exception as e:                       # noqa: BLE001 — reported, not replaced
                failed[f"{j['site']}:{j['name']}"] = repr(e)
                LOG.error("N %s: %s failed: %r", j["site"], j["name"], e)
    return draws, failed


def t2_pairs_reproduce(new: dict, old: dict) -> list[str]:
    """The pairs of the 2026-11-04 run that N does not reproduce exactly (the same queries, draws and resamples)."""
    bad = []
    for key, r in old.items():
        n = new.get(key)
        if n is None or any(abs(n[m]["est"] - r[m]["est"]) > 1e-12 for m in r if m != "pred"):
            bad.append(key)
    return bad
def t2_part(sites: list[str], draws: bool = False, workers: int = WORKERS) -> dict:
    """T (the 2026-11-04 metrics), or with `draws` N: the same metrics, T2's draws and the Euclidean floor."""
    import posthoc_drift as D
    import posthoc_drift_missing as DM
    import posthoc_transport as PT
    from perisim.config import load_config
    from perisim.knn.predict import KNNModel
    from perisim.pipeline import Pipeline
    from perisim.transport.analyses import context
    from perisim.transport.core import check_hashes, frozen_hashes

    pipe = Pipeline(load_config(None, "local", {}))
    cfg, ws, t = pipe.cfg, pipe.ws, "t2"
    # v0.6: the live code and configuration must be T2's frozen ones (not only unchanged during the run)
    t2_key = freeze_key({k: v for k, v in _frozen.check_t2_freeze(cfg, ws).items() if k != "src"})
    before = frozen_hashes(ws, [t])
    args = t2_args(pipe)
    fits, failed = t2_fit_draws(pipe, args, sites, workers) if draws else ({}, {})
    kcfg = cfg.experiment["knn"]
    fm = PT._load(ws.predictions("external", t) / "final_model.pkl")
    enc, w = fm["enc"], np.asarray(fm["w"], float)
    k = int(cfg.experiment["knn"]["k_primary"])
    assert k == KS[0], "T2's k is not the method's k of the plan"
    alpha = float(dict(fm["alpha"]).get(k, 1.0))
    look = D.frozen_points(enc, w)
    res = {}
    for s in sites:
        out = out_dir() / f"alignment_t2{'n' if draws else ''}_{s}.json"
        if out.exists():
            old = json.loads(out.read_text(encoding="utf-8"))
            if old.get("freeze_key") == t2_key:                  # v0.6: never a result of another frozen state
                res[s] = old
                continue
        t0 = time.perf_counter()
        ctx = context(cfg, ws, t, s, args)
        X, y, sp = ctx["X"], ctx["lab"]["own"], ctx["split"]
        pool, q = sp["pool"][y[sp["pool"]] >= 0], sp["test"][y[sp["test"]] >= 0]
        C = int(1 + max(y[pool].max(), y[q].max()))
        codes = D.value_codes(enc, X.ft, look)
        dfit = D._load(ws.predictions("posthoc", t) / f"drift_{s}.pkl")
        mfit = D._load(ws.predictions("posthoc", t) / f"drift_missing_{s}.pkl")
        if dfit.get("key") != D._cache_key(ws, t) or mfit.get("key") != DM._cache_key(ws, t):
            raise RuntimeError(f"alignment: the cached drift fits of {s} belong to another frozen model")
        loc = PT._load(ws.predictions("transport", t) / f"{s}_local_model.pkl")
        kL = loc["knn"]
        if not (np.array_equal(np.asarray(kL.ids).astype(str), np.asarray(X.ft.ids)[pool].astype(str))
                and np.array_equal(np.asarray(kL.y), y[pool])):
            raise RuntimeError(f"alignment: the local model's case base is not the pool of {s} in order")
        E_fz = enc.transform(X.ft) * enc.weights_dims(w)
        emb = {"frozen": (E_fz[pool], E_fz[q], y[pool], alpha),
               "frozen_lookup": (D.embed([lj["points"] for lj in look], codes[pool]),
                                 D.embed([lj["points"] for lj in look], codes[q]), y[pool], alpha),
               "local": (kL.Ew, loc["enc"].transform(X.ft.take(q)) * kL.sw, kL.y, float(kL.alpha[k])),
               "joint": (D.embed(dfit["joint"]["points"], codes[pool]), D.embed(dfit["joint"]["points"], codes[q]),
                         y[pool], alpha),
               "missing": (D.embed(mfit["fit"]["points"], codes[pool]), D.embed(mfit["fit"]["points"], codes[q]),
                           y[pool], alpha)}
        sizes = []
        if draws:
            ft_p, ft_q = X.ft.take(pool), X.ft.take(q)
            for d in fits[s]:
                if not d["path"].exists():              # failed: reported, not replaced
                    continue
                f = PT._load(d["path"])
                if f["ids_sha"] != d["ids_sha"]:
                    raise RuntimeError(f"alignment: the cached fit of T2 {s} {d['name']} is of other stays")
                sw = f["enc"].weights_dims(f["w"])
                kn = KNNModel(f["enc"].transform(ft_p), sw, y[pool], C, X.ft.subjects[pool], [k])
                kn.fit_alpha_and_ood(kcfg["alpha_grid"], k, kcfg["ood_percentile"])   # α by LOO on the pool
                emb[d["name"]] = (kn.Ew, f["enc"].transform(ft_q) * sw, y[pool], float(kn.alpha[k]))
                sizes.append({"name": d["name"], "n_patients": d["n_patients"], "n_rows": d["n_rows"],
                              **{kk: f[kk] for kk in ("lambda", "n_accepted", "stop_reason", "seconds")}})
            e_dims = enc.neutral_weights_dims(np.ones(enc.n_features))
            emb["euclidean"] = (enc.transform_neutral(ft_p) * e_dims, enc.transform_neutral(ft_q) * e_dims, y[pool],
                                alpha)                  # the frozen α (DECISIONS 2026-11-05)
        z = np.load(ws.predictions("posthoc", t) / f"{s}_preds.npz")
        same_rows = bool(np.array_equal(z["y_test"], y[q]) and np.array_equal(z["s_test"], X.ft.subjects[q].astype(str)))
        stage = {"frozen": z["perisim__frozenlocal__test"], "local": z["perisim__local__test"]} if same_rows else {}
        subj_q = X.ft.subjects[q].astype(str)
        o = np.random.default_rng([int(cfg.seed), 43, int(s == "eicu")]).permutation(q.size)
        _, first = np.unique(subj_q[o], return_index=True)
        one = np.sort(o[first])                         # one test stay per patient (Huh et al.'s items)
        _, pcode = np.unique(X.ft.subjects[pool].astype(str), return_inverse=True)
        cba = CaseBaseAlignment(subj_q, pcode, int(cfg.seed) + 59 * (2 + (s == "eicu")))
        huh = Huh(one.size, int(cfg.seed) + 61 * (2 + (s == "eicu")))
        preds, info = {}, {}
        for name, (Er, Eq, yr, a) in emb.items():
            Er, Eq = np.ascontiguousarray(Er, np.float32), np.ascontiguousarray(Eq, np.float32)
            _, I = search(Er, Eq, k)
            L = yr[I]
            n = np.stack([(L == c).sum(1) for c in range(C)], 1).astype(float)
            prior = np.bincount(yr, minlength=C) / len(yr)
            p = (n + a * prior * C) / (n.sum(1, keepdims=True) + a * C)
            preds[name] = p
            cba.add(name, I, np.ones_like(I, dtype=bool))
            huh.nn(name, Eq[one])
            info[name] = {"alpha": a, "logloss": float(-np.log(np.clip(p[np.arange(q.size), y[q]], 1e-15, 1)).mean()),
                          "reproduces": reproduction(stage[name], p) if name in stage else None}
            LOG.info("T %s: %s done (reproduces %s)", s, name, info[name]["reproduces"])
        bad = {nm: r["reproduces"] for nm, r in info.items() if not reproduces(r["reproduces"])}
        missing = [nm for nm in ("frozen", "local") if nm not in stage]     # v0.6: a missing row fails the gate
        if bad or missing or not same_rows:
            raise RuntimeError(f"alignment: T2 {s} does not reproduce the stage's predictions: {bad or 'other rows'}")
        names = list(emb)
        pairs = pair_table(names, cba, huh, preds)
        diffs = {"joint|local − frozen|local": diff_ci(cba, ("joint", "local"), ("frozen", "local")),
                 "missing|local − frozen|local": diff_ci(cba, ("missing", "local"), ("frozen", "local"))}
        r = {"site": s, "freeze_key": t2_key, "n_queries": int(q.size), "n_queries_patients": int(one.size), "n_case_base_rows": int(pool.size),
             "n_case_base_patients": int(np.unique(pcode).size), "chance": {f"k{kk}": kk / pool.size for kk in KS},
             "neutral_axes_quantised": [lj["name"] for lj in look if not lj["exact"]],
             "metrics": info, "pairs": pairs, "differences": diffs, "seconds": time.perf_counter() - t0}
        if draws:
            old = json.loads((base_dir() / f"alignment_t2_{s}.json").read_text(encoding="utf-8"))["pairs"]
            if bad_pairs := t2_pairs_reproduce(pairs, old):
                raise RuntimeError(f"alignment: N does not reproduce the 2026-11-04 pairs of T2 {s}: {bad_pairs}")
            top = [d["name"][:-2] for d in sizes if d["name"].endswith("_a") and d["name"][:-2] + "_b" in emb]
            if top:
                diffs[f"frozen|local − {top[-1]}_a|{top[-1]}_b"] = diff_ci(cba, ("frozen", "local"),
                                                                         (f"{top[-1]}_a", f"{top[-1]}_b"))
            diffs["frozen|local − euclidean|local"] = diff_ci(cba, ("frozen", "local"), ("euclidean", "local"))
            prior = np.bincount(y[pool], minlength=C) / pool.size
            r.update(sizes=sizes, failed={kk.split(":", 1)[1]: v for kk, v in failed.items() if kk.startswith(s + ":")},
                     ll_prevalence=float(-np.log(prior[y[q]]).mean()), reproduces_2026_11_04=len(old),
                     seconds=time.perf_counter() - t0)
        _write(out, r)
        res[s] = r
    check_hashes(before, frozen_hashes(ws, [t]))
    return res


# --------------------------------------------------------------------------- report
def _a(r: dict, d: int = 2) -> str:
    return f"{r['est']:.{d}f} ({r['ci'][0]:.{d}f}–{r['ci'][1]:.{d}f})" if r.get("ci") else f"{r['est']:.{d}f}"


def _h(r: dict) -> str:
    return f"{r['est']:.2f} (null {r['null_mean']:.3f}; calibrated {r['calibrated']:.2f})"


def families(r: dict) -> dict:
    """Mean within-sample alignment of the ablation's variant pairs that share their weights, their geometry or
    neither (the transported variant left out)."""
    vs = [v for v in r["variants"] if v != "transported"]
    out: dict = {"same weights": [], "same geometry": [], "neither": []}
    for a, b in combinations(vs, 2):
        (ga, wa), (gb, wb) = a.split("+"), b.split("+")
        out["same weights" if wa == wb else "same geometry" if ga == gb else "neither"].append(
            _pair(r["pairs"], a, b)["huh"]["est"])
    return {k: {"mean": float(np.mean(v)), "n_pairs": len(v)} for k, v in out.items()}


INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _ns(r: dict) -> list[str]:
    """The n of a result's draws whose two halves exist, in order ("500", …, "half")."""
    names = {d["name"] for d in r["sizes"]}
    return [d["name"][1:-2] for d in r["sizes"] if d["name"].endswith("_a") and d["name"][:-2] + "_b" in names]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID_C, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def figure(H: dict, path: Path, N: dict | None = None) -> str:
    """The n axis, per `hypo` site (and, with N, per T2 site): within-sample alignment of two disjoint draws and of a
    draw with the transported metric, against the patients per draw; the transported–local and Euclidean–local pairs
    as references."""
    plt = _plt()
    rows = [[(f"hypo · {X}" if N else X, r, "transported", "zscore+uniform") for X, r in H.items()]]
    if N:
        rows.append([(f"T2 · {s}", r, "frozen", "euclidean") for s, r in N.items()])
    ncol = max(len(x) for x in rows)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(4.2 * ncol, 3.4 * len(rows)), facecolor=SURFACE, sharey=True,
                            squeeze=False)
    for i, row in enumerate(rows):
        for j in range(len(row), ncol):
            axs[i][j].set_visible(False)
        for ax, (title, r, trn, eu) in zip(axs[i], row):
            p, npat, ns = r["pairs"], {s["name"]: s["n_patients"] for s in r["sizes"]}, _ns(r)
            x = [npat[f"n{n}_a"] for n in ns]
            for other, col, lab in (("b", BLUE, "two disjoint draws"), (None, ORANGE, "a draw – transported")):
                y = [_pair(p, f"n{n}_a", f"n{n}_{other}" if other else trn)["huh"]["est"] for n in ns]
                ax.plot(x, y, color=col, lw=2, marker="o", ms=5, label=lab)
            ax.axhline(_pair(p, trn, "local")["huh"]["est"], color=INK, lw=1.5, label="transported – local")
            ax.axhline(_pair(p, eu, "local")["huh"]["est"], color=INK2, lw=1.2, ls="--", label="Euclidean – local")
            ax.set_xscale("log")
            ax.set_xticks(x, [f"{v:,}" if k + 1 == len(x) or x[k + 1] / v > 1.25 else "" for k, v in enumerate(x)],
                          fontsize=8)                 # "half" can sit next to 4,000
            ax.minorticks_off()
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("patients per draw")
            ax.set_ylim(0, 0.6)
            _style(ax)
        axs[i][0].set_ylabel(f"neighbours shared (k = {K_HUH}, {N_HUH:,} queries)")
    axs[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("neighbour alignment of local metrics learned on n patients (hypo fold 0, T2's transport split; "
                 "exploratory; permutation null ≈ 0.01)" if N else
                 "hypo: neighbour alignment of local metrics learned on n patients (fold 0, exploratory; "
                 "permutation null ≈ 0.01)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def competence_rows(H: dict, N: dict, P: dict) -> list[dict]:
    """Per experiment, site and n: the patients and learning rows of the draws, their competence (1 − log-loss / the
    log-loss of the case base's prevalence) and the alignment of the two draws."""
    out = []
    for exp, R in (("hypo", H), ("T2", N)):
        for site, r in R.items():
            llp = P.get(site) if exp == "hypo" else r.get("ll_prevalence")
            if llp is None:
                continue
            sz = {d["name"]: d for d in r["sizes"]}
            for n in _ns(r):
                ab = (f"n{n}_a", f"n{n}_b")
                learn = [r["metrics"][x]["n_learn"] if exp == "hypo" else sz[x]["n_rows"] for x in ab]
                out.append({"experiment": exp, "site": site, "n": n, "patients": sz[ab[0]]["n_patients"],
                            "learning_rows": [None if v is None else int(v) for v in learn],
                            "competence": [1 - r["metrics"][x]["logloss"] / llp for x in ab],
                            "huh": _pair(r["pairs"], *ab)["huh"]["est"], "k50": _pair(r["pairs"], *ab)["k50_row"]["est"]})
    return out


def figure_competence(rows: list[dict], path: Path) -> str:
    """The two draws' within-sample alignment against their mean competence: one line per site, colour by
    experiment, marker by site."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 4.2), facecolor=SURFACE)
    marks = {"inspire": "o", "mimic": "s", "eicu": "^"}
    for exp, col in (("hypo", BLUE), ("T2", ORANGE)):
        for site in dict.fromkeys(r["site"] for r in rows if r["experiment"] == exp):
            pts = [r for r in rows if r["experiment"] == exp and r["site"] == site]
            ax.plot([float(np.mean(r["competence"])) for r in pts], [r["huh"] for r in pts], color=col, lw=1.5,
                    marker=marks.get(site, "o"), ms=6, label=f"{exp} · {site}")
    ax.set_xlabel("competence of the draws (1 − log-loss / log-loss of the prevalence)")
    ax.set_ylabel(f"two disjoint draws: neighbours shared (k = {K_HUH})")
    ax.set_ylim(0, 0.6)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("alignment against competence (exploratory; DECISIONS 2026-11-05)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def _pred(r: dict) -> str:
    q = r["pred"]
    return f"{q['mean_tv']:.3f} / " + (f"{q['r_logit']:.3f}" if "r_logit" in q else f"{q['same_argmax']:.3f}")


def render_n(H: dict, N: dict, P: dict) -> list[str]:
    """N: T2's draws (DECISIONS 2026-11-05), and the alignment against competence beside `hypo`'s."""
    L = ["", "## T2: local metrics learned on n patients (DECISIONS 2026-11-05)", "",
         "Each draw is fitted by the transport stage's A(iii) procedure on its patients' labelled pool stays; its α by "
         "patient-level leave-one-out on the pool, its case base. The Euclidean kNN (the neutral axes) uses the frozen "
         "α. The 2026-11-04 pairs were reproduced exactly ("
         + "; ".join(f"{s}: {r['reproduces_2026_11_04']} pairs" for s, r in N.items()) + ")."]
    fails = [f"{s} {n}: {e}" for s, r in N.items() for n, e in r["failed"].items()]
    if fails:
        L += ["", "**Failed fits (reported, not replaced):** " + "; ".join(fails)]
    ns = list(dict.fromkeys(n for r in N.values() for n in _ns(r)))
    L += ["", "| n | " + " | ".join(f"{s}: patients / stays / draws a–b / a–frozen / a–local" for s in N) + " |",
          "|---|" + "---|" * len(N)]
    for n in ns:
        cells = []
        for r in N.values():
            if n not in _ns(r):
                cells.append("—")
                continue
            sz, p = {d["name"]: d for d in r["sizes"]}[f"n{n}_a"], r["pairs"]
            cells.append(f"{sz['n_patients']:,} / {sz['n_rows']:,} / " + " / ".join(
                f"{_pair(p, a, b)['k50_row']['est']:.2f}"
                for a, b in ((f"n{n}_a", f"n{n}_b"), (f"n{n}_a", "frozen"), (f"n{n}_a", "local"))))
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    top = ns[-1] if ns else None
    key = [("frozen", "local"), ("euclidean", "local"), ("euclidean", "frozen")]
    if top:
        key[1:1] = [(f"n{top}_a", f"n{top}_b")]
        key += [("frozen", f"n{top}_a"), ("local", f"n{top}_a")]
    L += ["", "| pair | " + " | ".join(N) + " |", "|---|" + "---|" * len(N)]
    for a, b in key:
        for lab, f in (("case base", lambda q: _a(q["k50_row"])), ("within the sample", lambda q: _h(q["huh"])),
                       ("prediction mean TV / same most probable class", _pred)):
            L.append(f"| {a} – {b}, {lab} | " + " | ".join(f(_pair(r["pairs"], a, b)) for r in N.values()) + " |")
    L += ["", "Paired differences (case base, k = 50):", ""]
    L += [f"- {s}: " + "; ".join(f"{k}: {_a(v, 3)}" for k, v in r["differences"].items()) for s, r in N.items()]
    rows = competence_rows(H, N, P)
    if rows:
        L += ["", "### Alignment against competence (within the sample)", "",
              "Competence: 1 − log-loss / the log-loss of the case base's prevalence, on the test rows (`hypo`: the "
              "fold's test landmarks; T2: the test stays). Learning rows: `hypo` learns on up to 3 landmarks per "
              "patient (and at most 30,000), T2 on every labelled stay of the drawn patients.", "",
              "![competence](posthoc_alignment_competence.png)", "",
              "| experiment | site | n | patients | learning rows a / b | competence a / b | two draws, within the "
              "sample | two draws, case base |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['experiment']} | {r['site']} | {r['n']} | {r['patients']:,} | "
              + " / ".join("—" if v is None else f"{v:,}" for v in r["learning_rows"]) + " | "
              + " / ".join(f"{c:.3f}" for c in r["competence"]) + f" | {r['huh']:.2f} | {r['k50']:.2f} |" for r in rows]
    return L


def render(H: dict, T: dict, N: dict | None = None, P: dict | None = None) -> str:
    L = ["# Post hoc: do metrics learned apart retrieve the same neighbours? (exploratory)", "",
         "Mutual k-nearest-neighbour alignment (Huh et al., 2024): the share of each query's k nearest neighbours that "
         "two metrics have in common, averaged over the queries. Plan: docs/DECISIONS.md 2026-11-04. In the case base: "
         "k = 50, a neighbour identified by its row, 95 % CI from 500 patient resamples. Within the sample (Huh et "
         f"al.'s recipe): {N_HUH} queries, k = {K_HUH}, {DRAWS_HUH} draws, with the mean of a permutation null and the "
         "calibrated score (alignment − null) / (1 − null).", ""]
    if H:
        L += ["## `hypo` (fold 0)", "", "| site | queries | case-base patients | chance, k = 50 |", "|---|---|---|---|"]
        L += [f"| {X} | {r['n_queries']:,} | {r['n_case_base_patients']:,} | {r['chance']['k50']:.4f} |"
              for X, r in H.items()]
        L += ["", "Reproduction: every metric with stage or ablation predictions reproduced them on the queries to "
              "10⁻⁹" + "".join(f"; {X}'s {n} but for {m['n_queries_differ']} query (one vote, a float32 tie at the "
                                 f"k-th neighbour, relative gap {m['max_rel_gap_k']:.1e})"
                                 for X, r in H.items() for n, m in r["metrics"].items() if m.get("ties_only")) + ".",
              "", "![n axis](posthoc_alignment_n.png)"]
        key = [("transported", "local"), ("nhalf_a", "nhalf_b"), ("zscore+uniform", "local"),
               ("zscore+uniform", "transported"), ("transported", "nhalf_a"), ("local", "nhalf_a")]
        L += ["", "| pair | " + " | ".join(H) + " |", "|---|" + "---|" * len(H)]
        for a, b in key:
            L.append(f"| {a} – {b}, case base | " + " | ".join(_a(_pair(r["pairs"], a, b)["k50_row"]) for r in H.values())
                     + " |")
            L.append(f"| {a} – {b}, within the sample | "
                     + " | ".join(_h(_pair(r["pairs"], a, b)["huh"]) for r in H.values()) + " |")
            L.append(f"| {a} – {b}, prediction mean \\|Δp\\| / r | "
                     + " | ".join(f"{_pair(r['pairs'], a, b)['pred']['mean_tv']:.3f} / "
                                  f"{_pair(r['pairs'], a, b)['pred']['r_logit']:.3f}" for r in H.values()) + " |")
        L += ["", "Paired differences (case base, k = 50):", ""]
        for X, r in H.items():
            L += [f"- {X}: " + "; ".join(f"{k}: {_a(v, 3)}" for k, v in r["differences"].items())]
        L += ["", "### n: two disjoint draws of n training patients", "",
              "| n | " + " | ".join(f"{X}: draws a–b / a–transported / a–local" for X in H) + " |",
              "|---|" + "---|" * len(H)]
        for n in SIZES:
            cells = []
            for r in H.values():
                p = r["pairs"]
                cells.append(" / ".join(f"{_pair(p, a, b)['k50_row']['est']:.2f}"
                                        for a, b in ((f"n{n}_a", f"n{n}_b"), (f"n{n}_a", "transported"),
                                                     (f"n{n}_a", "local"))))
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L += ["", "### Architecture: the ablation's variants", "",
              "| variant | " + " | ".join(f"{X}: log-loss / mean alignment" for X in H) + " |", "|---|" + "---|" * len(H)]
        for v in next(iter(H.values()))["variants"]:
            L.append(f"| {v} | " + " | ".join(f"{r['metrics'][v]['logloss']:.4f} / "
                                              f"{r['architecture']['mean_alignment_k50_row'][v]:.2f}"
                                              for r in H.values()) + " |")
        L.append("| Spearman (log-loss, mean alignment) | " + " | ".join(
            f"{r['architecture']['spearman_logloss_vs_alignment_k50_row']:.2f}" for r in H.values()) + " |")
        L.append("| Spearman, within the sample | " + " | ".join(
            f"{r['architecture']['spearman_logloss_vs_alignment_huh']:.2f}" for r in H.values()) + " |")
        L += ["", "Variant pairs (the transported one left out), mean within-sample alignment:", "",
              "| the two variants share | pairs | " + " | ".join(H) + " |", "|---|---|" + "---|" * len(H)]
        fam = {X: families(r) for X, r in H.items()}
        for f in ("same weights", "same geometry", "neither"):
            L.append(f"| {f.removeprefix('same ')} | "
                     f"{next(iter(fam.values()))[f]['n_pairs']} | "
                     + " | ".join(f"{fam[X][f]['mean']:.2f}" for X in H) + " |")
    if T:
        L += ["", "## T2", "", "| site | queries | case-base stays | chance, k = 50 |", "|---|---|---|---|"]
        L += [f"| {s} | {r['n_queries']:,} | {r['n_case_base_rows']:,} | {r['chance']['k50']:.4f} |"
              for s, r in T.items()]
        L += ["", "| pair | " + " | ".join(T) + " |", "|---|" + "---|" * len(T)]
        for a, b in [("frozen", "frozen_lookup"), ("frozen", "local"), ("joint", "local"), ("missing", "local"),
                     ("frozen", "joint"), ("frozen", "missing")]:
            L.append(f"| {a} – {b}, case base | " + " | ".join(_a(_pair(r["pairs"], a, b)["k50_row"]) for r in T.values())
                     + " |")
            L.append(f"| {a} – {b}, within the sample | "
                     + " | ".join(_h(_pair(r["pairs"], a, b)["huh"]) for r in T.values()) + " |")
        L += ["", "Paired differences (case base, k = 50):", ""]
        for s, r in T.items():
            L += [f"- {s}: " + "; ".join(f"{k}: {_a(v, 3)}" for k, v in r["differences"].items())]
    if H and T:
        L += ["", "## `hypo` and T2 side by side (within the sample: the same n and k in both)", "",
              "| experiment | site | transported – local | local – local, two halves | Euclidean – local | "
              "re-deformed – local |", "|---|---|---|---|---|---|"]
        L += [f"| hypo | {X} | {_h(_pair(r['pairs'], 'transported', 'local')['huh'])} | "
              f"{_h(_pair(r['pairs'], 'nhalf_a', 'nhalf_b')['huh'])} | "
              f"{_h(_pair(r['pairs'], 'zscore+uniform', 'local')['huh'])} | — |" for X, r in H.items()]
        L += [f"| T2 | {s} | {_h(_pair(r['pairs'], 'frozen', 'local')['huh'])} | {_n_ref(N, s, 'nhalf_a', 'nhalf_b')} | "
              f"{_n_ref(N, s, 'euclidean', 'local')} | {_h(_pair(r['pairs'], 'joint', 'local')['huh'])} |"
              for s, r in T.items()]
    if N:
        L += render_n(H, N, P or {})
    return "\n".join(L) + "\n"


def _n_ref(N: dict | None, s: str, a: str, b: str) -> str:
    """A pair of N's (T2's draws, the Euclidean floor) for the side-by-side table, or — before N has run."""
    try:
        return _h(_pair(N[s]["pairs"], a, b)["huh"])
    except (KeyError, TypeError):
        return "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="*", default=["H", "T", "N"], choices=["H", "T", "N"])
    ap.add_argument("--sites", nargs="*", default=None, help="hypo sites (T2 always runs MIMIC-IV and eICU)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--workers", type=int, default=WORKERS, help="N: T2 draw fits side by side")
    ap.add_argument("--report-only", action="store_true", help="re-render the report from posthoc_alignment.json")
    ap.add_argument("--dry-run", action="store_true", help="N's mechanics at DRY_SIZES, into posthoc_alignment/dryrun/")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    global DRY
    DRY = a.dry_run
    if a.report_only:
        R = json.loads((out_dir() / "posthoc_alignment.json").read_text(encoding="utf-8"))
        return report(R["hypo"], R["t2"], R.get("t2_n", {}), R.get("hypo_prevalence_logloss", {}))
    rec = check()
    _FREEZE["key"] = freeze_key(rec)
    out_dir().mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    H, T, N, P, st = {}, {}, {}, {}, None
    if DRY:                                          # H and T from the last run, never recomputed here
        R0 = json.loads((base_dir() / "posthoc_alignment.json").read_text(encoding="utf-8"))
        H, T, a.parts = R0["hypo"], R0["t2"], ["N"]
    if "H" in a.parts:
        st = Stage(threads=a.threads)
        H = {X: hypo_site(st, X) for X in (a.sites or list(st.cfg["cohorts"]))}
    if "T" in a.parts:
        T = t2_part(T2_SITES)
    if "N" in a.parts:
        N = t2_part(T2_SITES, draws=True, workers=a.workers)
        st = st or Stage(threads=a.threads)
        P = {X: hypo_prevalence(st, X) for X in H}
    check()
    _write(out_dir() / "posthoc_alignment.json", {"freeze_hypo": rec["tag"], "parts": a.parts, "hypo": H, "t2": T,
                                                  "t2_n": N, "hypo_prevalence_logloss": P, "dry_run": DRY,
                                                  "seconds": time.perf_counter() - t})
    return report(H, T, N, P)


def report(H: dict, T: dict, N: dict | None = None, P: dict | None = None) -> int:
    if H:
        figure(H, out_dir() / "posthoc_alignment_n.png", N)
    if H and N and P:
        figure_competence(competence_rows(H, N, P), out_dir() / "posthoc_alignment_competence.png")
    (out_dir() / "posthoc_alignment.md").write_text(render(H, T, N, P), encoding="utf-8")
    LOG.info("alignment report: %s", out_dir() / "posthoc_alignment.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
