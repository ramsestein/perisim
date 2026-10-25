"""Building blocks of the transport analyses: patient-level splits, labels, case bases with site weights, point
metrics (including the calibration of every cumulative threshold of ordinal targets), frozen-artefact hashes and the
pre-declared variable classes of H6."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from perisim.evaluation import metrics as M
from perisim.experiment import ORDINAL, SEVERE, labels_for
from perisim.external import absolute_labels, patient_halves
from perisim.features.table import FeatureTable
from perisim.knn.predict import KNNModel

SEED_OFFSET = {"inspire": 0, "mimic": 1, "eicu": 2}
ROLES = ("test", "tuning", "pool")


# --------------------------------------------------------------------------- splits
def patient_split(subjects: np.ndarray, test_fraction: float, tuning_max: int, seed: int) -> dict[str, np.ndarray]:
    """Disjoint patient-level sets, as stay indices: `test` (a fixed fraction of the patients), `tuning` (at most
    `tuning_max` patients, never more than half of the rest) and `pool` (the rest: every case base is drawn from it)."""
    subj = np.asarray(subjects).astype(str)
    u = np.unique(subj)
    perm = np.random.default_rng(seed).permutation(u.size)
    n_test = int(round(test_fraction * u.size))
    n_tune = int(min(tuning_max, max(u.size - n_test, 0) // 2))
    role = {s: ("test" if i < n_test else "tuning" if i < n_test + n_tune else "pool") for i, s in enumerate(u[perm])}
    r = np.array([role[s] for s in subj])
    return {k: np.where(r == k)[0] for k in ROLES}


def assert_disjoint(split: dict[str, np.ndarray], subjects: np.ndarray) -> None:
    """No patient in two of the sets (rule 3 of the amendment)."""
    subj = np.asarray(subjects).astype(str)
    sets = {k: set(subj[v].tolist()) for k, v in split.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if sets[a] & sets[b]:
                raise AssertionError(f"transport: patients shared between {a} and {b}")


def patient_orders(subjects: np.ndarray, blocks: np.ndarray | None, reps: int, seed: int) -> dict[str, list[np.ndarray]]:
    """Orders of the patients for the accumulation curves: `random`, and `blocks` (time blocks in calendar order,
    random inside a block) when the cohort has time information; `reps` repetitions each."""
    subj = np.asarray(subjects).astype(str)
    u = np.unique(subj)
    rng = np.random.default_rng(seed)
    out = {"random": [rng.permutation(u) for _ in range(reps)]}
    if blocks is not None:
        first: dict[str, str] = {}
        for s, b in zip(subj, np.asarray(blocks).astype(str)):
            if s not in first or b < first[s]:
                first[s] = b
        pb = np.array([first[s] for s in u])
        out["blocks"] = [u[np.lexsort((rng.random(u.size), pb))] for _ in range(reps)]
    return out


def time_blocks(common_dir: Path, ids: np.ndarray) -> np.ndarray | None:
    """Calendar block of each stay: MIMIC-IV anchor-year group, eICU discharge year; None without time information
    (INSPIRE: times are relative to each patient)."""
    st = pl.read_parquet(Path(common_dir) / "stays_index.parquet")
    order = pl.DataFrame({"stay_id": np.asarray(ids).astype(str)})
    for col in ("year_group", "year"):
        if col not in st.columns:
            continue
        b = order.join(st.select("stay_id", pl.col(col).cast(pl.Utf8).alias("b")), on="stay_id", how="left")["b"]
        if b.drop_nulls().n_unique() >= 2:
            return b.fill_null("~").to_numpy()            # unknown block last
    return None


def concat_ft(a: FeatureTable, b: FeatureTable) -> FeatureTable:
    """Stays of two cohorts in one table (same harmonized features; identifiers are prefixed by cohort)."""
    return FeatureTable(np.concatenate([a.ids, b.ids]), np.concatenate([a.subjects, b.subjects]), a.specs,
                        {k: np.concatenate([a.num[k], b.num[k]]) for k in a.num},
                        {k: np.concatenate([a.cat[k], b.cat[k]]) for k in a.cat})


# --------------------------------------------------------------------------- labels
def target_labels(target: str, et: str, X, cfg) -> dict[str, np.ndarray]:
    """Labels of the cohort the queries come from, the same for every case base: T2 / T3 tertiles cross-fitted
    between two patient halves (each half labelled with the other half's cut-points, as in the external validation)
    and the absolute categories; T1 as it is, eICU reduced to severe / not severe."""
    if et == "t1_bin":
        return {"own": X.t1_severe.copy()}
    if target == "t1":
        return {"own": X.t1.copy()}
    idx = np.arange(len(X))
    half = patient_halves(X.ft.subjects, cfg.seed)
    V = {h: labels_for(target, X, idx[half == h], cfg.targets)[0] for h in (0, 1)}
    return {"own": np.where(half == 0, V[1], V[0]), "absolute": absolute_labels(target, X, cfg.targets)}


def source_labels(target: str, et: str, X, cfg) -> dict[str, np.ndarray]:
    """Labels of a non-local cohort entering a case base: its own cut-points, estimated on all its stays (INSPIRE:
    those of the final model); with the reduced eICU target it carries the reduced label (readmission or death)."""
    if et == "t1_bin":
        own = X.extra["t1_reduced"].copy()
    elif target in ("t1", "t3") and (X.t1 < 0).all():   # no full T1 in that cohort (eICU): it contributes no case
        own = np.full(len(X), -1, dtype=np.int64)
    else:
        own = labels_for(target, X, np.arange(len(X)), cfg.targets)[0]
    out = {"own": own}
    if target in ("t2", "t3"):
        out["absolute"] = absolute_labels(target, X, cfg.targets)
    return out


# --------------------------------------------------------------------------- case bases and predictions
def case_base(parts: list[tuple], w_dims: np.ndarray, C: int, kcfg: dict, k_grid: list[int],
              alpha: dict | None = None, with_ood: bool = True) -> KNNModel | None:
    """kNN over the concatenated parts [(embedding, labels, subjects, is_local)], labelled stays only. α: the frozen
    one when given, else fitted on the case base (patient-level LOO). The out-of-support threshold is always that
    of this case base (LOO), computed only when `with_ood`."""
    parts = [p for p in parts if p is not None and len(p[1])]
    if not parts:
        return None
    y = np.concatenate([np.asarray(p[1]) for p in parts])
    ok = y >= 0
    if ok.sum() < 3 or np.unique(y[ok]).size < 2:
        return None
    E = np.vstack([p[0] for p in parts])[ok]
    s = np.concatenate([np.asarray(p[2]).astype(str) for p in parts])[ok]
    loc = np.concatenate([np.full(len(p[1]), bool(p[3])) for p in parts])[ok]
    kn = KNNModel(E, w_dims, y[ok], C, s, k_grid)
    kn.is_local = loc
    if alpha is None or with_ood:
        kn.fit_alpha_and_ood(kcfg["alpha_grid"] if alpha is None else [1.0], int(kcfg["k_primary"]),
                             kcfg["ood_percentile"])
    if alpha is not None:
        kn.alpha = {k: float(alpha.get(k, alpha.get(str(k), 1.0))) for k in kn.k_grid}
    return kn


def gamma_schedule(n_local: int, n0: float) -> float:
    """Declared decreasing site weight: γ(n) = n₀ / (n₀ + n_local)."""
    return float(n0) / (float(n0) + float(n_local))


def predict_gammas(kn: KNNModel, E_q: np.ndarray, subj_q: np.ndarray, k: int,
                   gammas: dict[str, float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Probabilities for several site weights γ with ONE neighbour search (the distance does not depend on γ): each
    non-local neighbour's vote, and its counts in the smoothing prior, are multiplied by γ. Also the out-of-support
    flag of every query (NaN when the case base has no threshold)."""
    D, I, valid = kn.neighbors(E_q, subj_q)
    kk = min(k, kn.k_max)
    a = kn.alpha.get(k, 1.0)
    out = {}
    for name, g in gammas.items():
        kn.set_ref_weight(None if g == 1 else np.where(kn.is_local, 1.0, float(g)))
        out[name] = kn.probs(I, valid, kk, alpha=a)
    kn.set_ref_weight(None)
    if k in kn.ood_thr:
        ood = kn.kth_info(D, valid, k)[2].astype(float)
    else:
        ood = np.full(len(E_q), np.nan)
    return out, ood


# --------------------------------------------------------------------------- metrics
def cumulative_calibration(P: np.ndarray, y: np.ndarray) -> list[dict]:
    """Calibration of every cumulative threshold P(Y ≥ c), c = 1 … C−1 (for a binary target: the event)."""
    out = []
    for c in range(1, P.shape[1]):
        p, yb = P[:, c:].sum(1), (y >= c).astype(float)
        if 0 < yb.sum() < len(yb):
            a, b = M.calibration_line(p, yb)
            out.append({"threshold": c, "citl": M.calibration_in_the_large(p, yb), "intercept": a, "slope": b,
                        "ece": M.ece(p, yb)})
        else:
            out.append({"threshold": c, "citl": np.nan, "intercept": np.nan, "slope": np.nan, "ece": np.nan})
    return out


def macro_auroc(P: np.ndarray, y: np.ndarray) -> float:
    v = [M.auroc_binary(y == c, P[:, c]) for c in range(P.shape[1])]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def point_metrics(P: np.ndarray, y: np.ndarray, et: str, ood: np.ndarray | None = None) -> dict:
    """Metrics of one evaluation point (no bootstrap): log-loss, Brier, RPS, worst-class AUROC / AUPRC, macro AUROC,
    calibration of every cumulative threshold and the % out of support."""
    s = M.summary(P, y, ORDINAL[et], SEVERE[et])
    out = {"n": int(len(y)), "logloss": s["logloss"], "brier": s["brier"], "rps": s.get("rps", np.nan),
           "auroc_macro": macro_auroc(P, y), "auroc_worst": s.get("auroc_severe", np.nan),
           "auprc_worst": s.get("auprc_severe", np.nan)}
    for cc in cumulative_calibration(P, y):
        t = cc["threshold"]
        out.update({f"citl_ge{t}": cc["citl"], f"slope_ge{t}": cc["slope"], f"ece_ge{t}": cc["ece"]})
    out["ood_pct"] = float(np.nanmean(ood) * 100) if ood is not None and np.isfinite(ood).any() else np.nan
    return out


def boot_cumulative(P: np.ndarray, y: np.ndarray, subj: np.ndarray, c: int, which: str, n_boot: int, seed: int) -> dict:
    """Patient-bootstrap CI of the calibration in the large or the slope of the threshold P(Y ≥ c)."""
    def fn(p, yy):
        pg, yb = p[:, c:].sum(1), (yy >= c).astype(float)
        if not 0 < yb.sum() < len(yb):
            return float("nan")
        return M.calibration_in_the_large(pg, yb) if which == "citl" else M.calibration_line(pg, yb)[1]
    return M.bootstrap_metric(fn, P, y, subj, n_boot, seed)


def boot_auroc_difference(Pa: np.ndarray, Pb: np.ndarray, y: np.ndarray, subj: np.ndarray, n_boot: int,
                          seed: int) -> dict:
    """Paired patient-bootstrap CI of macro one-vs-rest AUROC(a) − AUROC(b) on the same stays."""
    C = Pa.shape[1]
    return M.bootstrap_metric(lambda p, yy: macro_auroc(p[:, :C], yy) - macro_auroc(p[:, C:], yy),
                              np.hstack([Pa, Pb]), y, subj, n_boot, seed)


# --------------------------------------------------------------------------- frozen artefacts and variable classes
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes(ws, targets: list[str]) -> dict[str, str]:
    """SHA-256 of the frozen artefacts: the freeze record, the frozen configuration and the final model(s)."""
    files = [ws.state / "freeze.json"] + sorted((ws.state / "frozen_config").glob("*")) + \
        [ws.predictions("external", t) / "final_model.pkl" for t in targets]
    return {Path(p).relative_to(ws.base).as_posix(): _sha256(Path(p)) for p in files if Path(p).is_file()}


def check_hashes(before: dict[str, str], after: dict[str, str]) -> None:
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if changed:
        raise RuntimeError(f"transport: frozen artefacts changed during the analysis: {changed}")


def variable_class(name: str) -> str:
    """Pre-declared classes of H6 (amendment): physiological, practice, coding, patient."""
    if name.startswith(("last_", "slope_", "preop_")):
        return "physiological"
    if name.startswith(("used_", "hours_since_", "med_")) or name in (
            "icu_los_h", "surgery_duration_min", "anesthesia_type", "admit_source", "emergency"):
        return "practice"
    if name.startswith(("dx_", "pcs_")) or name == "surgical_group":
        return "coding"
    if name in ("age", "sex_male", "bmi", "asa"):
        return "patient"
    return "other"
