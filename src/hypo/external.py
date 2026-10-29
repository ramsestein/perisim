"""Phase 4: the external stage (docs/prereg_hypo.md §5–§7), resumable, on the frozen version only.

Each rotation is named by its external site X; its transported metric and comparators are the frozen artefacts.
Per outer fold f of X's Phase 2 folds (case base = the training folds, test = the fold), units written once to
<workdir>/external/ (patient level, git-ignored) and skipped when present:
- transported_<X>_f<f>: PERISIM transported (frozen metric, X's case base): p, out of support, the three bands,
  y read with one reading per hour, the neighbours' carried outcomes and arterial-line agreement; in eICU the
  own-hospital share of the neighbours and H5's γ-weighted votes; the .json holds α and the explanation aggregates;
- local_<X>_f<f>: PERISIM local (a metric learned on X's training folds only, the same case base);
- comps_<X>_f<f>: the transported comparators, as they are and recalibrated on n local patients (the curve's draws);
- curve_<X>_f<f>: S1, case bases of n random training patients;
- hospitals_<X>_f<f>: H4 (sites with hospitals: eICU), own hospital against as many other patients, and the
  random-group control.
Fold 0 only: drift_ext-<X>_<site> (S4; the development sites as the control), drift-hospitals_ext-<X> (S6) and
shapes_<site> (S5), from `drift.py`. The freeze is checked at the start and at the end.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from .casebase import CaseBase
from .config import code_hash
from .cv import Sites, cv_dir, settings
from .data import Landmarks, learning_sample
from .frozen import Stop, check, external_dir, final_dir, load_pickle
from .metric import fit_metric
from .stats import label_shift, logloss_rows, recalibrate

LOG = logging.getLogger("hypo.external")
# smaller grids for the tests and a mechanics check on synthetic data (never on the real workdir: the freeze
# records whether it was made with quick settings and the stage refuses a mismatch)
QUICK = {"curve": {"n": [20, 40], "reps": 2, "reps_large": 1, "large_from": 40}, "recalibration_n": [20, 40],
         "hospitals": {"min_patients": 30, "draws": 2}, "n_boot_primary": 60, "n_boot": 40,
         "drift": {"n_ice": 40, "anchors": 600, "test": 400, "iters": 4, "joint_rounds": 1, "joint_iters": 3,
                   "chunk": 256, "k_candidates": 60}}


def xsettings(quick: bool) -> tuple[dict, dict, dict, dict]:
    cfg, mcfg, space = settings(quick)
    x = copy.deepcopy(cfg["external"])
    if quick:
        for k, v in QUICK.items():
            if isinstance(v, dict):
                x[k].update(v)
            else:
                x[k] = v
    return cfg, mcfg, space, x


def _write(path: Path, df: pl.DataFrame, info: dict) -> None:
    df.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    path.with_suffix(".tmp").replace(path)


def hospital_labels(lm: Landmarks, min_patients: int) -> dict[str, str]:
    """Hospitals with ≥ min_patients patients, labelled H1, H2, … by size (never by their identifier)."""
    if lm.hospital[0] is None:
        return {}
    d = pl.DataFrame({"h": lm.hospital.astype(str), "s": lm.ft.subjects.astype(str)}).unique()
    n = d.group_by("h").len().filter(pl.col("len") >= min_patients).sort(["len", "h"], descending=[True, False])
    return {h: f"H{i + 1}" for i, h in enumerate(n["h"].to_list())}


def gamma_vote(y_ref: np.ndarray, I: np.ndarray, valid: np.ndarray, own: np.ndarray, gamma: float, alpha: float,
               prior1: float) -> np.ndarray:
    """P(hypotension) with own-hospital neighbours weighted γ, the weights rescaled to the number of valid
    neighbours (γ = 1: perisim's smoothed vote exactly)."""
    w = np.where(own, gamma, 1.0) * valid
    nv = valid.sum(1).astype(np.float64)
    w = w * (nv / np.maximum(w.sum(1), 1e-12))[:, None]
    n1 = (w * (y_ref[I] == 1)).sum(1)
    p = (n1 + alpha * prior1 * 2) / (nv + alpha * 2)
    return np.where(nv > 0, p, prior1)


def explain(metric, Eq_w: np.ndarray, Er_w: np.ndarray, I: np.ndarray, valid: np.ndarray, y_ref: np.ndarray,
            p: np.ndarray, chunk: int = 4000) -> dict:
    """S8 aggregates by decile of p: each variable's share of the query–neighbour squared distance, and its pull
    (squared distance to the non-event neighbours minus that to the event neighbours)."""
    df = metric.enc.dim_feature
    F = metric.enc.n_features
    starts = np.searchsorted(df, np.arange(F))
    cuts = np.quantile(p, np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(p, cuts)
    share, pull = np.zeros((10, F)), np.zeros((10, F))
    n_share, n_pull = np.zeros(10), np.zeros(10)
    for s in range(0, len(p), chunk):
        sl = slice(s, s + chunk)
        d2 = np.add.reduceat((Eq_w[sl, None, :] - Er_w[I[sl]]) ** 2, starts, axis=2).astype(np.float64)
        v = valid[sl]
        tot = d2.sum(2)
        sh = np.where(v[..., None], d2 / np.maximum(tot, 1e-12)[..., None], 0.0).sum(1)
        nv = v.sum(1)
        ok = nv > 0
        ev, ne = (y_ref[I[sl]] == 1) & v, (y_ref[I[sl]] == 0) & v
        both = ev.any(1) & ne.any(1)
        pu = (d2 * ne[..., None]).sum(1) / np.maximum(ne.sum(1), 1)[:, None] - \
            (d2 * ev[..., None]).sum(1) / np.maximum(ev.sum(1), 1)[:, None]
        dd = dec[sl]
        for b in range(10):
            m = (dd == b) & ok
            share[b] += (sh[m] / nv[m][:, None]).sum(0)
            n_share[b] += m.sum()
            mb = (dd == b) & both
            pull[b] += pu[mb].sum(0)
            n_pull[b] += mb.sum()
    return {"variables": list(metric.enc.feature_names), "share_sum": share.tolist(), "share_n": n_share.tolist(),
            "pull_sum": pull.tolist(), "pull_n": n_pull.tolist(), "decile_cuts": cuts.tolist()}


def dry_dir() -> Path:
    return external_dir().with_name("external_dryrun")


class Stage:
    """The external stage. `dry=True` is the pre-freeze rehearsal on the real data with nothing external: each site
    X is served by the artefacts of a rotation whose development pair includes X (`dry_swap`), in-sample like
    Phase 2, written to external_dryrun/ (no freeze needed); it exercises every unit at the real sizes."""

    def __init__(self, *, quick: bool = False, threads: int = 16, dry: bool = False):
        self.quick, self.threads, self.dry = quick, threads, dry
        self.cfg, self.mcfg, self.space, self.x = xsettings(quick)
        self.S = Sites(self.cfg, cv_dir(False))
        self.out = dry_dir() if dry else external_dir()
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.cfg["seed"])
        self._m, self._E = {}, {}
        co = self.cfg["cohorts"]
        self.swap = {c: co[(i + 1) % len(co)] for i, c in enumerate(co)} if dry else {c: c for c in co}

    def development(self, ext: str) -> list[str]:
        """Sites inside the development pair of the artefacts that serve `ext`."""
        return [c for c in self.cfg["cohorts"] if c != self.swap[ext]]

    # ------------------------------------------------------------------ frozen artefacts and helpers
    def metric(self, ext: str):
        e = self.swap[ext]
        if e not in self._m:
            self._m = {e: load_pickle(final_dir() / f"metric_ext-{e}.pkl")}
        return self._m[e]

    def comp(self, ext: str, name: str):
        return load_pickle(final_dir() / f"comp_ext-{self.swap[ext]}_{name}.pkl")

    def embed(self, ext: str, site: str) -> np.ndarray:
        """The site's landmarks in the transported metric of rotation `ext` (unweighted), cached for one pair."""
        if (ext, site) not in self._E:
            self._E = {(ext, site): self.metric(ext).embed(self.S.lm(site))}
        return self._E[(ext, site)]

    def ci(self, c: str) -> int:
        return self.cfg["cohorts"].index(c)

    def draw(self, c: str, f: int, tr: np.ndarray, n: int, r: int) -> np.ndarray:
        """Positions in `tr` of the landmarks of n patients drawn at random (S1 and the recalibration share it)."""
        s = self.S.lm(c).ft.subjects[tr].astype(str)
        uniq = np.unique(s)
        rng = np.random.default_rng([self.seed, 7, self.ci(c), f, n, r])
        pick = rng.choice(uniq, size=min(n, uniq.size), replace=False)
        return np.where(np.isin(s, pick))[0]

    def reps(self, n: int) -> int:
        cv = self.x["curve"]
        return int(cv["reps_large"] if n >= int(cv["large_from"]) else cv["reps"])

    def case_base(self, metric, lm: Landmarks, rows: np.ndarray, E: np.ndarray, seed: int) -> tuple[CaseBase, dict]:
        cb = CaseBase(metric, lm.take(rows), self.mcfg, E=E[rows])
        return cb, cb.fit_alpha(seed)

    # ------------------------------------------------------------------ units
    def transported(self, X: str, f: int) -> None:
        path = self.out / f"transported_{X}_f{f}.parquet"
        if path.exists():
            return
        t = time.perf_counter()
        lm, (tr, te) = self.S.lm(X), self.S.split(X, f)
        metric, E = self.metric(X), self.embed(X, X)
        cb, a = self.case_base(metric, lm, tr, E, self.seed + 100 * f)
        q = lm.take(te)
        pr = cb.predict(q, E=E[te])
        I, valid = pr["I"], pr["valid"]
        ref = cb.lm
        yf = ref.frame["y_first"].cast(pl.Int64).to_numpy() if "y_first" in ref.frame.columns else ref.y
        art_r = ref.frame["art_line"].cast(pl.Float64).to_numpy()[I]
        art_q = q.frame["art_line"].cast(pl.Float64).to_numpy()[:, None]
        nv = np.maximum(valid.sum(1), 1)
        cols = {"row": te, "p": pr["p"], "ood": pr["ood"], "kth_pct": pr["kth_percentile"],
                **{f"p_band{b}": pr["p_band"][:, b] for b in range(3)},
                "p_first": cb.vote(I, valid, yf, 2)[:, 1],
                "art_share": ((art_r == art_q) & valid).sum(1) / nv,
                "nb_vaso_start": cb.carried(I, valid, "vaso_start"),
                "nb_map_min_next": cb.carried(I, valid, "map_min_next"),
                "nb_discharge_24h": cb.carried(I, valid, "icu_discharge_24h")}
        info = {"site": X, "fold": f, "n_case_base": int(tr.size), "n_test": int(te.size), **a}
        if lm.hospital[0] is not None:                          # H5 and the own-hospital share (S6)
            hr, hq = ref.hospital.astype(str), q.hospital.astype(str)
            own = hr[I] == hq[:, None]
            cols["own_share"] = (own & valid).sum(1) / nv
            L = cb.loo
            own_l = hr[L["I"]] == hr[L["rows"]][:, None]
            yl = ref.y[L["rows"]]
            prior1 = float(ref.y.mean())
            info["alpha_gamma"] = {}
            for g in [1.0] + [float(x) for x in self.x["gamma"]]:
                best = (np.inf, 1.0)
                for al in self.mcfg["alpha_grid"]:
                    pl_ = gamma_vote(ref.y, L["I"], L["valid"], own_l, g, float(al), prior1)
                    best = min(best, (float(logloss_rows(pl_, yl).mean()), float(al)))
                info["alpha_gamma"][str(g)] = {"alpha": best[1], "loo_logloss": best[0]}
                if g != 1.0:
                    cols[f"p_gamma{g:g}"] = gamma_vote(ref.y, I, valid, own, g, best[1], prior1)
                else:
                    cols["p_gamma1"] = gamma_vote(ref.y, I, valid, own, g, best[1], prior1)
        sw = metric.sqrt_w_dims
        info["explain"] = explain(metric, E[te] * sw, cb.Ew, I, valid, ref.y, pr["p"])
        info.update(seconds=time.perf_counter() - t, code_hash=code_hash())
        _write(path, pl.DataFrame(cols), info)
        LOG.info("transported %s fold %d: %.0f s", X, f, info["seconds"])

    def local(self, X: str, f: int) -> None:
        path = self.out / f"local_{X}_f{f}.parquet"
        if path.exists():
            return
        t = time.perf_counter()
        lm, (tr, te) = self.S.lm(X), self.S.split(X, f)
        seed = self.seed + 100 * f
        pool = lm.take(tr)
        rows = learning_sample(pool, int(self.mcfg["learn"]["per_patient"]), int(self.mcfg["learn"]["per_site"]), seed)
        metric = fit_metric(pool.take(rows), self.mcfg, self.space, seed=seed, threads=self.threads)
        cb = CaseBase(metric, pool, self.mcfg)
        a = cb.fit_alpha(seed)
        pr = cb.predict(lm.take(te))
        m = metric.info
        info = {"site": X, "fold": f, **a, "weights": metric.weights_table(),
                "shaped": {n: d["shaped"] for n, d in m["shape_diag"].items()},
                "dropped_site_constant": m["dropped_site_constant"], "n_accepted": m["n_accepted"],
                "metric_seconds": m["seconds"], "seconds": time.perf_counter() - t, "code_hash": code_hash()}
        _write(path, pl.DataFrame({"row": te, "p": pr["p"], "ood": pr["ood"], "kth_pct": pr["kth_percentile"],
                                   **{f"p_band{b}": pr["p_band"][:, b] for b in range(3)}}), info)
        LOG.info("local %s fold %d: %.0f s", X, f, info["seconds"])

    def comps(self, X: str, f: int) -> None:
        path = self.out / f"comps_{X}_f{f}.parquet"
        if path.exists():
            return
        t = time.perf_counter()
        lm, (tr, te) = self.S.lm(X), self.S.split(X, f)
        q, ref = lm.take(te), lm.take(tr)
        cols = {"row": te}
        for name in self.x["comparators"]:
            pred = self.comp(X, name)
            p_te = np.asarray(pred(q), dtype=np.float64)
            p_tr = np.asarray(pred(ref), dtype=np.float64)
            cols[f"p_{name}"] = p_te
            for n in self.x["recalibration_n"]:
                for r in range(self.reps(int(n))):
                    pos = self.draw(X, f, tr, int(n), r)
                    cols[f"p_{name}_n{n}_r{r}"] = recalibrate(p_tr[pos], ref.y[pos], p_te)
        info = {"site": X, "fold": f, "seconds": time.perf_counter() - t, "code_hash": code_hash()}
        _write(path, pl.DataFrame(cols), info)
        LOG.info("comparators %s fold %d: %.0f s", X, f, info["seconds"])

    def curve(self, X: str, f: int) -> None:
        path = self.out / f"curve_{X}_f{f}.parquet"
        if path.exists():
            return
        t = time.perf_counter()
        lm, (tr, te) = self.S.lm(X), self.S.split(X, f)
        metric, E = self.metric(X), self.embed(X, X)
        n_pat = np.unique(lm.ft.subjects[tr].astype(str)).size
        q = lm.take(te)
        cols, alphas = {"row": te}, {}
        for n in self.x["curve"]["n"]:
            if int(n) >= n_pat:
                continue
            for r in range(self.reps(int(n))):
                pos = self.draw(X, f, tr, int(n), r)
                cb, a = self.case_base(metric, lm, tr[pos], E, self.seed + 100 * f + r)
                cols[f"p_n{n}_r{r}"] = cb.predict(q, E=E[te])["p"]
                alphas[f"n{n}_r{r}"] = {"alpha": a["alpha"], "landmarks": int(pos.size)}
        info = {"site": X, "fold": f, "n_patients_pool": int(n_pat), "case_bases": alphas,
                "seconds": time.perf_counter() - t, "code_hash": code_hash()}
        _write(path, pl.DataFrame(cols), info)
        LOG.info("curve %s fold %d: %.0f s", X, f, info["seconds"])

    def hospitals(self, X: str, f: int) -> None:
        path = self.out / f"hospitals_{X}_f{f}.parquet"
        lm = self.S.lm(X)
        if path.exists() or lm.hospital[0] is None:
            return
        t = time.perf_counter()
        tr, te = self.S.split(X, f)
        metric, E = self.metric(X), self.embed(X, X)
        labels = hospital_labels(lm, int(self.x["hospitals"]["min_patients"]))
        R = int(self.x["hospitals"]["draws"])
        subj = lm.ft.subjects.astype(str)
        hosp = lm.hospital.astype(str)
        parts, info_g = [], {}

        def contrast(kind: str, label: str, gi: int, in_group: np.ndarray) -> None:
            qr, A, other = te[in_group[te]], tr[in_group[tr]], tr[~in_group[tr]]
            if qr.size == 0 or A.size == 0:
                return
            seed = self.seed + 100 * f + gi
            yq = lm.y[qr]
            cbA, aA = self.case_base(metric, lm, A, E, seed)
            ll_a = logloss_rows(cbA.predict(lm.take(qr), E=E[qr])["p"], yq)
            pi_a = float(lm.y[A].mean())
            n_a = np.unique(subj[A]).size
            pool = np.unique(subj[other])
            ll_b = np.zeros(qr.size)
            for r in range(R):
                rng = np.random.default_rng([self.seed, 11, f, gi, r, int(kind == "control")])
                pick = rng.choice(pool, size=min(n_a, pool.size), replace=False)
                B = other[np.isin(subj[other], pick)]
                cbB, _ = self.case_base(metric, lm, B, E, seed + r + 1)
                pb = cbB.predict(lm.take(qr), E=E[qr])["p"]
                ll_b += logloss_rows(label_shift(pb, pi_a, float(lm.y[B].mean())), yq) / R
            parts.append(pl.DataFrame({"kind": kind, "group": label, "row": qr, "ll_own": ll_a, "ll_other": ll_b}))
            info_g[f"{kind}:{label}"] = {"n_test": int(qr.size), "n_case_base": int(A.size), "patients": int(n_a),
                                         "alpha": aA["alpha"]}

        for gi, (h, lab) in enumerate(labels.items()):
            contrast("hospital", lab, gi, hosp == h)
        # control: every patient of the site assigned at random to groups of the eligible hospitals' sizes
        uniq = np.unique(subj)
        sizes = [np.unique(subj[hosp == h]).size for h in labels]
        perm = np.random.default_rng([self.seed, 13, f]).permutation(uniq)
        cut = np.r_[0, np.cumsum(sizes)]
        for gi, lab in enumerate(labels.values()):
            members = perm[cut[gi]:cut[gi + 1]]
            contrast("control", lab, 1000 + gi, np.isin(subj, members))
        info = {"site": X, "fold": f, "groups": info_g, "draws": R, "seconds": time.perf_counter() - t,
                "code_hash": code_hash()}
        _write(path, pl.concat(parts) if parts else pl.DataFrame(), info)
        LOG.info("hospitals %s fold %d: %.0f s", X, f, info["seconds"])

    # ------------------------------------------------------------------ the whole stage
    def run(self, externals: list[str], folds: list[int] | None = None, parts: list[str] | None = None) -> Path:
        if self.dry:
            rec = {"tag": "dry run (no freeze)"}
        else:
            rec = check()
            if bool(rec.get("quick", False)) != self.quick:
                raise Stop("the freeze was made with other settings (quick) than this run")
        from . import drift

        folds = list(range(int(self.cfg["cv"]["outer_folds"]))) if folds is None else folds
        parts = parts or ["transported", "local", "comps", "curve", "hospitals", "drift", "shapes"]
        log = self.out / "stage.json"
        st = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {"runs": []}
        st["runs"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": rec["tag"], "externals": externals,
                           "folds": folds, "parts": parts})
        log.write_text(json.dumps(st, indent=1), encoding="utf-8")
        for X in externals:
            for f in folds:
                for unit in ("transported", "local", "comps", "curve", "hospitals"):
                    if unit in parts:
                        getattr(self, unit)(X, f)
            if "drift" in parts:
                for c in self.cfg["cohorts"]:
                    if not self.dry or c in self.development(X):    # a dry run never touches an external site
                        drift.site_drift(self, X, c)
                drift.hospital_drift(self, X)
        if "shapes" in parts:
            for c in (externals if self.dry else self.cfg["cohorts"]):
                drift.site_shapes(self, c)
        if not self.dry:
            check()                                        # nothing changed while it ran
        st["runs"][-1]["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write_text(json.dumps(st, indent=1), encoding="utf-8")
        return self.out
