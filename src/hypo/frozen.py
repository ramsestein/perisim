"""Phase 3: the frozen artefacts, the freeze record and its check (docs/prereg_hypo.md §4).

- `final`: per rotation (named by its external site X), the transported metric learned once on all patients of the
  two development sites (Phase 1 procedure, seed + `external.final_seed_offset`) and the transported comparators
  fitted on the same landmarks pooled (no site column). Saved under <workdir>/final/ (git-ignored: the encoder keeps
  training values), with an aggregate .json beside each and reports/final.md.
- `freeze`: refused when external results exist or the repository has uncommitted changes; an annotated git tag
  `perisim-freeze-hypo-<time>`, a snapshot of the configuration and the hashes of the code, the configuration, the
  Phase 2 folds and local comparators and every frozen artefact, in <workdir>/state/freeze.json.
- `check`: the external stage's gate; stops when anything differs from the record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from joblib.externals import cloudpickle

from . import compare
from .config import REPO, code_hash, load, workdir
from .cv import cv_dir, settings
from .data import concat, learning_sample
from .data import load as load_lm
from .metric import fit_metric

LOG = logging.getLogger("hypo.frozen")


class Stop(RuntimeError):
    pass


def final_dir() -> Path:
    return workdir() / "final"


def state_dir() -> Path:
    return workdir() / "state"


def external_dir() -> Path:
    return workdir() / "external"


def dump(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        cloudpickle.dump(obj, fh)
    tmp.replace(path)


def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return cloudpickle.load(fh)


def sha(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def config_hash(n: int = 12) -> str:
    h = hashlib.sha256()
    for f in ("hypo.yaml", "model_space.yaml"):
        h.update(f.encode())
        h.update((REPO / "config" / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:n]


# ---------------------------------------------------------------------------- MAP thresholds from shape curves
def logodds_curve(enc, name: str, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bin mid-points, log-odds of hypotension) of a continuous block's shape table (C = 2: column 1 − column 0)."""
    b = enc.blocks[enc.feature_index()[name]]
    e = b.params["edges"]
    return 0.5 * (e[:-1] + e[1:]), np.asarray(g)[:, 1] - np.asarray(g)[:, 0]


def threshold(x: np.ndarray, lo: np.ndarray, ref: float, rise: float) -> float:
    """Highest x below `ref` at which the curve (linear between the points) is ≥ `rise` above its value at `ref`;
    NaN when it never is or `ref` is outside the curve's range."""
    if x.size < 2 or not (x[0] <= ref <= x[-1]):
        return float("nan")
    target = float(np.interp(ref, x, lo)) + rise
    xs = np.r_[x[x < ref], ref]
    ys = np.r_[lo[x < ref], target - rise]
    for i in range(xs.size - 1, 0, -1):            # walk down from ref: the first segment that reaches the target
        a, b = ys[i - 1], ys[i]
        if a >= target:
            if b >= target:
                return float(xs[i])
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (target - a) / (b - a)) if b != a else float(xs[i - 1])
    return float("nan")


def thresholds(enc, curves: dict, shapes_of: dict | None, tcfg: dict) -> dict:
    """Per MAP variable: the threshold of the consensus g_j and the median / 10–90 % over the accepted models."""
    out = {}
    for v in tcfg["variables"]:
        if v not in enc.feature_index():
            continue
        b = enc.blocks[enc.feature_index()[v]]
        per = [threshold(*logodds_curve(enc, v, c), float(tcfg["reference"]), float(tcfg["rise"]))
               for c in curves.get(v, [])]
        per = np.asarray(per, float)
        fin = per[np.isfinite(per)]
        g = b.shape if shapes_of is None else shapes_of.get(v)
        cons = threshold(*logodds_curve(enc, v, g), float(tcfg["reference"]), float(tcfg["rise"])) \
            if g is not None else float("nan")
        out[v] = {"consensus": cons, "models": int(per.size), "models_with_threshold": int(fin.size),
                  "median": float(np.median(fin)) if fin.size else float("nan"),
                  "p10": float(np.percentile(fin, 10)) if fin.size else float("nan"),
                  "p90": float(np.percentile(fin, 90)) if fin.size else float("nan")}
    return out


def curve_table(enc, name: str, g: np.ndarray, min_cell: int) -> list[dict]:
    b = enc.blocks[enc.feature_index()[name]]
    vc = b.params.get("value_counts")
    if b.kind == "continuous":
        x, lo = logodds_curve(enc, name, g)
        return [{"x": float(xi), "logodds": float(li), "n": int(round(float(vc[i]), -1))}
                for i, (xi, li) in enumerate(zip(x, lo)) if vc is None or vc[i] >= min_cell]
    lo = np.asarray(g)[:, 1] - np.asarray(g)[:, 0]
    return [{"x": str(c), "logodds": float(li), "n": int(round(float(vc[i]), -1))}
            for i, (c, li) in enumerate(zip(b.params["values"], lo)) if vc is None or vc[i] >= min_cell]


def metric_summary(metric, xcfg: dict) -> dict:
    """Aggregates of a metric: weights, shapes (log-odds per bin, n ≥ min_cell), MAP thresholds."""
    enc, m = metric.enc, metric.info
    return {"weights": metric.weights_table(), "dropped_site_constant": m["dropped_site_constant"],
            "shaped": {n: d["shaped"] for n, d in m["shape_diag"].items()},
            "shapes": {b.name: curve_table(enc, b.name, b.shape, int(xcfg["min_cell"]))
                       for b in enc.blocks if b.shape is not None},
            "thresholds": thresholds(enc, m["curves"], None, xcfg["threshold"]),
            "n_learn": m["n_learn"], "learn_per_site": m["learn_per_site"], "n_accepted": m["n_accepted"],
            "n_competitive": m["n_competitive"], "metric_seconds": m["seconds"]}


# ---------------------------------------------------------------------------- the frozen artefacts
def final(*, quick: bool = False, threads: int = 16) -> Path:
    cfg, mcfg, space = settings(quick)
    xcfg = cfg["external"]
    out = final_dir()
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"]) + int(xcfg["final_seed_offset"])
    lms = {}
    for ext in cfg["cohorts"]:
        dev = [c for c in cfg["cohorts"] if c != ext]
        for c in dev:
            if c not in lms:
                lms[c] = load_lm(c, cfg)
        pool = concat([lms[c] for c in dev])
        path = out / f"metric_ext-{ext}.pkl"
        if not path.exists():
            t = time.perf_counter()
            rows = learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)
            metric = fit_metric(pool.take(rows), mcfg, space, seed=seed, threads=threads)
            info = {"external": ext, "development": dev, "seed": seed, "quick": quick, "code_hash": code_hash(),
                    "seconds": time.perf_counter() - t, **metric_summary(metric, xcfg)}
            path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
            dump(metric, path)
            LOG.info("final metric external-%s: %.0f s", ext, info["seconds"])
        for name in xcfg["comparators"]:
            path = out / f"comp_ext-{ext}_{name}.pkl"
            if path.exists():
                continue
            t = time.perf_counter()
            pred, cinfo = compare.fit(name, pool, mcfg, space["baselines"], n_inner=int(cfg["cv"]["inner_folds"]),
                                      seed=seed, threads=threads)
            info = {"external": ext, "development": dev, "method": name, "n_train": len(pool), "seed": seed,
                    "quick": quick, "code_hash": code_hash(), "seconds": time.perf_counter() - t, **cinfo}
            path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
            dump(pred, path)
            LOG.info("final %s external-%s: %.0f s", name, ext, info["seconds"])
    render_final(cfg)
    return out


def render_final(cfg: dict) -> None:
    L = ["# `hypo` Phase 3: the frozen artefacts", "",
         "Per rotation (named by its external site), the transported metric learned on all patients of the two "
         "development sites, and the transported comparators. Weights sum to 1; MAP thresholds: the highest MAP below "
         f"{cfg['external']['threshold']['reference']} mmHg where the odds of hypotension double "
         "(consensus g_j; median and 10–90 % over the accepted models).", ""]
    for ext in cfg["cohorts"]:
        p = final_dir() / f"metric_ext-{ext}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        L += [f"## Metric for external-{ext} (learned on {' + '.join(r['development'])}; code {r['code_hash']})", "",
              f"{r['n_learn']:,} learning landmarks {r['learn_per_site']}; {r['n_accepted']} accepted models "
              f"({r['n_competitive']} competitive); left out (constant within each site): "
              f"{', '.join(r['dropped_site_constant']) or 'none'}; {r['seconds'] / 60:.0f} min.", "",
              "| variable | weight | shaped |", "|---|---|---|"]
        for n, w in sorted(r["weights"].items(), key=lambda x: -x[1]):
            L.append(f"| {n} | {w:.3f} | {'yes' if r['shaped'].get(n) else 'no'} |")
        L += ["", "| MAP threshold | consensus | models: median (10–90 %) | models with one |", "|---|---|---|---|"]
        for v, t in r["thresholds"].items():
            L.append(f"| {v} | {t['consensus']:.1f} | {t['median']:.1f} ({t['p10']:.1f}–{t['p90']:.1f}) | "
                     f"{t['models_with_threshold']}/{t['models']} |")
        comps = []
        for name in cfg["external"]["comparators"]:
            q = final_dir() / f"comp_ext-{ext}_{name}.json"
            if q.exists():
                c = json.loads(q.read_text(encoding="utf-8"))
                comps.append(f"{name} ({c['n_train']:,} landmarks, {c['seconds']:.0f} s)")
        L += ["", "Transported comparators: " + (", ".join(comps) or "—") + ".", ""]
    rep = workdir() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "final.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- freeze record and check
def _hashes(cfg: dict) -> dict:
    """Everything the external stage depends on, hashed."""
    d, fd = cv_dir(False), final_dir()
    comps = sorted(d.glob("comp_*_f*_*.parquet"))
    h = hashlib.sha256()
    for f in comps:
        h.update(f.name.encode())
        h.update(sha(f, 64).encode())
    art = sorted(list(fd.glob("*.pkl")))
    return {"code": code_hash(), "config": config_hash(),
            "folds": {c: sha(d / f"folds_{c}.parquet") for c in cfg["cohorts"] if (d / f"folds_{c}.parquet").exists()},
            "phase2_comparators": {"files": len(comps), "sha": h.hexdigest()[:16]},
            "final": {f.name: sha(f) for f in art}}


def _expected_artefacts(cfg: dict) -> list[str]:
    return [f"metric_ext-{e}.pkl" for e in cfg["cohorts"]] + \
        [f"comp_ext-{e}_{n}.pkl" for e in cfg["cohorts"] for n in cfg["external"]["comparators"]]


def freeze(registration: str | None, *, no_git: bool = False, quick: bool = False) -> dict:
    """`quick` (tests only): accept artefacts made with quick settings; recorded, and the stage must match it."""
    cfg = load()
    ext = external_dir()
    if ext.exists() and any(ext.glob("*.parquet")):
        raise Stop("external results already exist: freezing again would allow iterating after seeing them")
    missing = [a for a in _expected_artefacts(cfg) if not (final_dir() / a).exists()]
    if missing:
        raise Stop(f"frozen artefacts missing (run `python -m hypo final` first): {', '.join(missing)}")
    cur = code_hash()
    for j in final_dir().glob("*.json"):
        info = json.loads(j.read_text(encoding="utf-8"))
        if info.get("code_hash") != cur or bool(info.get("quick")) != quick:
            raise Stop(f"{j.name} was made with other code ({info.get('code_hash')}) or quick settings: "
                       "remove final/ and run `python -m hypo final` again")
    folds = [c for c in cfg["cohorts"] if not (cv_dir(False) / f"folds_{c}.parquet").exists()]
    if folds:
        raise Stop(f"Phase 2 folds missing for {', '.join(folds)}")
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    tag = f"perisim-freeze-hypo-{stamp}"
    git = {}
    if not no_git:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                               text=True).stdout
        if dirty.strip():
            raise Stop("the repository has uncommitted changes: commit them before freezing\n" + dirty)
        r = subprocess.run(["git", "-C", str(REPO), "-c", "user.name=perisim", "-c", "user.email=perisim@localhost",
                            "tag", "-a", tag, "-m", "PERISIM freeze, experiment hypo (docs/prereg_hypo.md)"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise Stop(f"could not create the git tag {tag}: {r.stderr.strip()}")
        git = {"tag": tag, "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                    capture_output=True, text=True).stdout.strip()}
    st = state_dir()
    (st / "frozen_config").mkdir(parents=True, exist_ok=True)
    for f in ("hypo.yaml", "model_space.yaml"):
        shutil.copyfile(REPO / "config" / f, st / "frozen_config" / f)
    rec = {"tag": tag, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "registration": registration or "PENDING",
           "quick": quick,
           "git": git, "hashes": _hashes(cfg), "prereg_sha": sha(REPO / "docs" / "prereg_hypo.md")}
    (st / "freeze.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    LOG.info("frozen: %s (registration %s)", tag, rec["registration"])
    return rec


def check() -> dict:
    """The freeze record, after checking that nothing it covers has changed."""
    p = state_dir() / "freeze.json"
    if not p.exists():
        raise Stop("no freeze record: the external stage runs only on the frozen version (`python -m hypo freeze`)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    now = _hashes(load())
    diff = [k for k in ("code", "config", "folds", "phase2_comparators", "final") if now[k] != rec["hashes"][k]]
    if diff:
        raise Stop(f"changed since the freeze {rec['tag']}: {', '.join(diff)}. Restore the frozen version "
                   f"(git checkout {rec['tag']}); a change after external results exist is a deviation")
    return rec
