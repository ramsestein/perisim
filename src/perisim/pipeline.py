"""Stage orchestration: idempotent and resumable (an OOM or a pod restart never means starting over).

Each stage has a key = hash(relevant configuration + relevant code + keys of upstream stages). If the key
matches the marker in state/, the stage is skipped. Governance gates stop the pipeline with actionable
instructions:
  - §7   completeness audit (prevalence < 0.5 % or missingness > 30 %) → OPEN_QUESTIONS;
  - D7   stopping parameters fixed with the pilot BEFORE the full pipeline (pre-declared rule);
  - §6.3 external validation requires a freeze (code and configuration hashes).

Compute (§9.1): processes × threads come from the container's real limits (perisim.resources) — at most
4 jobs × 4 threads. The three cohorts are extracted in parallel and all sensitivity variants share one job
queue, so no cores sit idle between variants.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from perisim.config import REPO_ROOT, Config, code_hash, stable_hash
from perisim.resources import apply_library_limits, plan
from perisim.utils import (LOG, append_note, dir_size_gb, free_disk_gb, now_iso, read_json, rss_gb, set_threads,
                           timed, write_json, write_parquet)
from perisim.workspace import Workspace

COHORTS = ("inspire", "mimic", "eicu")
MAIN_STAGES = ["env", "discover", "extract", "targets", "features", "leakage", "audit", "harmonize", "pilot", "cv",
               "temporal", "report", "sensitivity", "external", "export"]
VARIANT_PREP_STAGES = ["targets", "features", "leakage"]
GB_PER_EXTRACTION = 24.0   # memory budget of one cohort extraction (MIMIC-IV chartevents/labevents dominate)


class Stop(Exception):
    """A governance gate is closed: the pipeline stops with an actionable message."""


@dataclass
class StageDef:
    config: tuple[str, ...]
    code: tuple[str, ...]
    upstream: tuple[str, ...]
    per_cohort: bool = False


# Config dependencies may contain "{cohort}" (per-cohort slices). Only what changes a stage's OUTPUT is listed,
# so that e.g. a bootstrap setting does not re-run the CV (AUDIT F6); upstream stages are chained through the
# run_id of their last execution (AUDIT A5).
_EXTRACT_CFG = ("harmonization.sources.{cohort}", "harmonization.variables", "harmonization.medications",
                "harmonization.surgical_groups", "harmonization.admit_source", "harmonization.discharge_location",
                "cohorts.common", "cohorts.{cohort}", "experiment.features.preop_lab_lookback_days", "targets.window")
_MODEL_CFG = ("experiment.importance", "experiment.metric", "experiment.knn", "experiment.cv.outer_folds",
              "experiment.cv.inner_folds", "model_space", "experiment.seed", "experiment.features.feature_set")
STAGE_DEFS: dict[str, StageDef] = {
    "env": StageDef((), ("utils.py", "resources.py"), ()),
    "discover": StageDef(("harmonization.sources", "experiment.paths.datasets_root", "experiment.paths.inspire_dir",
                          "experiment.paths.mimic_dir", "experiment.paths.eicu_dir"), ("data",), ()),
    "extract": StageDef(_EXTRACT_CFG, ("data", "cohort"), ("discover",), True),
    "targets": StageDef(("targets", "experiment.time", "experiment.features.support_gap_hours"),
                        ("targets", "timeline.py"), ("extract",), True),
    "features": StageDef(("experiment.features", "experiment.time", "harmonization.medications",
                          "harmonization.preop_labs"), ("features", "timeline.py", "cohort"), ("extract",), True),
    "leakage": StageDef(("experiment.time",), ("features",), ("features",)),
    "audit": StageDef(("experiment.evaluation.small_cell_threshold", "targets.t1"),
                      ("evaluation/audit.py", "evaluation/html.py", "evaluation/privacy.py"), ("targets", "features")),
    "harmonize": StageDef(("experiment.features", "experiment.cohorts"), ("pipeline.py",), ("features", "audit")),
    "pilot": StageDef(_MODEL_CFG + ("experiment.pilot",), ("importance", "metric", "knn", "experiment.py", "pipeline.py"),
                      ("harmonize", "targets")),
    "cv": StageDef(_MODEL_CFG + ("experiment.baselines", "experiment.targets_to_model", "targets.t2", "targets.t3"),
                   ("importance", "metric", "knn", "experiment.py", "baselines"), ("pilot", "harmonize", "targets")),
    "temporal": StageDef(("experiment.cv.temporal",), (), ("cv",)),
    "report": StageDef(("experiment.evaluation", "experiment.knn"), ("evaluation",), ("cv",)),
    "sensitivity": StageDef(("experiment.sensitivity",), (), ("cv",)),
    "external": StageDef(_MODEL_CFG + ("experiment.cohorts", "experiment.targets_to_model", "experiment.evaluation",
                                       "experiment.time", "targets"),
                         ("experiment.py", "external.py", "evaluation"), ("cv", "harmonize")),
    "export": StageDef((), ("evaluation",), ()),
}
# Stages computed once for the main analysis and shared by the sensitivity variants
_SHARED_STAGES = {"env": "_", "discover": "_", "extract": "_", "harmonize": "main", "pilot": "main"}


def _get(d: dict, dotted: str):
    node = d
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


@contextmanager
def _child_threads(n: int):
    """Thread variables inherited by spawned workers (read at import time in the child)."""
    keys = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = str(n)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Pipeline:
    def __init__(self, cfg: Config, ws: Workspace | None = None, force: bool = False):
        self.cfg = cfg
        self.ws = ws or Workspace(cfg.workdir, cfg.variant)
        self.force = force
        self.variant = cfg.variant
        self._keys: dict[str, str] = {}
        self.disc = read_json(self.ws.shared_artifacts() / "discovery.json", {}) or {}
        self.res = plan(cfg.experiment.get("resources"))
        apply_library_limits(self.res["cpus"])

    # ------------------------------------------------------------------ keys and markers
    def key(self, stage: str, cohort: str | None = None) -> str:
        """hash(relevant config + relevant code + key AND output digest of every upstream stage).

        Including the upstream OUTPUT digest means that an upstream re-run that changes its outputs (``--force``,
        a re-discovered cohort, a cohort mounted later…) invalidates everything downstream, while a re-run that
        reproduces identical outputs invalidates nothing (AUDIT A5)."""
        cache_k = f"{stage}:{cohort}"
        if cache_k in self._keys:
            return self._keys[cache_k]
        sd = STAGE_DEFS[stage]
        cfgd = self.cfg.to_dict()
        conf = {}
        for path in sd.config:
            if "{cohort}" in path:
                for c in ([cohort] if cohort else COHORTS):
                    conf[path.format(cohort=c)] = _get(cfgd, path.format(cohort=c))
            else:
                conf[path] = _get(cfgd, path)
        up = []
        for u in sd.upstream:
            cs = ([cohort] if cohort else self.available_cohorts()) if STAGE_DEFS[u].per_cohort else [None]
            for c in cs:
                m = read_json(self._marker(u, c)) or {}
                up.append([u, c, self.key(u, c), m.get("digest") or m.get("run_id")])
        payload = {"stage": stage, "cohort": cohort, "variant": _SHARED_STAGES.get(stage, self.variant),
                   "config": conf, "code": code_hash(list(sd.code)) if sd.code else None, "up": up}
        if stage == "extract" and cohort:
            payload["src"] = self.disc.get(cohort, {}).get("path")
        k = stable_hash(payload, 16)
        self._keys[cache_k] = k
        return k

    def _marker(self, stage: str, cohort: str | None) -> Path:
        variant = _SHARED_STAGES.get(stage, self.variant)
        d = self.ws.state / variant
        d.mkdir(parents=True, exist_ok=True)
        return d / (f"{stage}__{cohort}.json" if cohort else f"{stage}.json")

    def done(self, stage: str, cohort: str | None = None) -> bool:
        m = read_json(self._marker(stage, cohort))
        return bool(m) and m.get("key") == self.key(stage, cohort) and m.get("status") == "done"

    def mark(self, stage: str, cohort: str | None = None, outputs: list[Path] | None = None,
             digest: str | None = None, **info) -> None:
        """Marks a stage done. `digest` (or the content hash of `outputs`) identifies WHAT it produced; stages
        without declared outputs get a random run_id, so any re-run invalidates their dependants."""
        import uuid

        if digest is None and outputs:
            digest = _files_digest(outputs)
        write_json(self._marker(stage, cohort), {"key": self.key(stage, cohort), "status": "done",
                                                 "digest": digest, "run_id": uuid.uuid4().hex,
                                                 "finished": now_iso(), **info})
        self._keys.clear()          # downstream keys depend on this digest

    # ------------------------------------------------------------------ helpers
    def available_cohorts(self) -> list[str]:
        return [c for c in COHORTS if self.disc.get(c, {}).get("path") and self.disc.get(c, {}).get("schema_ok", True)]

    def threads(self) -> int:
        return int(self.res["threads_per_job"])

    def jobs(self) -> int:
        return int(self.res["jobs"])

    def note(self, kind: str, title: str, body: str) -> None:
        append_note(self.ws.docs / f"{kind}.md", f"[{self.variant}] {title}", body)

    def check_disk(self, label: str) -> None:
        free = free_disk_gb(self.ws.root)
        LOG.info("Disk after %s: %.1f GB free; workdir %.2f GB", label, free, dir_size_gb(self.ws.root))
        if np.isfinite(free) and free < float(self.cfg.experiment["resources"]["min_free_disk_gb"]):
            raise Stop(f"Only {free:.1f} GB free (< min_free_disk_gb). Free some space in {self.ws.root} "
                       f"(e.g. delete {self.ws.root / 'cache'}) and relaunch.")

    # ------------------------------------------------------------------ run
    def run(self, stages: list[str] | None = None) -> int:
        stages = stages or MAIN_STAGES
        t0 = time.perf_counter()
        for st in stages:
            fn = getattr(self, f"stage_{st}")
            try:
                with timed(f"stage {st} [{self.variant}]"):
                    fn()
            except Stop as exc:
                LOG.warning("⛔ Pipeline stopped at stage '%s':\n%s", st, exc)
                self.note("OPEN_QUESTIONS", f"Stopped at {st}", str(exc))
                write_json(self.ws.state / "last_stop.json", {"stage": st, "message": str(exc), "at": now_iso()})
                return 3
            except Exception:
                LOG.error("✖ Error in stage '%s':\n%s", st, traceback.format_exc())
                write_json(self.ws.state / "last_error.json", {"stage": st, "trace": traceback.format_exc(), "at": now_iso()})
                return 1
        (self.ws.state / "last_stop.json").unlink(missing_ok=True)
        (self.ws.state / "last_error.json").unlink(missing_ok=True)
        LOG.info("✔ Stages completed in %.1f min (variant %s)", (time.perf_counter() - t0) / 60, self.variant)
        return 0

    # ================================================================== stages
    def stage_env(self) -> None:
        import importlib

        info = {"python": sys.version, "platform": platform.platform(), "os_cpu_count": os.cpu_count(),
                "resources_plan": self.res, "code_hash": code_hash(), "config_hash": self.cfg.hash(),
                "profile": self.cfg.profile, "workdir": str(self.ws.root), "time": now_iso(),
                "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS")}}
        info["free_disk_gb"] = round(free_disk_gb(self.ws.root), 1)
        vers = {}
        for mod in ("numpy", "polars", "pyarrow", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "torch", "faiss",
                    "yaml", "psutil", "matplotlib", "sage"):
            try:
                m = importlib.import_module(mod)
                vers[mod] = getattr(m, "__version__", "?")
            except Exception as exc:
                vers[mod] = f"NOT AVAILABLE ({type(exc).__name__})"
        info["packages"] = vers
        missing = [m for m in ("numpy", "polars", "pyarrow", "scipy", "sklearn", "lightgbm", "xgboost", "torch", "yaml")
                   if str(vers[m]).startswith("NOT")]
        write_json(self.ws.shared_artifacts() / "env.json", info)
        if sys.version_info < (3, 11):
            raise Stop("Python ≥ 3.11 is required (§8.1)")
        if missing:
            raise Stop(f"Missing packages: {missing}. Run: bash run_pod.sh setup")
        probe = self.ws.state / ".probe"
        probe.write_text("ok")
        os.replace(probe, self.ws.state / ".probe2")
        (self.ws.state / ".probe2").unlink()
        r = self.res
        LOG.info("Resources: %d usable CPUs (host reports %s; cgroup quota %s), %s GB memory (cgroup limit %s) "
                 "→ %d jobs × %d threads; %.1f GB free disk; faiss=%s", r["cpus"], os.cpu_count(), r["cgroup_cpu_quota"],
                 r["memory_gb"], r["cgroup_memory_gb"], r["jobs"], r["threads_per_job"], info["free_disk_gb"],
                 vers["faiss"])
        if r["jobs"] * r["threads_per_job"] < 16 and r["cpus"] < 16:
            LOG.warning("The container has fewer than the 16 vCPUs of SPECS §9.1: the run will be slower than the pilot "
                        "estimate assumes for 4 × 4.")
        self.mark("env")

    def stage_discover(self) -> None:
        from perisim.data.discover import discover_cohorts, schema_markdown, schema_report

        paths_ok = all(self.disc.get(c, {}).get("path") and os.path.isdir(self.disc[c]["path"]) for c in COHORTS)
        if self.done("discover") and not self.force and paths_ok:
            LOG.info("discover: unchanged (%s)", ", ".join(f"{c}={self.disc.get(c, {}).get('path')}" for c in COHORTS))
            return
        if self.done("discover") and not paths_ok:
            LOG.info("discover: a cohort is missing or its folder disappeared → searching again")
        disc = discover_cohorts(self.cfg)
        min_n = int(self.cfg.experiment["evaluation"]["small_cell_threshold"])
        for c in COHORTS:
            p = disc[c].get("path")
            if not p:
                continue
            rep = schema_report(self.cfg, c, Path(p), self.ws.cache(c), min_n)
            write_json(self.ws.reports / f"schema_{c}.json", rep)
            (self.ws.reports / f"schema_{c}.md").write_text(schema_markdown(rep), encoding="utf-8")
            disc[c]["schema_ok"] = rep["ok"]
            if not rep["ok"]:
                miss = {t: e.get("missing") for t, e in rep["tables"].items() if e.get("missing")}
                self.note("OPEN_QUESTIONS", f"{c} schema does not match",
                          f"Columns not found: {miss}. Fix config/harmonization.yaml (see reports/schema_{c}.md).")
        write_json(self.ws.shared_artifacts() / "discovery.json", disc)
        self.disc = disc
        self._keys.clear()
        self._disc_digest = stable_hash({c: disc[c].get("path") for c in COHORTS})
        dev = self.cfg.experiment["cohorts"]["development"]
        if not disc[dev].get("path"):
            raise Stop(f"Development cohort ({dev}) not found under {disc['_datasets_root']['path']}.\n"
                       f"Top-level folders: {disc['_datasets_root']['top_level'][:40]}\n"
                       f"Set PERISIM_INSPIRE_DIR=/path/to/inspire and relaunch.")
        if not disc[dev].get("schema_ok", True):
            raise Stop(f"The {dev} schema does not match config/harmonization.yaml: see {self.ws.reports}/schema_{dev}.md")
        self.mark("discover", digest=self._disc_digest)

    def stage_extract(self) -> None:
        """Common-format extraction; the cohorts run in parallel processes when CPU and memory allow."""
        todo = [c for c in self.available_cohorts() if self.force or not self.done("extract", c)]
        for c in self.available_cohorts():
            if c not in todo:
                LOG.info("extract[%s]: unchanged", c)
        if not todo:
            return
        mem = self.res["memory_gb"] or GB_PER_EXTRACTION * len(todo)
        par = max(1, min(len(todo), self.res["cpus"] // 4 or 1, int(mem // GB_PER_EXTRACTION) or 1))
        thr = max(1, self.res["cpus"] // par)
        args = [{"cfg": self.cfg.to_dict(), "profile": self.cfg.profile, "cohort": c, "src": self.disc[c]["path"],
                 "workdir": str(self.ws.root), "threads": thr} for c in todo]
        LOG.info("extract: %d cohort(s) in %d parallel process(es) × %d threads", len(todo), par, thr)
        results = []
        if par == 1:
            results = [_extract_entry(a) for a in args]
        else:
            with _child_threads(thr), cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(_extract_entry, a): a["cohort"] for a in args}
                for fu in cf.as_completed(futs):
                    try:
                        results.append(fu.result())
                    except Exception as exc:  # worker died (e.g. OOM)
                        results.append({"cohort": futs[fu], "error": f"process terminated ({exc!r})"})
        stops, errors = [], []
        for r in results:
            c = r["cohort"]
            if r.get("stop"):
                stops.append(r["stop"])
            elif r.get("error"):
                errors.append(f"{c}: {r['error']}")
            else:
                for f in r["flow"]:
                    LOG.info("  [%s] %-62s %8s stays", c, f["step"], f"{f['n_stays']:,}")
                LOG.info("extract[%s]: %d index stays, %s inputs_long rows, %s outcomes", c, r["n_index"],
                         f"{r['n_inputs']:,}", f"{r['n_outcomes']:,}")
                cm = self.ws.common(c)
                self.mark("extract", c, outputs=[cm / "stays_index.parquet", cm / "inputs_long.parquet",
                                                 cm / "outcomes.parquet"], n_index=r["n_index"], n_inputs=r["n_inputs"])
        if errors:
            raise RuntimeError("Extraction failed (relaunching only redoes the failed cohorts):\n" + "\n".join(errors))
        if stops:
            raise Stop("\n".join(stops))
        self.check_disk("extract")

    def stage_targets(self) -> None:
        from perisim.targets.build import build_targets

        for c in self.available_cohorts():
            if self.done("targets", c) and not self.force:
                continue
            cm, dv = self.ws.common(c), self.ws.derived(c)
            stays = pl.read_parquet(cm / "stays_index.parquet")
            inp = pl.read_parquet(cm / "inputs_long.parquet")
            outc = pl.read_parquet(cm / "outcomes.parquet")
            tg, car, diag = build_targets(stays, inp, outc, self.cfg.targets, c,
                                          float(self.cfg.experiment["time"]["cutoff_margin_hours"]),
                                          float(self.cfg.experiment["features"]["support_gap_hours"]))
            write_parquet(tg, dv / "targets.parquet")
            write_parquet(car, dv / "carried.parquet")
            LOG.info("targets[%s]: T1 severe %.2f %%", c, 100 * float(tg["t1_severe"].mean() or 0))
            self.mark("targets", c, outputs=[dv / "targets.parquet", dv / "carried.parquet"], **diag)

    def stage_features(self) -> None:
        from perisim.features.build import build_features

        for c in self.available_cohorts():
            if self.done("features", c) and not self.force:
                continue
            cm, dv = self.ws.common(c), self.ws.derived(c)
            stays = pl.read_parquet(cm / "stays_index.parquet")
            inp = pl.read_parquet(cm / "inputs_long.parquet")
            X, specs, prov = build_features(stays, inp, self.cfg, c)
            write_parquet(X, dv / "features.parquet")
            write_json(dv / "feature_specs.json", [s.to_dict() for s in specs])
            write_parquet(prov, dv / "feature_provenance.parquet")
            avail = {s.name: float(1 - X[s.name].is_null().mean()) for s in specs}
            write_json(dv / "feature_availability.json", avail)
            LOG.info("features[%s]: %d stays × %d variables", c, X.height, len(specs))
            self.mark("features", c, outputs=[dv / "features.parquet", dv / "feature_specs.json"],
                      n_features=len(specs))

    def stage_leakage(self) -> None:
        """§4.4.1: automatic, row-by-row test that no input is later than t0 − margin."""
        from perisim.features.build import check_no_leakage

        if self.done("leakage") and not self.force:
            return
        for c in self.available_cohorts():
            prov = pl.read_parquet(self.ws.derived(c) / "feature_provenance.parquet")
            bad = check_no_leakage(prov)
            if bad.height:
                raise Stop(f"TEMPORAL LEAKAGE in {c}: {bad.height} stays with inputs after the cutoff. "
                           f"Check src/perisim/features/build.py")
            LOG.info("leakage[%s]: %d rows checked, 0 violations", c, prov.height)
            from perisim.features.build import metamorphic_leak_check

            cm, dv = self.ws.common(c), self.ws.derived(c)
            diff = metamorphic_leak_check(pl.read_parquet(cm / "stays_index.parquet"),
                                          pl.read_parquet(cm / "inputs_long.parquet"), self.cfg, c,
                                          X=pl.read_parquet(dv / "features.parquet"))
            if diff:
                raise Stop(f"TEMPORAL LEAKAGE in {c}: these inputs change when the data after the cutoff are "
                           f"removed: {diff[:20]}. Check src/perisim/features/build.py")
            LOG.info("leakage[%s]: metamorphic check passed (inputs identical without post-cutoff data)", c)
        tests = REPO_ROOT / "tests"
        if tests.is_dir():
            env = {**os.environ, "PERISIM_TEST_WORKDIR": str(self.ws.root), "PERISIM_TEST_VARIANT": self.variant}
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tests / "test_no_leakage.py"),
                                str(tests / "test_vdm_in_fold.py")], env=env, capture_output=True, text=True)
            LOG.info("anti-leakage pytest:\n%s", r.stdout[-2000:])
            if r.returncode != 0:
                raise Stop(f"Anti-leakage tests failing (M2):\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}")
        self.mark("leakage")

    def stage_audit(self) -> None:
        from perisim.evaluation.audit import completeness
        from perisim.features.build import FeatureSpec

        issues_all = []
        for c in self.available_cohorts():
            cm, dv = self.ws.common(c), self.ws.derived(c)
            rep_path = self.ws.reports / f"completeness_{c}.html"
            if not (self.done("audit") and rep_path.exists()) or self.force:
                rep, html_ = completeness(c, read_json(cm / "flow.json"), pl.read_parquet(cm / "stays_index.parquet"),
                                          pl.read_parquet(cm / "inputs_long.parquet"), pl.read_parquet(dv / "targets.parquet"),
                                          pl.read_parquet(dv / "features.parquet"),
                                          [FeatureSpec(**d) for d in read_json(dv / "feature_specs.json")], self.cfg)
                rep_path.write_text(html_, encoding="utf-8")
                write_json(self.ws.artifacts() / f"completeness_{c}.json", rep)
            else:
                rep = read_json(self.ws.artifacts() / f"completeness_{c}.json")
            issues_all += rep["issues"]
        noted = self.ws.artifacts() / "audit_issues_noted.json"
        if issues_all and (read_json(noted) or {}).get("hash") != stable_hash(issues_all):
            self.note("OPEN_QUESTIONS", "Completeness audit (§7)",
                      "\n".join(f"- {i}" for i in issues_all) + "\n\nSee reports/completeness_<cohort>.html.")
            write_json(noted, {"hash": stable_hash(issues_all)})
        gates = self.cfg.experiment["gates"]
        ack_file = self.ws.state / "audit_ack.json"
        h = stable_hash(issues_all)
        if issues_all and _truthy(gates.get("audit_acknowledged")):
            write_json(ack_file, {"issues_hash": h, "issues": issues_all, "at": now_iso()})
        acked = (read_json(ack_file) or {}).get("issues_hash") == h   # the SAME issues were acknowledged before
        if issues_all and gates.get("audit_stop_on_issues", True) and not acked:
            raise Stop("§7 completeness audit raised issues:\n  - " + "\n  - ".join(issues_all) +
                       f"\nReports: {self.ws.reports}/completeness_*.html\n"
                       "After reviewing them (and recording the decision in the repo's docs/OPEN_QUESTIONS.md / "
                       "DECISIONS.md), continue with:  PERISIM_ACK_AUDIT=true bash run_pod.sh")
        self.mark("audit", digest=stable_hash(issues_all), issues=issues_all)

    def stage_harmonize(self) -> None:
        """Input sets: core (intersection of the 3 cohorts), extended (INSPIRE ∩ MIMIC), full (INSPIRE)."""
        fcfg = self.cfg.experiment["features"]
        thr = 1 - float(fcfg["max_missing_for_availability"])
        avail = {c: read_json(self.ws.derived(c) / "feature_availability.json") for c in self.available_cohorts()}
        missing = [c for c in COHORTS if c not in avail]
        if missing and not self.cfg.experiment["cohorts"]["allow_missing_cohorts"]:
            raise Stop(f"Cohorts not available: {missing}. The input intersection (§4.1) requires all three.\n"
                       "If their mount is pending (IRB/DRB), wait for it or, to proceed with INSPIRE only, set "
                       "experiment.cohorts.allow_missing_cohorts: true (it will be recorded).")
        dev = self.cfg.experiment["cohorts"]["development"]
        ok = {c: {f for f, a in v.items() if a >= thr} for c, v in avail.items()}
        sets = {"full": sorted(ok[dev])}
        sets["extended"] = sorted(ok[dev] & ok.get("mimic", ok[dev]))
        sets["core"] = sorted(set.intersection(*[ok[c] for c in ok]))
        # availability fractions would reveal small missing counts: they stay in derived/ (never exported)
        write_json(self.ws.artifacts() / "feature_sets.json", {**sets, "missing_cohorts": missing})
        chosen = fcfg["feature_set"]
        LOG.info("harmonize: core=%d, extended=%d, full=%d → using '%s'", len(sets["core"]), len(sets["extended"]),
                 len(sets["full"]), chosen)
        if missing:
            self.note("DECISIONS", "Modelling without all cohorts", f"Missing cohorts: {missing}.")
        if len(sets[chosen]) < 5:
            raise Stop(f"The '{chosen}' set has only {len(sets[chosen])} variables: review the harmonization.")
        self.mark("harmonize", digest=stable_hash(sets))

    # ------------------------------------------------------------------ modelling
    def feature_names(self) -> list[str]:
        fs = read_json(self.ws.for_variant("main").artifacts() / "feature_sets.json") or \
            read_json(self.ws.artifacts() / "feature_sets.json")
        return fs[self.cfg.experiment["features"]["feature_set"]]

    def d7(self) -> dict:
        return read_json(self.ws.shared_artifacts() / "d7.json", {}) or {}

    def stage_pilot(self) -> None:
        """M3: 1 outer fold, T1; timings, memory, number of models until stopping and cost extrapolation.

        Runs as ONE job with the per-job thread count on purpose, so that its timings are representative of
        the CV jobs it extrapolates to.
        """
        from perisim.experiment import load_cohort_data, outer_folds, run_fold
        from perisim.importance.loop import min_delta_from_pilot

        if self.variant != "main":
            return
        if self.done("pilot") and not self.force:
            return
        pc = self.cfg.experiment["pilot"]
        dev = self.cfg.experiment["cohorts"]["development"]
        data = load_cohort_data(self.ws.common(dev), self.ws.derived(dev), self.feature_names())
        folds = outer_folds(data, int(self.cfg.experiment["cv"]["outer_folds"]), self.cfg.seed)
        f = int(pc["outer_fold"])
        set_threads(self.threads())
        st = {"n_max": int(pc["n_models"]), "min_delta": None}
        t0 = time.perf_counter()
        res, _ = run_fold(self.cfg, data, pc["target"], np.where(folds != f)[0], np.where(folds == f)[0],
                          seed=self.cfg.seed, threads=self.threads(),
                          baselines=self.cfg.experiment["baselines"] if pc.get("include_baselines") else [],
                          stopping_override=st, early_stop=False, select_hparams=False)
        wall = time.perf_counter() - t0
        loop = res["importance_loop"]
        stp = self.cfg.experiment["importance"]["stopping"]
        md_cfg = stp.get("min_delta")
        md = min_delta_from_pilot(loop["delta_smooth"], int(stp["n_min"]),
                                  float(pc["min_delta_rule"]["fraction_of_delta_at_nmin"])) \
            if md_cfg in (None, "auto") else float(md_cfg)
        n_stop = _simulate_stop(loop["delta_smooth"], md, int(stp["P"]), int(stp["n_min"]), int(stp["n_max"]))
        acc = [m for m in loop["models"] if m.get("accepted")]
        fam_cost: dict[str, list[float]] = {}
        for m in acc:
            fam_cost.setdefault(m["family"], []).append(m["fit_seconds"] + m.get("sage_seconds", 0) + m.get("nca_seconds", 0))
        fam_med = {k: float(np.median(v)) for k, v in fam_cost.items()}
        exclude = []
        others = [v for k, v in fam_med.items() if k != "mlp"]
        if "mlp" in fam_med and others and fam_med["mlp"] > float(pc["mlp_cost_ratio_exclude"]) * float(np.median(others)):
            exclude.append("mlp")
        per_model = float(np.median([m["fit_seconds"] + m.get("sage_seconds", 0) + m.get("nca_seconds", 0) for m in acc])) if acc else 0
        t = res["timings"]
        n_targets = len(self.cfg.experiment["targets_to_model"])
        n_jobs = int(self.cfg.experiment["cv"]["outer_folds"]) * n_targets
        lam_cost = t.get("nca_final", 0) * len(self.cfg.experiment["metric"]["lambda_grid"]) * \
            int(self.cfg.experiment["cv"]["inner_folds"]) * 0.7
        per_job = per_model * n_stop * (1 + (loop["n_attempts"] - loop["n_accepted"]) / max(loop["n_accepted"], 1) * 0.3) \
            + lam_cost + 3 * t.get("nca_final", 0) + t.get("baselines", 0) + 120
        par = self.jobs()
        sens = self.cfg.experiment["sensitivity"]
        n_var = len(sens["variants"]) if sens.get("enabled") else 0
        waves = lambda n: int(np.ceil(n / par))  # noqa: E731  (one shared queue: idle cores only in the last wave)
        d7 = {"min_delta": md, "tol_delta": float(stp["tol_delta"]), "W": int(stp["W"]), "P": int(stp["P"]),
              "n_min": int(stp["n_min"]), "n_max": int(stp["n_max"]), "epsilon": float(self.cfg.experiment["importance"]["epsilon"]),
              "exclude_families": exclude, "rule": pc["min_delta_rule"], "fixed_at": now_iso()}
        pilot = {"wall_seconds": wall, "rss_gb_peak": rss_gb(), "n_models_run": loop["n_accepted"],
                 "n_attempts": loop["n_attempts"], "acceptance_rate": loop["n_accepted"] / max(loop["n_attempts"], 1),
                 "simulated_models_until_stop": n_stop, "per_model_seconds_median": per_model,
                 "family_cost_median_s": fam_med, "timings": t, "delta": loop["delta"],
                 "delta_smooth": loop["delta_smooth"], "d7": d7, "resources_plan": self.res,
                 "extrapolation": {"jobs": n_jobs, "seconds_per_job": per_job, "parallel": par,
                                   "hours_total_cv": waves(n_jobs) * per_job / 3600,
                                   "hours_sensitivity": waves(n_jobs * n_var) * per_job / 3600,
                                   "hours_external_final_models": per_job / 3600 * waves(n_targets)},
                 "fold": f, "target": pc["target"]}
        write_json(self.ws.shared_artifacts() / "pilot.json", pilot)
        write_json(self.ws.shared_artifacts() / "d7.json", d7)
        _pilot_html(self.ws.reports / "pilot.html", pilot)
        self.note("DECISIONS", "D7 fixed with the pilot (pre-declared rule)",
                  f"min_delta={md:.6g} (rule {pc['min_delta_rule']}), tol_delta={d7['tol_delta']}, W={d7['W']}, "
                  f"P={d7['P']}, n_min={d7['n_min']}, n_max={d7['n_max']}, ε={d7['epsilon']}. "
                  f"Families excluded for cost: {exclude or 'none'}. "
                  f"Models until stopping (simulated): {n_stop}. Estimated CV cost: {pilot['extrapolation']['hours_total_cv']:.1f} h.")
        LOG.info("pilot: %.1f min; simulated stop at %d models; estimated CV %.1f h + sensitivity %.1f h", wall / 60,
                 n_stop, pilot["extrapolation"]["hours_total_cv"], pilot["extrapolation"]["hours_sensitivity"])
        self.mark("pilot", digest=stable_hash({k: v for k, v in d7.items() if k != "fixed_at"}))

    def cv_jobs(self, tag: str = "cv", split: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict]:
        """Self-contained (target × fold) jobs still pending for this pipeline/variant."""
        from perisim.experiment import load_cohort_data, outer_folds

        d7 = self.d7()
        if not d7:
            raise Stop("artifacts/_shared/d7.json is missing: run the pilot stage (M3) first.")
        dev = self.cfg.experiment["cohorts"]["development"]
        names = self.feature_names()
        if split is None:
            data = load_cohort_data(self.ws.common(dev), self.ws.derived(dev), names)
            folds = outer_folds(data, int(self.cfg.experiment["cv"]["outer_folds"]), self.cfg.seed)
            write_parquet(pl.DataFrame({"stay_id": data.ft.ids.astype(str), "subject_id": data.ft.subjects.astype(str),
                                        "outer_fold": folds}), self.ws.artifacts() / "folds.parquet")
            splits = [(f, np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(set(folds))]
        else:
            splits = [(0, split[0], split[1])]
        ckey = self.key("cv")
        common = {"cfg": self.cfg.to_dict(), "profile": self.cfg.profile, "variant": self.variant, "d7": d7,
                  "threads": self.threads(), "features": names, "common_dir": str(self.ws.common(dev)),
                  "derived_dir": str(self.ws.derived(dev)), "key": ckey}
        jobs = []
        for t in self.cfg.experiment["targets_to_model"]:
            for f, tr, te in splits:
                out = self.ws.artifacts(tag, t) / f"fold{f}.json"
                prev = read_json(out)
                if prev and prev.get("_key") == ckey and not self.force:
                    continue
                jobs.append({**common, "target": t, "fold": int(f), "train": tr, "test": te, "out": str(out),
                             "pred": str(self.ws.predictions(tag, t) / f"fold{f}")})
        return jobs

    def stage_cv(self, tag: str = "cv", split: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        if tag == "cv" and self.done("cv") and not self.force:
            return
        jobs = self.cv_jobs(tag, split)
        if jobs:
            self.run_jobs(jobs, tag)
        else:
            LOG.info("%s: all jobs already done", tag)
        if tag == "cv":
            self._real_data_tests(["test_patient_split.py", "test_vdm_in_fold.py"], "after the CV (§6.1, §5.2)")
            self.mark("cv")

    def _real_data_tests(self, files: list[str], when: str) -> None:
        tests = REPO_ROOT / "tests"
        if not tests.is_dir():
            return
        env = {**os.environ, "PERISIM_TEST_WORKDIR": str(self.ws.root), "PERISIM_TEST_VARIANT": self.variant}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *[str(tests / f) for f in files]], env=env,
                           capture_output=True, text=True)
        LOG.info("real-data tests %s:\n%s", when, r.stdout[-1500:])
        if r.returncode != 0:
            raise Stop(f"Real-data tests failing {when}:\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}")

    def run_jobs(self, jobs: list[dict], label: str) -> None:
        par = max(1, min(len(jobs), self.jobs()))
        LOG.info("%s: %d jobs (target × fold) in %d processes × %d threads", label, len(jobs), par, self.threads())
        failures = []
        if par == 1:
            for j in jobs:
                r = _job_entry(j)
                if r is not True:
                    failures.append(r)
        else:
            with _child_threads(self.threads()), \
                    cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(_job_entry, j): j for j in jobs}
                for fu in cf.as_completed(futs):
                    j = futs[fu]
                    try:
                        r = fu.result()
                    except Exception as exc:  # worker died (e.g. OOM)
                        r = f"[{j['variant']}] {j['target']} fold {j['fold']}: process terminated ({exc!r})"
                    if r is not True:
                        failures.append(r)
                    else:
                        LOG.info("  ✓ [%s] %s fold %d", j["variant"], j["target"], j["fold"])
        if failures:
            raise RuntimeError("Failed jobs (relaunching resumes only these):\n" + "\n".join(map(str, failures)))

    def stage_temporal(self) -> None:
        from perisim.experiment import load_cohort_data

        tc = self.cfg.experiment["cv"]["temporal"]
        if self.variant != "main" or str(tc.get("enabled")).lower() in ("false", "0", "no"):
            return
        dev = self.cfg.experiment["cohorts"]["development"]
        data = load_cohort_data(self.ws.common(dev), self.ws.derived(dev), self.feature_names())
        yr = data.year
        if np.isfinite(yr).mean() < 0.95:
            LOG.info("temporal: no reliable year in %s (%.0f %% available) → skipped (§6.1)", dev, 100 * np.isfinite(yr).mean())
            self.note("DECISIONS", "Temporal validation skipped", "The relative dates do not allow reconstructing the year.")
            return
        a, b = tc["train_years"]
        c, d = tc["test_years"]
        tr = np.where((yr >= a) & (yr <= b))[0]
        te = np.where((yr >= c) & (yr <= d))[0]
        if tr.size < 100 or te.size < 50:
            LOG.info("temporal: insufficient sizes (train=%d, test=%d)", tr.size, te.size)
            return
        self.stage_cv(tag="temporal", split=(tr, te))

    def stage_report(self) -> None:
        from perisim.evaluation.report import aggregate_cv, render_results

        if self.done("report") and not self.force:
            return
        n_outer = int(self.cfg.experiment["cv"]["outer_folds"])
        res = aggregate_cv(self.ws.artifacts(), self.ws.predictions(), self.cfg.experiment["targets_to_model"], self.cfg,
                           expected_key=self.key("cv"), expected_folds=set(range(n_outer)))
        write_json(self.ws.artifacts() / "results.json", res)
        (self.ws.reports / f"results_{self.variant}.html").write_text(
            render_results(res, self.cfg, f"PERISIM results — internal validation ({self.variant})"), encoding="utf-8")
        if any(self.ws.artifacts("temporal").rglob("fold*.json")):
            rt = aggregate_cv(self.ws.artifacts(), self.ws.predictions(), self.cfg.experiment["targets_to_model"],
                              self.cfg, tag="temporal", expected_key=self.key("cv"), expected_folds={0})
            write_json(self.ws.artifacts() / "results_temporal.json", rt)
            (self.ws.reports / "results_temporal.html").write_text(
                render_results(rt, self.cfg, "PERISIM results — temporal validation (2011–2018 → 2019–2020)"),
                encoding="utf-8")
        for t, r in res["targets"].items():
            lams = [f["lambda"] for f in r["folds"]]
            self.note("DECISIONS", f"Selected λ ({t})", f"λ per fold: {lams} (a result in itself, §5.4)")
            for f in r["folds"]:
                if f["plateau_above_tol"]:
                    self.note("DECISIONS", f"High Δ̃ plateau ({t}, fold {f['fold']})",
                              f"Metric stabilized with high residual noise (Δ̃={f['plateau']:.4f} > tol): "
                              "real heterogeneity between models (§5.3.2).")
                if f["stop_reason"] != "patience":
                    self.note("DECISIONS", f"No convergence ({t}, fold {f['fold']})",
                              f"The loop ended by {f['stop_reason']} with {f['n_accepted']} models.")
        self.mark("report")

    def stage_sensitivity(self) -> None:
        """Sensitivity variants: cheap preparation per variant, then ONE shared queue for all CV jobs."""
        sc = self.cfg.experiment.get("sensitivity", {})
        if self.variant != "main" or not sc.get("enabled"):
            return
        subs: list[Pipeline] = []
        for v in sc.get("variants", []):
            vcfg = self.cfg.with_overrides(v.get("overrides", {}), variant=v["name"])
            sub = Pipeline(vcfg, self.ws.for_variant(v["name"]), force=self.force)
            sub.disc = self.disc
            LOG.info("── sensitivity: preparing %s", v["name"])
            rc = sub.run(VARIANT_PREP_STAGES)
            if rc == 3:
                raise Stop(f"Variant {v['name']} stopped (see log)")
            if rc != 0:
                raise RuntimeError(f"Variant {v['name']} failed")
            subs.append(sub)
        jobs = [j for sub in subs if self.force or not sub.done("cv") for j in sub.cv_jobs()]
        if jobs:
            self.run_jobs(jobs, f"sensitivity ({len(subs)} variants)")
        for sub in subs:
            sub.mark("cv")
            if sub.run(["report"]) != 0:
                raise RuntimeError(f"Report of variant {sub.variant} failed")

    def stage_external(self) -> None:
        from perisim.external import run_external

        if self.variant != "main":
            return
        fr = read_json(self.ws.state / "freeze.json")
        if self.cfg.experiment["gates"]["external_requires_freeze"]:
            if not fr:
                LOG.warning("external: WAITING for the freeze (§6.3). Review reports/results_main.html and, once the "
                            "pipeline is final:  bash run_pod.sh freeze --registration <OSF_ID>  then relaunch.")
                return
            if fr.get("code_hash") != code_hash() or fr.get("config_hash") != self.cfg.scientific_hash():
                raise Stop("Code or configuration changed since the freeze (§6.3: iterating is forbidden). "
                           "Restore the frozen version (git checkout <tag>) or freeze again if no external results "
                           "have been looked at yet.")
        if self.done("external") and not self.force:
            return
        run_external(self)
        self.mark("external")

    def stage_export(self) -> None:
        from perisim.evaluation.export import build_export

        build_export(self.ws, self.cfg)


def _files_digest(paths: list[Path]) -> str:
    """Content hash of a stage's output files (chunked: multi-GB Parquet files are fine)."""
    import hashlib

    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).name.encode())
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        except FileNotFoundError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def _truthy(v) -> bool:
    """Strict boolean for gate flags: only true / "true" / "1" / "yes" count (PERISIM_ACK_AUDIT=n is False)."""
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def _simulate_stop(smooth: list[float], min_delta: float, P: int, n_min: int, n_max: int) -> int:
    best, since = np.inf, 0
    for i, s in enumerate(smooth, start=1):
        if s < best - min_delta:
            best, since = s, 0
        else:
            since += 1
        if i >= n_min and since >= P:
            return i
    return min(n_max, max(len(smooth), n_min)) if len(smooth) < n_max else n_max


def _pilot_html(path: Path, pilot: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from perisim.evaluation.html import box, fig_img, page, table

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(range(1, len(pilot["delta"]) + 1), pilot["delta"], lw=0.8, alpha=0.5, label="Δ")
    ax.plot(range(1, len(pilot["delta_smooth"]) + 1), pilot["delta_smooth"], lw=1.8, label="Δ̃")
    ax.axvline(pilot["simulated_models_until_stop"], color="k", ls="--", lw=0.8, label="simulated stop")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("accepted models")
    ex = pilot["extrapolation"]
    rp = pilot["resources_plan"]
    sec = [box(f"M3 pilot (fold {pilot['fold']}, {pilot['target']}). D7 fixed by a pre-declared rule: "
               f"min_delta = {pilot['d7']['min_delta']:.4g}; families excluded: {pilot['d7']['exclude_families'] or 'none'}. "
               f"Resources: {rp['cpus']} CPUs, {rp['memory_gb']} GB → {rp['jobs']} jobs × {rp['threads_per_job']} threads."),
           fig_img(fig),
           table(["Measure", "Value"], [["Pilot duration (min)", pilot["wall_seconds"] / 60],
                                        ["Memory (GB, process)", pilot["rss_gb_peak"]],
                                        ["Models run / attempts", f"{pilot['n_models_run']} / {pilot['n_attempts']}"],
                                        ["Models until stopping (simulated)", pilot["simulated_models_until_stop"]],
                                        ["Seconds per model (median)", pilot["per_model_seconds_median"]],
                                        ["CV jobs", ex["jobs"]], ["Estimated CV hours", ex["hours_total_cv"]],
                                        ["Estimated sensitivity hours", ex["hours_sensitivity"]],
                                        ["Estimated external final-model hours", ex["hours_external_final_models"]]], nd=2),
           table(["Family", "Median cost per model (s)"], [[k, v] for k, v in pilot["family_cost_median_s"].items()], nd=1),
           table(["Step", "seconds"], [[k, v] for k, v in pilot["timings"].items()], nd=1)]
    path.write_text(page("SAGE and metric pilot (M3)", sec, "PERISIM · aggregates only"), encoding="utf-8")


# --------------------------------------------------------------------------- child-process entry points
def _extract_entry(args: dict) -> dict:
    """Extraction of one cohort in its own process. Returns counts, a gate message or an error."""
    c = args["cohort"]
    try:
        from perisim.cohort.select import select_index_stays
        from perisim.data import get_extractor
        from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA, validate
        from perisim.utils import setup_logging

        setup_logging()
        apply_library_limits(int(args["threads"]))
        cfg = Config(**args["cfg"], profile=args["profile"], variant="main")
        ws = Workspace(args["workdir"])
        out = ws.common(c)
        ex = get_extractor(c)(cfg, Path(args["src"]), ws.cache(c))
        with timed(f"extract[{c}] stays"):
            cand = ex.extract_stays()
        errs = validate(cand, STAYS_SCHEMA, "stays", unique="stay_id")
        if errs:
            return {"cohort": c, "error": "; ".join(errs)}
        idx, flow = select_index_stays(cand, cfg.cohorts, c)
        write_parquet(cand, out / "stays_candidates.parquet")
        write_parquet(idx, out / "stays_index.parquet")
        write_json(out / "flow.json", flow)
        if idx.height == 0:
            return {"cohort": c, "stop": f"{c}: no index stays after the criteria. Check reports/schema_{c}.md "
                                         f"and config/cohorts.yaml"}
        with timed(f"extract[{c}] events"):
            inp, outc = ex.extract_events(idx)
        errs = validate(inp, INPUTS_SCHEMA, "inputs_long") + validate(outc, OUTCOMES_SCHEMA, "outcomes")
        if errs:
            return {"cohort": c, "error": "; ".join(errs)}
        write_parquet(inp, out / "inputs_long.parquet")
        write_parquet(outc, out / "outcomes.parquet")
        return {"cohort": c, "flow": flow, "n_index": idx.height, "n_inputs": inp.height, "n_outcomes": outc.height}
    except Exception:
        return {"cohort": c, "error": traceback.format_exc()}


def _job_entry(args: dict):
    """One (target × fold) job in its own process. Returns True or an error message."""
    try:
        from perisim.experiment import fold_seed, load_cohort_data, run_fold
        from perisim.utils import setup_logging

        setup_logging()
        set_threads(int(args["threads"]))
        apply_library_limits(int(args["threads"]))
        cfg = Config(**args["cfg"], profile=args["profile"], variant=args["variant"])
        data = load_cohort_data(Path(args["common_dir"]), Path(args["derived_dir"]), args["features"])
        d7 = args["d7"]
        st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
        res, out = run_fold(cfg, data, args["target"], np.asarray(args["train"]), np.asarray(args["test"]),
                            seed=fold_seed(cfg, int(args["fold"])), threads=int(args["threads"]),
                            baselines=cfg.experiment["baselines"], stopping_override=st,
                            exclude_families=set(d7.get("exclude_families", [])), early_stop=True)
        pred = Path(args["pred"])
        pred.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(out["pred"], pred.with_name(pred.name + "_pred.parquet"))
        write_parquet(out["ood"], pred.with_name(pred.name + "_ood.parquet"))
        write_parquet(out["p4"] if out["p4"].height else pl.DataFrame({"stay_id": []}), pred.with_name(pred.name + "_p4.parquet"))
        res["_key"] = args["key"]
        res["fold"] = int(args["fold"])
        write_json(Path(args["out"]), res)
        return True
    except Exception:
        return f"[{args.get('variant')}] {args.get('target')} fold {args.get('fold')}:\n{traceback.format_exc()}"
