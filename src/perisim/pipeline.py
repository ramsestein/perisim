"""Orquestación de etapas, idempotente y reanudable (un OOM o un reinicio del pod no obliga a empezar de cero).

Cada etapa tiene una clave = hash(configuración relevante + código relevante + claves de las etapas previas).
Si la clave coincide con la del marcador en state/, la etapa se salta. Las puertas de gobernanza paran el
pipeline con instrucciones claras:
  - §7  auditoría de completitud (prevalencia < 0,5 % o ausencia > 30 %) → OPEN_QUESTIONS;
  - D7  parámetros de parada fijados con el piloto ANTES del pipeline completo (regla declarada);
  - §6.3 la validación externa exige congelación (freeze) con hash de código y configuración.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from perisim.config import REPO_ROOT, Config, code_hash, stable_hash
from perisim.utils import (LOG, append_note, dir_size_gb, free_disk_gb, now_iso, read_json, rss_gb, set_threads,
                           timed, write_json, write_parquet)
from perisim.workspace import Workspace

COHORTS = ("inspire", "mimic", "eicu")
MAIN_STAGES = ["env", "discover", "extract", "targets", "features", "leakage", "audit", "harmonize", "pilot", "cv",
               "temporal", "report", "sensitivity", "external", "export"]
VARIANT_STAGES = ["targets", "features", "leakage", "cv", "report"]


class Stop(Exception):
    """Puerta de gobernanza cerrada: el pipeline se detiene con un mensaje accionable."""


@dataclass
class StageDef:
    config: tuple[str, ...]
    code: tuple[str, ...]
    upstream: tuple[str, ...]
    per_cohort: bool = False


STAGE_DEFS: dict[str, StageDef] = {
    "env": StageDef((), ("utils.py",), ()),
    "discover": StageDef(("harmonization.sources", "experiment.paths", "experiment.resources"), ("data",), ()),
    "extract": StageDef(("harmonization", "cohorts", "experiment.features.preop_lab_lookback_days", "targets.window"),
                        ("data", "cohort"), ("discover",), True),
    "targets": StageDef(("targets", "experiment.time", "experiment.features.support_gap_hours"),
                        ("targets", "timeline.py"), ("extract",), True),
    "features": StageDef(("experiment.features", "experiment.time", "harmonization.medications",
                          "harmonization.preop_labs"), ("features", "timeline.py", "cohort"), ("extract",), True),
    "leakage": StageDef((), ("features",), ("features",)),
    "audit": StageDef(("experiment.evaluation", "targets.t1"), ("evaluation/audit.py", "evaluation/html.py"),
                      ("targets", "features")),
    "harmonize": StageDef(("experiment.features", "experiment.cohorts"), (), ("features", "audit")),
    "pilot": StageDef(("experiment.importance", "experiment.metric", "experiment.knn", "experiment.pilot",
                       "experiment.cv", "model_space", "experiment.seed"),
                      ("importance", "metric", "knn", "experiment.py"), ("harmonize", "targets")),
    "cv": StageDef(("experiment.importance", "experiment.metric", "experiment.knn", "experiment.cv",
                    "experiment.baselines", "experiment.targets_to_model", "model_space", "targets.t2", "targets.t3",
                    "experiment.seed"),
                   ("importance", "metric", "knn", "experiment.py", "baselines"), ("pilot", "harmonize", "targets")),
    "temporal": StageDef(("experiment.cv.temporal",), (), ("cv",)),
    "report": StageDef(("experiment.evaluation", "experiment.knn"), ("evaluation",), ("cv",)),
    "sensitivity": StageDef(("experiment.sensitivity",), (), ("cv",)),
    "external": StageDef(("experiment",), ("experiment.py", "evaluation"), ("cv", "harmonize")),
    "export": StageDef((), ("evaluation",), ()),
}


def _get(d: dict, dotted: str):
    node = d
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


class Pipeline:
    def __init__(self, cfg: Config, ws: Workspace | None = None, force: bool = False):
        self.cfg = cfg
        self.ws = ws or Workspace(cfg.workdir, cfg.variant)
        self.force = force
        self.variant = cfg.variant
        self._keys: dict[str, str] = {}
        self.disc = read_json(self.ws.shared_artifacts() / "discovery.json", {}) or {}

    # ------------------------------------------------------------------ claves y marcadores
    def key(self, stage: str, cohort: str | None = None) -> str:
        cache_k = f"{stage}:{cohort}"
        if cache_k in self._keys:
            return self._keys[cache_k]
        sd = STAGE_DEFS[stage]
        cfgd = self.cfg.to_dict()
        payload = {"stage": stage, "cohort": cohort, "variant": self.variant if stage not in ("discover", "extract") else "_",
                   "config": {p: _get(cfgd, p) for p in sd.config},
                   "code": code_hash(list(sd.code)) if sd.code else None,
                   "up": [self.key(u, cohort if STAGE_DEFS[u].per_cohort and cohort else None) for u in sd.upstream]}
        if stage in ("extract",) and cohort:
            payload["src"] = self.disc.get(cohort, {}).get("path")
        k = stable_hash(payload, 16)
        self._keys[cache_k] = k
        return k

    def _marker(self, stage: str, cohort: str | None) -> Path:
        variant = "_" if stage in ("env", "discover", "extract") else self.variant
        d = self.ws.state / variant
        d.mkdir(parents=True, exist_ok=True)
        return d / (f"{stage}__{cohort}.json" if cohort else f"{stage}.json")

    def done(self, stage: str, cohort: str | None = None) -> bool:
        m = read_json(self._marker(stage, cohort))
        return bool(m) and m.get("key") == self.key(stage, cohort) and m.get("status") == "done"

    def mark(self, stage: str, cohort: str | None = None, **info) -> None:
        write_json(self._marker(stage, cohort), {"key": self.key(stage, cohort), "status": "done",
                                                 "finished": now_iso(), **info})

    # ------------------------------------------------------------------ utilidades
    def available_cohorts(self) -> list[str]:
        return [c for c in COHORTS if self.disc.get(c, {}).get("path") and self.disc.get(c, {}).get("schema_ok", True)]

    def threads(self) -> int:
        return int(self.cfg.experiment["resources"]["threads_per_job"])

    def note(self, kind: str, title: str, body: str) -> None:
        append_note(self.ws.docs / f"{kind}.md", f"[{self.variant}] {title}", body)

    def check_disk(self, label: str) -> None:
        free = free_disk_gb(self.ws.root)
        LOG.info("Disco tras %s: %.1f GB libres; workdir %.2f GB", label, free, dir_size_gb(self.ws.root))
        if np.isfinite(free) and free < float(self.cfg.experiment["resources"]["min_free_disk_gb"]):
            raise Stop(f"Quedan {free:.1f} GB libres (< min_free_disk_gb). Libera espacio en {self.ws.root} "
                       f"(p. ej. borra {self.ws.root / 'cache'}) y relanza.")

    # ------------------------------------------------------------------ ejecución
    def run(self, stages: list[str] | None = None) -> int:
        stages = stages or MAIN_STAGES
        t0 = time.perf_counter()
        for st in stages:
            fn = getattr(self, f"stage_{st}")
            try:
                with timed(f"etapa {st} [{self.variant}]"):
                    fn()
            except Stop as exc:
                LOG.warning("⛔ Pipeline detenido en la etapa '%s':\n%s", st, exc)
                self.note("OPEN_QUESTIONS", f"Parada en {st}", str(exc))
                write_json(self.ws.state / "last_stop.json", {"stage": st, "message": str(exc), "at": now_iso()})
                return 3
            except Exception:
                LOG.error("✖ Error en la etapa '%s':\n%s", st, traceback.format_exc())
                write_json(self.ws.state / "last_error.json", {"stage": st, "trace": traceback.format_exc(), "at": now_iso()})
                return 1
        (self.ws.state / "last_stop.json").unlink(missing_ok=True)
        LOG.info("✔ Etapas completadas en %.1f min (variante %s)", (time.perf_counter() - t0) / 60, self.variant)
        return 0

    # ================================================================== etapas
    def stage_env(self) -> None:
        import importlib

        info = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                "code_hash": code_hash(), "config_hash": self.cfg.hash(), "profile": self.cfg.profile,
                "workdir": str(self.ws.root), "time": now_iso(),
                "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS")}}
        try:
            import psutil

            info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            pass
        info["free_disk_gb"] = round(free_disk_gb(self.ws.root), 1)
        vers = {}
        for mod in ("numpy", "polars", "pyarrow", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "torch", "faiss",
                    "yaml", "psutil", "matplotlib", "sage"):
            try:
                m = importlib.import_module(mod)
                vers[mod] = getattr(m, "__version__", "?")
            except Exception as exc:
                vers[mod] = f"NO DISPONIBLE ({type(exc).__name__})"
        info["packages"] = vers
        missing = [m for m in ("numpy", "polars", "pyarrow", "scipy", "sklearn", "lightgbm", "xgboost", "torch", "yaml")
                   if str(vers[m]).startswith("NO")]
        write_json(self.ws.shared_artifacts() / "env.json", info)
        if sys.version_info < (3, 11):
            raise Stop("Se requiere Python ≥ 3.11 (§8.1)")
        if missing:
            raise Stop(f"Faltan paquetes: {missing}. Ejecuta: bash run_pod.sh setup")
        probe = self.ws.state / ".probe"
        probe.write_text("ok")
        os.replace(probe, self.ws.state / ".probe2")
        (self.ws.state / ".probe2").unlink()
        LOG.info("Entorno: %s CPU, %s GB RAM, %s GB libres, faiss=%s", info["cpu_count"], info.get("ram_gb"),
                 info["free_disk_gb"], vers["faiss"])
        self.mark("env")

    def stage_discover(self) -> None:
        from perisim.data.discover import discover_cohorts, schema_markdown, schema_report

        if self.done("discover") and not self.force:
            LOG.info("discover: sin cambios (%s)", ", ".join(f"{c}={self.disc.get(c, {}).get('path')}" for c in COHORTS))
            return
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
                self.note("OPEN_QUESTIONS", f"Esquema de {c} no coincide",
                          f"Columnas no encontradas: {miss}. Corregir config/harmonization.yaml (ver reports/schema_{c}.md).")
        write_json(self.ws.shared_artifacts() / "discovery.json", disc)
        self.disc = disc
        self._keys.clear()
        dev = self.cfg.experiment["cohorts"]["development"]
        if not disc[dev].get("path"):
            raise Stop(f"No se encuentra la cohorte de desarrollo ({dev}) bajo {disc['_datasets_root']['path']}.\n"
                       f"Carpetas de primer nivel: {disc['_datasets_root']['top_level'][:40]}\n"
                       f"Define PERISIM_INSPIRE_DIR=/ruta/a/inspire y relanza.")
        if not disc[dev].get("schema_ok", True):
            raise Stop(f"El esquema de {dev} no coincide con config/harmonization.yaml: ver {self.ws.reports}/schema_{dev}.md")
        self.mark("discover")

    def stage_extract(self) -> None:
        from perisim.cohort.select import select_index_stays
        from perisim.data import get_extractor
        from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA, validate

        for c in self.available_cohorts():
            if self.done("extract", c) and not self.force:
                LOG.info("extract[%s]: sin cambios", c)
                continue
            out = self.ws.common(c)
            ex = get_extractor(c)(self.cfg, Path(self.disc[c]["path"]), self.ws.cache(c))
            with timed(f"extract[{c}] estancias"):
                cand = ex.extract_stays()
            errs = validate(cand, STAYS_SCHEMA, "stays", unique="stay_id")
            if errs:
                raise RuntimeError("; ".join(errs))
            idx, flow = select_index_stays(cand, self.cfg.cohorts, c)
            write_parquet(cand, out / "stays_candidates.parquet")
            write_parquet(idx, out / "stays_index.parquet")
            write_json(out / "flow.json", flow)
            for f in flow:
                LOG.info("  %-62s %8s estancias", f["step"], f"{f['n_stays']:,}")
            if idx.height == 0:
                raise Stop(f"{c}: ninguna estancia índice tras los criterios. Revisar reports/schema_{c}.md y config/cohorts.yaml")
            with timed(f"extract[{c}] eventos"):
                inp, outc = ex.extract_events(idx)
            for e in validate(inp, INPUTS_SCHEMA, "inputs_long") + validate(outc, OUTCOMES_SCHEMA, "outcomes"):
                raise RuntimeError(e)
            write_parquet(inp, out / "inputs_long.parquet")
            write_parquet(outc, out / "outcomes.parquet")
            LOG.info("extract[%s]: %d estancias índice, %s filas inputs_long, %s outcomes", c, idx.height,
                     f"{inp.height:,}", f"{outc.height:,}")
            self.mark("extract", c, n_index=idx.height, n_inputs=inp.height)
            self.check_disk(f"extract[{c}]")

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
            LOG.info("targets[%s]: T1 grave %.2f %%", c, 100 * float(tg["t1_grave"].mean() or 0))
            self.mark("targets", c, **diag)

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
            LOG.info("features[%s]: %d estancias × %d variables", c, X.height, len(specs))
            self.mark("features", c, n_features=len(specs))

    def stage_leakage(self) -> None:
        """§4.4.1: test automático, fila a fila, de que ningún input supera t0 − margen."""
        from perisim.features.build import check_no_leakage

        if self.done("leakage") and not self.force:
            return
        for c in self.available_cohorts():
            prov = pl.read_parquet(self.ws.derived(c) / "feature_provenance.parquet")
            bad = check_no_leakage(prov)
            if bad.height:
                raise Stop(f"FUGA TEMPORAL en {c}: {bad.height} estancias con inputs posteriores al corte. "
                           f"Revisar src/perisim/features/build.py")
            LOG.info("leakage[%s]: %d filas comprobadas, 0 violaciones", c, prov.height)
        tests = REPO_ROOT / "tests"
        if tests.is_dir():
            env = {**os.environ, "PERISIM_TEST_WORKDIR": str(self.ws.root), "PERISIM_TEST_VARIANT": self.variant}
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tests / "test_no_leakage.py"),
                                str(tests / "test_vdm_in_fold.py")], env=env, capture_output=True, text=True)
            LOG.info("pytest anti-fuga:\n%s", r.stdout[-2000:])
            if r.returncode != 0:
                raise Stop(f"Tests anti-fuga en rojo (M2):\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}")
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
            self.note("OPEN_QUESTIONS", "Auditoría de completitud (§7)",
                      "\n".join(f"- {i}" for i in issues_all) + "\n\nRevisar reports/completeness_<cohorte>.html.")
            write_json(noted, {"hash": stable_hash(issues_all)})
        gates = self.cfg.experiment["gates"]
        if issues_all and gates.get("audit_stop_on_issues", True) and not bool(gates.get("audit_acknowledged")):
            raise Stop("Auditoría §7 con incidencias:\n  - " + "\n  - ".join(issues_all) +
                       f"\nInformes: {self.ws.reports}/completeness_*.html\n"
                       "Tras revisarlas (y anotar la decisión en docs/OPEN_QUESTIONS.md / DECISIONS.md del repo), "
                       "continúa con:  PERISIM_ACK_AUDIT=true bash run_pod.sh")
        self.mark("audit", issues=issues_all)

    def stage_harmonize(self) -> None:
        """Conjuntos de inputs: core (intersección de las 3 cohortes), extended (INSPIRE ∩ MIMIC), full (INSPIRE)."""
        fcfg = self.cfg.experiment["features"]
        thr = 1 - float(fcfg["max_missing_for_availability"])
        avail = {c: read_json(self.ws.derived(c) / "feature_availability.json") for c in self.available_cohorts()}
        missing = [c for c in COHORTS if c not in avail]
        if missing and not self.cfg.experiment["cohorts"]["allow_missing_cohorts"]:
            raise Stop(f"Cohortes no disponibles: {missing}. La intersección de inputs (§4.1) exige las tres.\n"
                       "Si su montaje está pendiente (IRB/DRB), espera a que estén o, para avanzar solo con INSPIRE, "
                       "pon experiment.cohorts.allow_missing_cohorts: true (quedará registrado).")
        dev = self.cfg.experiment["cohorts"]["development"]
        ok = {c: {f for f, a in v.items() if a >= thr} for c, v in avail.items()}
        sets = {"full": sorted(ok[dev])}
        sets["extended"] = sorted(ok[dev] & ok.get("mimic", ok[dev]))
        sets["core"] = sorted(set.intersection(*[ok[c] for c in ok]))
        write_json(self.ws.artifacts() / "feature_sets.json", {**sets, "missing_cohorts": missing,
                                                                "availability": avail})
        chosen = fcfg["feature_set"]
        LOG.info("harmonize: core=%d, extended=%d, full=%d → se usa '%s'", len(sets["core"]), len(sets["extended"]),
                 len(sets["full"]), chosen)
        if missing:
            self.note("DECISIONS", "Modelado sin todas las cohortes", f"Cohortes ausentes: {missing}.")
        if len(sets[chosen]) < 5:
            raise Stop(f"El conjunto '{chosen}' solo tiene {len(sets[chosen])} variables: revisar la armonización.")
        self.mark("harmonize")

    # ------------------------------------------------------------------ modelado
    def feature_names(self) -> list[str]:
        fs = read_json(self.ws.for_variant("main").artifacts() / "feature_sets.json") or \
            read_json(self.ws.artifacts() / "feature_sets.json")
        return fs[self.cfg.experiment["features"]["feature_set"]]

    def d7(self) -> dict:
        return read_json(self.ws.shared_artifacts() / "d7.json", {}) or {}

    def stage_pilot(self) -> None:
        """M3: 1 fold externo, T1; tiempos, memoria, nº de modelos hasta la parada y extrapolación del coste."""
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
        n_jobs = int(self.cfg.experiment["cv"]["outer_folds"]) * len(self.cfg.experiment["targets_to_model"])
        lam_cost = t.get("nca_final", 0) * len(self.cfg.experiment["metric"]["lambda_grid"]) * \
            int(self.cfg.experiment["cv"]["inner_folds"]) * 0.7
        per_job = per_model * n_stop * (1 + (loop["n_attempts"] - loop["n_accepted"]) / max(loop["n_accepted"], 1) * 0.3) \
            + lam_cost + 3 * t.get("nca_final", 0) + t.get("baselines", 0) + 120
        par = int(self.cfg.experiment["resources"]["max_parallel_jobs"])
        d7 = {"min_delta": md, "tol_delta": float(stp["tol_delta"]), "W": int(stp["W"]), "P": int(stp["P"]),
              "n_min": int(stp["n_min"]), "n_max": int(stp["n_max"]), "epsilon": float(self.cfg.experiment["importance"]["epsilon"]),
              "exclude_families": exclude, "rule": pc["min_delta_rule"], "fixed_at": now_iso()}
        pilot = {"wall_seconds": wall, "rss_gb_peak": rss_gb(), "n_models_run": loop["n_accepted"],
                 "n_attempts": loop["n_attempts"], "acceptance_rate": loop["n_accepted"] / max(loop["n_attempts"], 1),
                 "simulated_models_until_stop": n_stop, "per_model_seconds_median": per_model,
                 "family_cost_median_s": fam_med, "timings": t, "delta": loop["delta"],
                 "delta_smooth": loop["delta_smooth"], "d7": d7,
                 "extrapolation": {"jobs": n_jobs, "seconds_per_job": per_job, "parallel": par,
                                   "hours_total_cv": per_job * n_jobs / par / 3600,
                                   "hours_with_sensitivity": per_job * n_jobs / par / 3600 *
                                   (1 + len(self.cfg.experiment["sensitivity"]["variants"]) * bool(
                                       self.cfg.experiment["sensitivity"]["enabled"]))},
                 "fold": f, "target": pc["target"]}
        write_json(self.ws.shared_artifacts() / "pilot.json", pilot)
        write_json(self.ws.shared_artifacts() / "d7.json", d7)
        _pilot_html(self.ws.reports / "pilot.html", pilot)
        self.note("DECISIONS", "D7 fijado con el piloto (regla declarada)",
                  f"min_delta={md:.6g} (regla {pc['min_delta_rule']}), tol_delta={d7['tol_delta']}, W={d7['W']}, "
                  f"P={d7['P']}, n_min={d7['n_min']}, n_max={d7['n_max']}, ε={d7['epsilon']}. "
                  f"Familias excluidas por coste: {exclude or 'ninguna'}. "
                  f"Modelos hasta la parada (simulado): {n_stop}. Coste CV estimado: {pilot['extrapolation']['hours_total_cv']:.1f} h.")
        LOG.info("pilot: %.1f min; parada simulada a %d modelos; CV estimada %.1f h", wall / 60, n_stop,
                 pilot["extrapolation"]["hours_total_cv"])
        self.mark("pilot")

    def stage_cv(self, folds_override: np.ndarray | None = None, tag: str = "cv",
                 split: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        from perisim.experiment import load_cohort_data, outer_folds

        if tag == "cv" and self.done("cv") and not self.force:
            return
        d7 = self.d7()
        if not d7:
            raise Stop("Falta artifacts/_shared/d7.json: ejecutar antes la etapa pilot (M3).")
        dev = self.cfg.experiment["cohorts"]["development"]
        data = load_cohort_data(self.ws.common(dev), self.ws.derived(dev), self.feature_names())
        if split is None:
            folds = outer_folds(data, int(self.cfg.experiment["cv"]["outer_folds"]), self.cfg.seed)
            write_parquet(pl.DataFrame({"stay_id": data.ft.ids.astype(str), "subject_id": data.ft.subjects.astype(str),
                                        "outer_fold": folds}), self.ws.artifacts() / "folds.parquet")
            jobs_split = [(f, np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(set(folds))]
        else:
            jobs_split = [(0, split[0], split[1])]
        ckey = self.key("cv")
        jobs = []
        for t in self.cfg.experiment["targets_to_model"]:
            for f, tr, te in jobs_split:
                out = self.ws.artifacts(tag, t) / f"fold{f}.json"
                prev = read_json(out)
                if prev and prev.get("_key") == ckey and not self.force:
                    continue
                jobs.append({"target": t, "fold": int(f), "train": tr, "test": te, "out": str(out),
                             "pred": str(self.ws.predictions(tag, t) / f"fold{f}"), "key": ckey})
        if not jobs:
            LOG.info("%s: todos los trabajos ya están hechos", tag)
        else:
            LOG.info("%s: %d trabajos (target × fold) en %d procesos × %d hilos", tag, len(jobs),
                     min(len(jobs), int(self.cfg.experiment["resources"]["max_parallel_jobs"])), self.threads())
            self._run_jobs(jobs, d7)
        if tag == "cv":
            self.mark("cv")

    def _run_jobs(self, jobs: list[dict], d7: dict) -> None:
        par = max(1, min(len(jobs), int(self.cfg.experiment["resources"]["max_parallel_jobs"])))
        common = {"cfg": self.cfg.to_dict(), "profile": self.cfg.profile, "variant": self.variant,
                  "workdir": str(self.ws.root), "d7": d7, "threads": self.threads(),
                  "features": self.feature_names()}
        dev = self.cfg.experiment["cohorts"]["development"]
        common["common_dir"] = str(self.ws.common(dev))
        common["derived_dir"] = str(self.ws.derived(dev))
        failures = []
        if par == 1:
            for j in jobs:
                r = _job_entry({**common, **j})
                if r is not True:
                    failures.append(r)
        else:
            old = {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS")}
            os.environ["POLARS_MAX_THREADS"] = str(self.threads())
            os.environ["OMP_NUM_THREADS"] = str(self.threads())
            try:
                with cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
                    futs = {ex.submit(_job_entry, {**common, **j}): j for j in jobs}
                    for fu in cf.as_completed(futs):
                        j = futs[fu]
                        try:
                            r = fu.result()
                        except Exception as exc:  # proceso muerto (p. ej. OOM)
                            r = f"{j['target']} fold {j['fold']}: proceso terminado ({exc!r})"
                        if r is not True:
                            failures.append(r)
                        else:
                            LOG.info("  ✓ %s fold %d", j["target"], j["fold"])
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        if failures:
            raise RuntimeError("Trabajos fallidos (relanzar reanuda solo estos):\n" + "\n".join(map(str, failures)))

    def stage_temporal(self) -> None:
        from perisim.experiment import load_cohort_data

        tc = self.cfg.experiment["cv"]["temporal"]
        if self.variant != "main" or str(tc.get("enabled")).lower() in ("false", "0", "no"):
            return
        dev = self.cfg.experiment["cohorts"]["development"]
        data = load_cohort_data(self.ws.common(dev), self.ws.derived(dev), self.feature_names())
        yr = data.year
        if np.isfinite(yr).mean() < 0.95:
            LOG.info("temporal: sin año fiable en %s (%.0f %% disponible) → se omite (§6.1)", dev, 100 * np.isfinite(yr).mean())
            self.note("DECISIONS", "Validación temporal omitida", "La fecha relativa no permite reconstruir el año.")
            return
        a, b = tc["train_years"]
        c, d = tc["test_years"]
        tr = np.where((yr >= a) & (yr <= b))[0]
        te = np.where((yr >= c) & (yr <= d))[0]
        if tr.size < 100 or te.size < 50:
            LOG.info("temporal: tamaños insuficientes (train=%d, test=%d)", tr.size, te.size)
            return
        self.stage_cv(tag="temporal", split=(tr, te))

    def stage_report(self) -> None:
        from perisim.evaluation.report import aggregate_cv, render_results

        if self.done("report") and not self.force:
            return
        res = aggregate_cv(self.ws.artifacts(), self.ws.predictions(), self.cfg.experiment["targets_to_model"], self.cfg)
        write_json(self.ws.artifacts() / "results.json", res)
        (self.ws.reports / f"results_{self.variant}.html").write_text(
            render_results(res, self.cfg, f"Resultados PERISIM — validación interna ({self.variant})"), encoding="utf-8")
        if any(self.ws.artifacts("temporal").rglob("fold*.json")):
            rt = aggregate_cv(self.ws.artifacts(), self.ws.predictions(), self.cfg.experiment["targets_to_model"],
                              self.cfg, tag="temporal")
            write_json(self.ws.artifacts() / "results_temporal.json", rt)
            (self.ws.reports / "results_temporal.html").write_text(
                render_results(rt, self.cfg, "Resultados PERISIM — validación temporal (2011–2018 → 2019–2020)"),
                encoding="utf-8")
        for t, r in res["targets"].items():
            lams = [f["lambda"] for f in r["folds"]]
            self.note("DECISIONS", f"λ elegido ({t})", f"λ por fold: {lams} (resultado en sí mismo, §5.4)")
            for f in r["folds"]:
                if f["plateau_above_tol"]:
                    self.note("DECISIONS", f"Meseta de Δ̃ alta ({t}, fold {f['fold']})",
                              f"Métrica estabilizada con ruido residual alto (Δ̃={f['plateau']:.4f} > tol): "
                              "heterogeneidad real entre modelos (§5.3.2).")
                if f["stop_reason"] != "patience":
                    self.note("DECISIONS", f"Sin convergencia ({t}, fold {f['fold']})",
                              f"El bucle terminó por {f['stop_reason']} con {f['n_accepted']} modelos.")
        self.mark("report")

    def stage_sensitivity(self) -> None:
        sc = self.cfg.experiment.get("sensitivity", {})
        if self.variant != "main" or not sc.get("enabled"):
            return
        for v in sc.get("variants", []):
            vcfg = self.cfg.with_overrides(v.get("overrides", {}), variant=v["name"])
            sub = Pipeline(vcfg, self.ws.for_variant(v["name"]), force=self.force)
            sub.disc = self.disc
            LOG.info("── sensibilidad: %s", v["name"])
            rc = sub.run(VARIANT_STAGES)
            if rc == 3:
                raise Stop(f"Variante {v['name']} detenida (ver log)")
            if rc != 0:
                raise RuntimeError(f"Variante {v['name']} con errores")

    def stage_external(self) -> None:
        from perisim.external import run_external

        if self.variant != "main":
            return
        fr = read_json(self.ws.state / "freeze.json")
        if self.cfg.experiment["gates"]["external_requires_freeze"]:
            if not fr:
                LOG.warning("external: PENDIENTE de congelación (§6.3). Revisa reports/results_main.html y, cuando el "
                            "pipeline esté cerrado:  bash run_pod.sh freeze --registration <ID_OSF>  y relanza.")
                return
            if fr.get("code_hash") != code_hash() or fr.get("config_hash") != self.cfg.hash():
                raise Stop("El código o la configuración han cambiado desde la congelación (§6.3: prohibido iterar). "
                           "Restaura la versión congelada (git checkout <tag>) o vuelve a congelar si aún no se han "
                           "mirado resultados externos.")
        if self.done("external") and not self.force:
            return
        run_external(self)
        self.mark("external")

    def stage_export(self) -> None:
        from perisim.evaluation.export import build_export

        build_export(self.ws, self.cfg)


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
    ax.axvline(pilot["simulated_models_until_stop"], color="k", ls="--", lw=0.8, label="parada simulada")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("modelos aceptados")
    ex = pilot["extrapolation"]
    sec = [box(f"Piloto M3 (fold {pilot['fold']}, {pilot['target']}). D7 fijado por regla declarada: "
               f"min_delta = {pilot['d7']['min_delta']:.4g}; familias excluidas: {pilot['d7']['exclude_families'] or 'ninguna'}."),
           fig_img(fig),
           table(["Medida", "Valor"], [["Duración del piloto (min)", pilot["wall_seconds"] / 60],
                                       ["Memoria (GB, proceso)", pilot["rss_gb_peak"]],
                                       ["Modelos ejecutados / intentos", f"{pilot['n_models_run']} / {pilot['n_attempts']}"],
                                       ["Modelos hasta la parada (simulado)", pilot["simulated_models_until_stop"]],
                                       ["Segundos por modelo (mediana)", pilot["per_model_seconds_median"]],
                                       ["Trabajos de la CV", ex["jobs"]], ["Horas estimadas CV", ex["hours_total_cv"]],
                                       ["Horas estimadas con sensibilidad", ex["hours_with_sensitivity"]]], nd=2),
           table(["Familia", "Coste mediano por modelo (s)"], [[k, v] for k, v in pilot["family_cost_median_s"].items()], nd=1),
           table(["Etapa", "segundos"], [[k, v] for k, v in pilot["timings"].items()], nd=1)]
    path.write_text(page("Piloto de SAGE y de la métrica (M3)", sec, "PERISIM · solo agregados"), encoding="utf-8")


# --------------------------------------------------------------------------- trabajo en proceso hijo
def _job_entry(args: dict):
    """Un trabajo (target × fold) en un proceso aparte. Devuelve True o un mensaje de error."""
    try:
        from perisim.config import Config
        from perisim.experiment import load_cohort_data, run_fold
        from perisim.utils import setup_logging

        setup_logging()
        set_threads(int(args["threads"]))
        cfg = Config(**args["cfg"], profile=args["profile"], variant=args["variant"])
        data = load_cohort_data(Path(args["common_dir"]), Path(args["derived_dir"]), args["features"])
        d7 = args["d7"]
        st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
        res, out = run_fold(cfg, data, args["target"], np.asarray(args["train"]), np.asarray(args["test"]),
                            seed=cfg.seed + 1000 * int(args["fold"]), threads=int(args["threads"]),
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
        return f"{args.get('target')} fold {args.get('fold')}:\n{traceback.format_exc()}"
