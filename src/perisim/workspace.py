"""Working directory layout (on KHDP: updown/perisim/<user>/work).

One experiment per target (decided 2026-10-24: T2, then T1, then T3). The data layer is shared by all
experiments; everything that depends on the target lives under experiments/<name>/:

    common/<cohort>/                  extraction to the common format (stays, inputs_long, outcomes) + flow  [shared]
    cache/<cohort>/                   filtered extracts of the original tables (zstd Parquet)               [shared]
    artifacts/_data/                  discovery of the mounted cohorts                                      [shared]
    reports/                          schema reports of the discovery                                       [shared]
    state/_/                          markers of the shared stages (env, discover, extract)                 [shared]
    experiments/<name>/
        derived/<variant>/<cohort>/   targets, carried, features (variant-dependent)
        artifacts/<variant>/          folds, weights, curves, metrics (aggregates)
        predictions/<variant>/        per-patient outputs and fold models (NEVER exported)
        reports/                      HTML / MD reports (aggregates)
        export/                       the only candidates for export review (§9)
        state/                        markers of the experiment's stages, freeze
        docs/                         OPEN_QUESTIONS / DECISIONS generated automatically in the pod
"""

from __future__ import annotations

from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path, variant: str = "main", experiment: str | None = None):
        self.base = Path(root).expanduser()
        self.variant = variant
        self.experiment = experiment
        self.root = self.base / "experiments" / experiment if experiment else self.base

    def _d(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _shared(self, *parts: str) -> Path:
        p = self.base.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def for_variant(self, variant: str) -> "Workspace":
        return Workspace(self.base, variant, self.experiment)

    # ------------------------------------------------------------------ shared data layer
    def cache(self, cohort: str) -> Path:
        return self._shared("cache", cohort)

    def common(self, cohort: str) -> Path:
        return self._shared("common", cohort)

    def data_artifacts(self) -> Path:
        return self._shared("artifacts", "_data")

    @property
    def data_reports(self) -> Path:
        return self._shared("reports")

    @property
    def shared_state(self) -> Path:
        return self._shared("state")

    # ------------------------------------------------------------------ experiment
    def derived(self, cohort: str) -> Path:
        return self._d("derived", self.variant, cohort)

    def artifacts(self, *sub: str) -> Path:
        return self._d("artifacts", self.variant, *sub)

    def shared_artifacts(self) -> Path:
        """Shared by the variants of one experiment (pilot, D7 stopping parameters, environment)."""
        return self._d("artifacts", "_shared")

    def predictions(self, *sub: str) -> Path:
        return self._d("predictions", self.variant, *sub)

    @property
    def reports(self) -> Path:
        return self._d("reports")

    @property
    def export(self) -> Path:
        return self._d("export")

    @property
    def state(self) -> Path:
        return self._d("state")

    @property
    def docs(self) -> Path:
        return self._d("docs")

    @property
    def logs(self) -> Path:
        return self._d("logs")
