"""Working directory layout (on KHDP: updown/perisim/<user>/work).

    common/<cohort>/        extraction to the common format (stays, inputs_long, outcomes) + flow
    derived/<variant>/<cohort>/   targets, carried, features (variant-dependent)
    artifacts/<variant>/    folds, weights, curves, metrics (aggregates)
    predictions/<variant>/  per-patient outputs (NEVER exported)
    reports/                HTML / MD reports (aggregates)
    export/                 the only candidates for export review (§9)
    state/                  markers of completed stages
    docs/                   OPEN_QUESTIONS / DECISIONS generated automatically in the pod
    cache/<cohort>/         filtered extracts of the original tables (zstd Parquet)
"""

from __future__ import annotations

from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path, variant: str = "main"):
        self.root = Path(root).expanduser()
        self.variant = variant

    def _d(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def for_variant(self, variant: str) -> "Workspace":
        return Workspace(self.root, variant)

    def cache(self, cohort: str) -> Path:
        return self._d("cache", cohort)

    def common(self, cohort: str) -> Path:
        return self._d("common", cohort)

    def derived(self, cohort: str) -> Path:
        return self._d("derived", self.variant, cohort)

    def artifacts(self, *sub: str) -> Path:
        return self._d("artifacts", self.variant, *sub)

    def shared_artifacts(self) -> Path:
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
