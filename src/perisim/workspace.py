"""Estructura del directorio de trabajo (en KHDP: updown/perisim/<usuario>/work).

    common/<cohorte>/       extracción al formato común (stays, inputs_long, outcomes) + flow
    derived/<variante>/<cohorte>/   targets, carried, features (dependen de la variante)
    artifacts/<variante>/   folds, pesos, curvas, métricas (agregados)
    predictions/<variante>/ salidas por paciente (NUNCA se exportan)
    reports/                informes HTML / MD (agregados)
    export/                 lo único candidato a revisión de exportación (§9)
    state/                  marcadores de etapas completadas
    docs/                   OPEN_QUESTIONS / DECISIONS generados automáticamente en el pod
    cache/<cohorte>/        extractos filtrados de las tablas originales (Parquet zstd)
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
