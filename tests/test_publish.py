"""Copy of the aggregate outputs to updown after a run (2026-10-24): reports, docs and export, never patient data."""

from perisim.evaluation.export import publish
from perisim.workspace import Workspace


def test_publish_copies_aggregates_only(tmp_path):
    ws = Workspace(tmp_path / "work", experiment="t2")
    (ws.reports / "results_main.html").write_text("<html></html>")
    (ws.data_reports / "schema_inspire.md").write_text("# Schema")
    (ws.docs / "DECISIONS.md").write_text("notes")
    (ws.export / "artifacts").mkdir(parents=True, exist_ok=True)
    (ws.export / "artifacts" / "results.json").write_text("{}")
    (ws.export / "perisim_export_2026-10-24.zip").write_bytes(b"PK")
    (ws.predictions("cv", "t2") / "fold0.jsonl").write_text('{"patient_id": "INSPIRE-1"}')
    (ws.common("inspire") / "stays_index.parquet").write_bytes(b"PAR1")
    (ws.cache("inspire") / "labs_x.parquet").write_bytes(b"PAR1")
    dest = tmp_path / "updown" / "results"
    n = publish(ws, dest)
    got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert got == ["experiments/t2/docs/DECISIONS.md", "experiments/t2/export/artifacts/results.json",
                   "experiments/t2/export/perisim_export_2026-10-24.zip", "experiments/t2/reports/results_main.html",
                   "reports/schema_inspire.md"]
    assert n == len(got)
