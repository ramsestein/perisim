"""Transportability and incremental case base (SPECS §6.2b, pre-registration amendment of 2026-10-24).

Run after the external validation, on the frozen pipeline; replaces the earlier `transfer` stage, whose variants it
contains. The frozen artefacts (consensus shapes g_j, weights w_j, encoder, α, λ) are never refitted or modified:
their hashes are recorded at the start and checked at the end. See docs/prereg_amendment_transport.md.
"""

from perisim.transport.run import run_transport, transport_jobs  # noqa: F401
