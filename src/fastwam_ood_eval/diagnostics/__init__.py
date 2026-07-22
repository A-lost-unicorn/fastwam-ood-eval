"""Opt-in Phase 2 shadow future diagnostics."""

from fastwam_ood_eval.diagnostics.aggregate import aggregate_diagnostics
from fastwam_ood_eval.diagnostics.artifact_writer import DiagnosticArtifactWriter
from fastwam_ood_eval.diagnostics.diagnostic_runner import (
    diagnostic_protocol_fingerprint,
    load_source_jobs,
    run_diagnostic_episode,
    run_diagnostic_worker,
    validate_probe_capability,
    validate_source_provenance,
)
from fastwam_ood_eval.diagnostics.report import generate_diagnostic_report

__all__ = [
    "DiagnosticArtifactWriter",
    "aggregate_diagnostics",
    "diagnostic_protocol_fingerprint",
    "generate_diagnostic_report",
    "load_source_jobs",
    "run_diagnostic_episode",
    "run_diagnostic_worker",
    "validate_probe_capability",
    "validate_source_provenance",
]
