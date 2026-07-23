"""Required thirteen-section report for associational future diagnostics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.5g}" if isinstance(value, float) else str(value)


def _table(rows: Sequence[Mapping[str, Any]], limit: int = 14) -> str:
    if not rows:
        return "No eligible records."
    lines = [
        "| Group | Metric | Episodes | Episode-weighted mean | Clip-weighted diagnostic |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {row.get('group', 'unknown')} | {row.get('metric', 'unknown')} | "
            f"{row.get('episodes', 0)} | {_fmt(row.get('episode_weighted_mean'))} | "
            f"{_fmt(row.get('clip_weighted_mean_diagnostic'))} |"
        )
    return "\n".join(lines)


def _visual_cases(summary_dir: Path, limit: int = 5) -> str:
    path = summary_dir / "all_diagnostics.csv"
    if not path.is_file():
        return "No indexed side-by-side artifacts."
    cases: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("side_by_side_video_path")
            if value:
                source_root = row.get("artifact_source_root")
                target = (
                    Path(source_root) / value
                    if source_root
                    else summary_dir.parent / value
                ).resolve()
                relative = Path(os.path.relpath(target, summary_dir)).as_posix()
                cases.append(f"- [{value}](<{relative}>)")
            if len(cases) >= limit:
                break
    return "\n".join(cases) if cases else "No indexed side-by-side artifacts."


def generate_diagnostic_report(
    experiment_dir: Path, metrics: dict[str, Any] | None = None
) -> Path:
    experiment_dir = Path(experiment_dir)
    summary_dir = experiment_dir / "summary"
    if metrics is None:
        metrics_path = summary_dir / "diagnostic_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Aggregate diagnostics first; missing {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = experiment_dir / "diagnostic_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    source_path = experiment_dir / "source_manifest.json"
    source = (
        json.loads(source_path.read_text(encoding="utf-8"))
        if source_path.is_file() else {}
    )
    denominators = metrics.get("denominators", {})
    paired = metrics.get("paired_id_ood", {})
    overall = metrics.get("overall", [])
    outcome = metrics.get("by_outcome", [])
    condition = metrics.get("by_condition", [])
    perturbation = metrics.get("by_perturbation", [])
    resource_rows = [
        row for row in condition
        if "latency" in str(row.get("metric", "")) or "memory" in str(row.get("metric", ""))
    ]
    config = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    diagnostics = (
        config.get("diagnostics")
        if isinstance(config.get("diagnostics"), Mapping)
        else {}
    )
    mode = str(diagnostics.get("mode", "unknown"))
    if mode == "unconditional_future":
        research_question = (
            "Does the released Fast-WAM video branch's unconditional future agree with the "
            "realized outcome of the separately chosen, unchanged Fast-WAM action, and do those "
            "associations differ between success/failure or clean/OOD episodes? The protected "
            "policy action is recorded but is not an input to the video branch."
        )
        causal_limitation = (
            "`causal_interpretation_allowed=false`. The released `libero_uncond` video expert "
            "conditions on the current observation, language, and proprioception, but not on the "
            "protected policy action. Agreement therefore measures compatibility between an "
            "unconditional task-conditioned future prior and the realized action outcome. It is "
            "neither an action-conditioned dynamics prediction nor evidence that the action "
            "reads or benefits from an explicit future."
        )
        conditioning_limitations = """- The protected policy action is deliberately not fed to the video branch. Motion-direction cosine compares the predicted visual-representation change with the realized post-action change; it is not a direct cosine between 7-DoF actions and pixels.
- The released video and action experts may share training context, so agreement is an association and must not be described as causal dependence.
- Direct failure attribution requires blinded human labels in addition to automatic representation metrics; ambiguous cases must remain ambiguous."""
        temporal_architecture_limitation = (
            "- Step alignment follows the official decoded-frame/action frequency ratio; it "
            "does not imply that a decoded visual change can be mapped directly to one action "
            "dimension."
        )
        answer_scope = (
            "- Whether the release checkpoint's unconditional future matches the realized future "
            "at exact control-step offsets.\n"
            "- Whether unconditional future-consistency metrics are associated with successful "
            "versus failed episodes.\n"
            "- Whether OOD conditions or perturbation groups are associated with higher "
            "future-prediction error."
        )
    else:
        research_question = (
            "Does an action-conditioned shadow future agree with what the unchanged Fast-WAM "
            "action actually produces, and do those associations differ between success/failure "
            "or clean/OOD episodes?"
        )
        causal_limitation = (
            "`causal_interpretation_allowed=false`. The released `libero_uncond` video expert "
            "has `action_conditioned=false` and the strict probe must reject it. Even with a "
            "compatible checkpoint, the released Fast-WAM action branch does not read the "
            "generated future. Success/failure differences therefore remain associations."
        )
        conditioning_limitations = """- For the audited T=9 path, direct action cross-attention forms two groups of 16, but `first_frame_causal` permits all future latents to exchange information. The transitive dependency closure of every future frame is therefore the full 32-action horizon. A horizon of 10 must be rejected before reset.
- The released `libero_uncond` checkpoint cannot supply this action-conditioned diagnostic without a compatible checkpoint.
- Upstream loads MoT with `strict=False`; real probes require exact action-embedding loading and a source-reviewed checkpoint/config/training-recipe allowlist."""
        temporal_architecture_limitation = (
            "- Temporal DiT patch sizes other than one have no proven decoded-frame action-"
            "dependency mapping in this implementation and are rejected."
        )
        answer_scope = (
            "- Whether a compatible action-conditioned shadow future matches the actually "
            "observed future at covered temporal offsets.\n"
            "- Whether future-consistency metrics are associated with successful versus failed "
            "episodes.\n"
            "- Whether OOD conditions or perturbation groups are associated with higher "
            "future-prediction error."
        )
    report = f"""# Fast-WAM Future Consistency Diagnostic

## 1. Research question

{research_question} This Phase 2 study measures consistency; it does not modify the Phase 1 policy or its executable action.

## 2. Important causal limitation

{causal_limitation}

## 3. Checkpoint and upstream provenance

- Diagnostic experiment: `{manifest.get('experiment_id', experiment_dir.name)}`
- Source experiment: `{manifest.get('source_experiment_id', source.get('source_experiment_id', 'unknown'))}`
- Future mode: `{mode}`
- Protocol fingerprint: `{manifest.get('protocol_fingerprint', 'unknown')}`
- Provenance: `{json.dumps(manifest.get('provenance', {}), ensure_ascii=False, sort_keys=True)}`
- Source manifest SHA-256: `{source.get('source_manifest_sha256', 'unknown')}`

## 4. Diagnostic protocol

The baseline action chunk is copied and hashed before the observer probe runs; only that protected copy is executed. Probe seeds use episode seed + configured offset + ordinal probe index. Frame offsets use only the upstream-derived action/video ratio. Seconds are exact only if control frequency was verified at runtime; configured-only timing is approximate, and a missing real-backend ratio is unavailable rather than guessed.

Coverage: planned jobs={denominators.get('planned_jobs', 0)}, planned clips maximum={denominators.get('planned_clips_maximum', 'N/A')}, jobs with rows={denominators.get('completed_jobs_with_probe_rows', 0)}, generated clips={denominators.get('generated_clips', 0)}, exact/approximate/unavailable/error={denominators.get('exact_clips', 0)}/{denominators.get('approximate_clips', 0)}/{denominators.get('unavailable_clips', 0)}/{denominators.get('error_clips', 0)}, aligned future frames={denominators.get('aligned_future_frames', 0)}.

## 5. Overall consistency

Primary summaries first reduce clips within `job_id` (mean, median, worst value, static fraction), then weight episodes equally. Clip-weighted values are diagnostic only because clips within one episode are not independent. Latent L1/cosine are decoded-frame re-encoding proxies and exclude frame zero.

{_table(overall)}

## 6. Successful vs failed episodes

{_table(outcome)}

Outcome may itself change how many future frames are executable and alignable. Always interpret this table with its episode and aligned-frame denominators.

## 7. Clean vs OOD

{_table(condition)}

Eligible paired episodes: {paired.get('eligible_episode_pairs', 0)}. Paired metric differences (OOD minus ID) with episode-cluster bootstrap intervals: `{json.dumps(paired.get('metrics', {}), ensure_ascii=False, sort_keys=True)}`. Missing pairs or unavailable metrics are not imputed.

## 8. Results by perturbation

{_table(perturbation)}

Each group combines perturbation category and level; small groups are descriptive case studies rather than stable population estimates.

## 9. Static-future cases

`static_future_cases.csv` lists probes whose predicted motion energy is at or below the configured threshold. Static is a thresholded representation-space diagnostic, not proof that the video model ignored available conditioning or that the robot should not move.

## 10. Visual case studies

Per-rank `current_frames/` contains each probe's input frame, `predicted_futures/` contains the full generated sequence, and `actual_futures/` plus `side_by_side/` contain only temporally aligned observations/comparisons. Optional arrays are isolated in `latents/`. Artifact paths are indexed by `all_diagnostics.csv`; Phase 1 `episode_results.jsonl` and source outputs remain untouched.

Representative indexed side-by-side paths:

{_visual_cases(summary_dir)}

## 11. Runtime overhead

{_table(resource_rows)}

`diagnostic_latency_ms` covers shadow preprocessing, future generation, actual-frame preprocessing, paired metrics, and artifact writing while excluding environment stepping. Peak GPU memory is the synchronized future-generation window. Neither value is Phase 1 policy cost, and an end-to-end deployment estimate would also need the unchanged action-policy and simulator costs.

## 12. Limitations

{conditioning_limitations}
{temporal_architecture_limitation}
- Step-offset alignment does not make configured-only wall-clock timestamps exact.
- Decoded-frame VAE re-encodings are approximate, not native temporal video latents.
- The configured static-motion threshold is a manifest-pinned initial diagnostic threshold; it must be calibrated on no-op/static smoke clips before formal interpretation. Representation direction is not physical optical-flow direction.
- Simulator OOD findings do not establish real-robot robustness, and outcome associations are vulnerable to task difficulty and truncation confounds.

## 13. Conclusion

Can answer:

{answer_scope}

Cannot answer:

- Whether the base Fast-WAM action is causally determined by an explicit predicted future.
- Whether adding explicit future imagination would necessarily improve OOD success.
- Whether Joint WAM or IDM is better without recipe-matched checkpoints and a separately executed comparison.
"""
    output = summary_dir / "thought2_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


generate_report = generate_diagnostic_report

__all__ = ["generate_diagnostic_report", "generate_report"]
