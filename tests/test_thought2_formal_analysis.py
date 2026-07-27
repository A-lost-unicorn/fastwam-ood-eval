from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fastwam_ood_eval.diagnostics.formal_analysis import (
    analyze_thought2_formal,
)


FIELDS = (
    "probe_id",
    "job_id",
    "probe_index",
    "environment_step",
    "suite",
    "task_id",
    "task_name",
    "episode_index",
    "episode_seed",
    "condition",
    "perturbation_category",
    "perturbation_level",
    "success",
    "termination_reason",
    "status",
    "action_unchanged",
    "aligned_future_frame_count",
    "future_latent_cosine_distance",
    "future_latent_l1",
    "motion_direction_cosine",
    "predicted_motion_energy",
    "actual_motion_energy",
    "motion_energy_ratio",
    "generation_latency_ms",
    "diagnostic_latency_ms",
    "generation_peak_memory_mb",
    "artifact_source_root",
    "current_frame_path",
    "predicted_video_path",
    "actual_video_path",
    "side_by_side_video_path",
)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(root: Path) -> tuple[Path, Path]:
    experiment = root / "thought2"
    combined = experiment / "combined"
    summary = combined / "summary"
    summary.mkdir(parents=True)
    threshold = 0.01
    (combined / "diagnostic_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "diagnostics": {
                        "static_motion_threshold": threshold,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    static_summary = (
        experiment / "static" / "combined" / "summary"
    )
    static_summary.mkdir(parents=True)
    (static_summary / "static_calibration_summary.json").write_text(
        json.dumps(
            {
                "candidate_static_motion_threshold": threshold,
                "freeze_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    categories = (
        "camera_viewpoints",
        "light_conditions",
        "background_textures",
        "objects_layout",
        "robot_initial_states",
    )
    levels = ("easy", "medium", "hard")
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    raw_by_condition: dict[str, list[dict[str, object]]] = {
        "clean": [],
        "ood": [],
    }

    def add(
        job_id: str,
        *,
        task: int,
        episode: int,
        condition: str,
        category: str,
        level: str,
        success: bool,
        cosine: float,
    ) -> None:
        probe_id = f"{job_id}-probe"
        row = {
            "probe_id": probe_id,
            "job_id": job_id,
            "probe_index": 0,
            "environment_step": 30,
            "suite": "suite_a",
            "task_id": task,
            "task_name": f"task_{task}",
            "episode_index": episode,
            "episode_seed": 1000 + episode,
            "condition": condition,
            "perturbation_category": category,
            "perturbation_level": level,
            "success": success,
            "termination_reason": "success" if success else "max_steps",
            "status": "completed",
            "action_unchanged": True,
            "aligned_future_frame_count": 2,
            "future_latent_cosine_distance": cosine,
            "future_latent_l1": cosine + 0.03,
            "motion_direction_cosine": 0.9 - cosine,
            "predicted_motion_energy": 0.20,
            "actual_motion_energy": 0.18,
            "motion_energy_ratio": 1.1,
            "generation_latency_ms": 3000.0,
            "diagnostic_latency_ms": 5000.0,
            "generation_peak_memory_mb": 24000.0,
            "artifact_source_root": str(root),
            "current_frame_path": "current.png",
            "predicted_video_path": "predicted.mp4",
            "actual_video_path": "actual.mp4",
            "side_by_side_video_path": "comparison.mp4",
        }
        rows.append(row)
        source_rows.append(
            {
                "job_id": job_id,
                "success": success,
                "termination_reason": row["termination_reason"],
            }
        )
        raw_by_condition[condition].append(
            {
                "probe_id": probe_id,
                "job_id": job_id,
                "action_unchanged": True,
                "action_hash": "same",
                "action_hash_before": "same",
                "action_hash_after": "same",
            }
        )

    for task in range(2):
        clean_value = 0.10 + 0.02 * task
        for episode in range(2):
            add(
                f"clean-{task}-{episode}",
                task=task,
                episode=episode,
                condition="clean",
                category="",
                level="",
                success=True,
                cosine=clean_value,
            )
        episode = 10
        for category_index, category in enumerate(categories):
            for level_index, level in enumerate(levels):
                success = (category_index + level_index + task) % 2 == 0
                add(
                    f"ood-{task}-{category}-{level}",
                    task=task,
                    episode=episode,
                    condition="ood",
                    category=category,
                    level=level,
                    success=success,
                    cosine=clean_value + 0.08 + 0.01 * level_index,
                )
                episode += 1
    _write_csv(summary / "all_diagnostics.csv", rows, FIELDS)
    (summary / "diagnostic_metrics.json").write_text(
        json.dumps(
            {
                "clips": len(rows),
                "denominators": {"planned_jobs": len(rows)},
            }
        ),
        encoding="utf-8",
    )
    source = root / "thought1.csv"
    _write_csv(
        source,
        source_rows,
        ("job_id", "success", "termination_reason"),
    )
    for condition, raw_rows in raw_by_condition.items():
        path = (
            experiment
            / "diagnostics"
            / "suite_a"
            / condition
            / "workers"
            / "rank_0"
            / "diagnostics.jsonl"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in raw_rows),
            encoding="utf-8",
        )
    return experiment, source


def test_formal_analysis_is_task_equal_isolated_and_manifested(tmp_path):
    experiment, source = _build_fixture(tmp_path)
    output = tmp_path / "analysis"
    result = analyze_thought2_formal(
        experiment_dir=experiment,
        thought1_summary_csv=source,
        output_dir=output,
        bootstrap_replicates=100,
        bootstrap_seed=17,
    )
    assert result["episodes"] == 34
    assert result["probes"] == 34
    assert result["outcome_matches"] == 34
    payload = json.loads(
        (output / "formal_analysis.json").read_text(encoding="utf-8")
    )
    primary = next(
        row
        for row in payload["primary_contrasts"]
        if row["metric"] == "future_latent_cosine_distance"
        and row["probe_mode"] == "all_available"
    )
    assert primary["eligible_tasks"] == 2
    assert primary["clean_episodes"] == 4
    assert primary["ood_episodes"] == 30
    assert primary["ood_minus_clean"] == pytest.approx(0.09)
    assert payload["causal_interpretation_allowed"] is False
    manifest = json.loads(
        (output / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_files_rewritten"] is False
    assert manifest["parameters"]["bootstrap_replicates"] == 100
    assert (output / "report.md").is_file()


def test_formal_analysis_refuses_existing_output(tmp_path):
    experiment, source = _build_fixture(tmp_path)
    output = tmp_path / "analysis"
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh path"):
        analyze_thought2_formal(
            experiment_dir=experiment,
            thought1_summary_csv=source,
            output_dir=output,
            bootstrap_replicates=100,
        )
