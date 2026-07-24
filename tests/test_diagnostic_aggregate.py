from __future__ import annotations

import json

import pytest

from fastwam_ood_eval.diagnostics.aggregate import aggregate_diagnostics
from fastwam_ood_eval.diagnostics.artifact_writer import load_all_completed_jobs
from fastwam_ood_eval.diagnostics.report import generate_diagnostic_report


def _row(job_id: str, probe: int, *, success: bool, condition: str, value: float) -> dict:
    return {
        "schema_version": 1,
        "diagnostic_id": f"{job_id}-{probe}",
        "experiment_id": "diag",
        "source_experiment_id": "source",
        "job_id": job_id,
        "replan_index": probe,
        "origin_env_step": probe,
        "suite": "suite",
        "task_id": 0,
        "episode_index": 0,
        "episode_seed": 10,
        "condition": condition,
        "perturbation_category": None if condition == "clean" else "camera",
        "perturbation_level": None if condition == "clean" else "easy",
        "success": success,
        "episode_success": success,
        "termination_reason": "success" if success else "max_steps",
        "status": "completed",
        "num_video_frames": 5,
        "approximate_alignment": False,
        "static_future_flag": value == 0,
        "metrics": {"future_latent_l1": value, "static_future_flag": value == 0},
        "generation_latency_ms": 5.0,
        "generation_peak_memory_mb": 100.0,
        "extra": {"protocol_fingerprint": "fp", "aligned_future_frame_count": 2},
    }


def _manifest(*, fingerprint: str, inference_steps: int = 20, planned_jobs: int = 2) -> dict:
    return {
        "kind": "future_shadow_diagnostics",
        "protocol_fingerprint": fingerprint,
        "planned_job_count": planned_jobs,
        "config": {
            "checkpoint": {"path": "same.pt", "model_name": "same"},
            "benchmark": {
                "backend": "mock",
                "suite": "suite",
                "control_horizon": 16,
                "image_size": [224, 448],
            },
            "diagnostics": {
                "mode": "action_conditioned_future",
                "num_video_frames": 9,
                "num_inference_steps": inference_steps,
                "static_motion_threshold": 1.0,
                "motion_epsilon": 1e-8,
                "probe_strategy": "first",
                "max_probes_per_episode": 1,
                "explicit_replan_indices": [],
            },
        },
        "provenance": {"checkpoint_hash": "hash", "fastwam_commit": "commit"},
    }


def test_empty_aggregate_and_exact_report_contract(tmp_path):
    metrics = aggregate_diagnostics(tmp_path)
    expected = {
        "all_diagnostics.csv",
        "consistency_by_outcome.csv",
        "consistency_by_condition.csv",
        "consistency_by_perturbation.csv",
        "static_future_cases.csv",
    }
    assert {path.name for path in (tmp_path / "summary").glob("*.csv")} == expected
    output = generate_diagnostic_report(tmp_path, metrics)
    assert output.name == "thought2_report.md"
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Fast-WAM Future Consistency Diagnostic\n")
    assert text.count("\n## ") == 13
    headings = [
        "Research question", "Important causal limitation", "Checkpoint and upstream provenance",
        "Diagnostic protocol", "Overall consistency", "Successful vs failed episodes",
        "Clean vs OOD", "Results by perturbation", "Static-future cases",
        "Visual case studies", "Runtime overhead", "Limitations", "Conclusion",
    ]
    assert all(f"## {index}. {heading}" in text for index, heading in enumerate(headings, 1))


def test_aggregate_is_episode_weighted_and_filters_stale_protocol(tmp_path):
    (tmp_path / "workers" / "rank_0").mkdir(parents=True)
    (tmp_path / "diagnostic_manifest.json").write_text(
        json.dumps(_manifest(fingerprint="fp")), encoding="utf-8"
    )
    rows = [
        _row("long", index, success=True, condition="clean", value=10.0) for index in range(2)
    ] + [_row("short", 0, success=False, condition="ood", value=0.0)]
    stale = _row("stale", 0, success=False, condition="ood", value=999.0)
    stale["extra"]["protocol_fingerprint"] = "old"
    path = tmp_path / "workers" / "rank_0" / "diagnostics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [*rows, stale]) + "\n", encoding="utf-8")
    metrics = aggregate_diagnostics(tmp_path)
    assert metrics["clips"] == 3
    assert metrics["episodes"] == 2
    assert metrics["denominators"]["planned_jobs"] == 2
    assert metrics["denominators"]["aligned_future_frames"] == 6
    assert metrics["causal_interpretation_allowed"] is False
    generation_latency = next(
        row for row in metrics["overall"] if row["metric"] == "generation_latency_ms"
    )
    assert generation_latency["episode_weighted_mean"] == 5.0
    perturbation_csv = (tmp_path / "summary" / "consistency_by_perturbation.csv").read_text()
    assert "camera/easy" in perturbation_csv
    manifest = json.loads(
        (tmp_path / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "aggregated"
    assert manifest["aggregation"] == {
        "episodes": 2,
        "clips": 3,
        "error_clips": 0,
        "summary_dir": str(tmp_path / "summary"),
    }


def test_multi_input_rejects_incompatible_inference_protocols(tmp_path):
    roots = [tmp_path / "clean", tmp_path / "ood"]
    for index, root in enumerate(roots):
        root.mkdir()
        (root / "diagnostic_manifest.json").write_text(
            json.dumps(
                _manifest(
                    fingerprint=f"fp-{index}",
                    inference_steps=10 + index,
                )
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_diagnostics(tmp_path / "comparison", roots)


def test_aggregate_refuses_to_write_into_thought1_source(tmp_path):
    (tmp_path / "experiment_manifest.json").write_text(
        '{"experiment_id":"thought1"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Thought 1/source"):
        aggregate_diagnostics(tmp_path)

    assert not (tmp_path / "summary").exists()


@pytest.mark.parametrize("manifest_text", [None, "{not-json"])
def test_nonempty_diagnostic_input_requires_valid_manifest(tmp_path, manifest_text):
    worker = tmp_path / "workers" / "rank_0"
    worker.mkdir(parents=True)
    (worker / "diagnostics.jsonl").write_text(
        json.dumps(_row("job", 0, success=True, condition="clean", value=1.0)) + "\n",
        encoding="utf-8",
    )
    if manifest_text is not None:
        (tmp_path / "diagnostic_manifest.json").write_text(
            manifest_text, encoding="utf-8"
        )

    with pytest.raises(ValueError, match="manifest"):
        aggregate_diagnostics(tmp_path)


def test_latest_attempt_wins_across_rank_directories(tmp_path):
    (tmp_path / "diagnostic_manifest.json").write_text(
        json.dumps(_manifest(fingerprint="fp", planned_jobs=1)), encoding="utf-8"
    )
    old = _row("same-job", 0, success=False, condition="ood", value=99.0)
    old.update({"attempt_started_ns": 100, "recorded_at_ns": 101})
    new = _row("same-job", 0, success=True, condition="ood", value=1.0)
    new.update({"attempt_started_ns": 200, "recorded_at_ns": 201})
    for rank, row in ((2, old), (0, new)):
        worker = tmp_path / "workers" / f"rank_{rank}"
        worker.mkdir(parents=True)
        (worker / "diagnostics.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        completion = {
            "job_id": "same-job",
            "protocol_fingerprint": "fp",
            "status": "completed" if rank == 0 else "error",
            "termination_reason": "success" if rank == 0 else "exception",
            "attempt_started_ns": row["attempt_started_ns"],
            "recorded_at_ns": row["recorded_at_ns"],
        }
        (worker / "completed_jobs.jsonl").write_text(
            json.dumps(completion) + "\n", encoding="utf-8"
        )

    metrics = aggregate_diagnostics(tmp_path)
    assert metrics["clips"] == 1
    overall_l1 = next(
        row for row in metrics["overall"] if row["metric"] == "future_latent_l1"
    )
    assert overall_l1["episode_weighted_mean"] == 1.0
    assert load_all_completed_jobs(tmp_path)[("same-job", "fp")]["status"] == "completed"


def test_error_probe_scalars_are_excluded_from_consistency_means(tmp_path):
    (tmp_path / "diagnostic_manifest.json").write_text(
        json.dumps(_manifest(fingerprint="fp", planned_jobs=2)), encoding="utf-8"
    )
    worker = tmp_path / "workers" / "rank_0"
    worker.mkdir(parents=True)
    good = _row("good", 0, success=True, condition="clean", value=1.0)
    bad = _row("bad", 0, success=False, condition="clean", value=999.0)
    bad["status"] = "error"
    bad["error"] = "metric failed"
    (worker / "diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (good, bad)) + "\n",
        encoding="utf-8",
    )

    metrics = aggregate_diagnostics(tmp_path)
    overall_l1 = next(
        row for row in metrics["overall"] if row["metric"] == "future_latent_l1"
    )
    assert overall_l1["episodes"] == 2
    assert overall_l1["episodes_with_metric"] == 1
    assert overall_l1["episode_weighted_mean"] == 1.0


def test_multi_input_denominators_and_visual_roots_are_preserved(tmp_path):
    target = tmp_path / "comparison"
    roots = [tmp_path / "clean", tmp_path / "ood"]
    planned = [2, 3]
    for index, (root, planned_jobs) in enumerate(zip(roots, planned)):
        worker = root / "workers" / "rank_0"
        worker.mkdir(parents=True)
        fingerprint = f"fp-{index}"
        (root / "diagnostic_manifest.json").write_text(
            json.dumps(_manifest(fingerprint=fingerprint, planned_jobs=planned_jobs)),
            encoding="utf-8",
        )
        row = _row(
            f"job-{index}",
            0,
            success=not index,
            condition="clean" if index == 0 else "ood",
            value=float(index + 1),
        )
        row["extra"]["protocol_fingerprint"] = fingerprint
        row["side_by_side_video_path"] = f"workers/rank_0/side_by_side/case-{index}.mp4"
        (worker / "diagnostics.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )

    metrics = aggregate_diagnostics(target, roots)
    assert metrics["denominators"]["planned_jobs"] == 5
    assert metrics["denominators"]["planned_clips_maximum"] == 5
    manifest = json.loads(
        (target / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["aggregation_kind"] == "multi_input_comparison"
    assert manifest["planned_job_count"] == 0
    assert manifest["config"]["diagnostics"]["mode"] == "action_conditioned_future"
    assert len(manifest["comparison_inputs"]) == 2
    assert manifest["aggregation_provenance"]["git_dirty"] in (True, False, None)
    repeated = aggregate_diagnostics(target, roots)
    assert repeated["denominators"]["planned_jobs"] == 5
    report = generate_diagnostic_report(target, metrics).read_text(encoding="utf-8")
    assert "case-0.mp4" in report and "case-1.mp4" in report
    assert str(roots[0].name) in report or "../clean/" in report
