from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import write_config
from fastwam_ood_eval import cli
from fastwam_ood_eval.config import ConfigError, load_config
from fastwam_ood_eval.diagnostics.static_calibration import (
    STANDARD_NOOP_ACTION,
    calibration_compatibility_fingerprint,
    compute_static_calibration_metrics,
    preflight_static_calibration_output,
    run_static_calibration_worker,
)
from fastwam_ood_eval.diagnostics.static_calibration_aggregate import (
    aggregate_static_calibration,
    generate_static_calibration_report,
    higher_quantile,
    linear_quantile,
)
from fastwam_ood_eval.envs.mock import MockBenchmarkEnv
from fastwam_ood_eval.evaluation.jobs import plan_jobs
from fastwam_ood_eval.policy.mock import MockPolicy


class NeverSuccessEnv(MockBenchmarkEnv):
    def __init__(self):
        super().__init__(max_steps=20)
        self.control_freq = 20.0
        self.actions = []
        self.resets = 0

    def reset(self, job):
        self.resets += 1
        observation = super().reset(job)
        self.target = 10_000
        return observation

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64))
        return super().step(action)

    def is_success(self):
        return False


class NoActPolicy(MockPolicy):
    def __init__(self):
        super().__init__(control_horizon=2)
        self.act_calls = 0

    def act(self, observation):
        self.act_calls += 1
        raise AssertionError("Static calibration must never sample a policy action")


class DeterministicNoisyEncoder:
    def observation_to_model_frame(self, observation):
        value = int(observation["robot_state"][0])
        return np.full((2, 2, 3), value, dtype=np.uint8)

    def encode_frame_embeddings(self, frames):
        values = np.stack(frames).astype(np.float64)
        noise = np.arange(len(frames), dtype=np.float64).reshape(-1, 1, 1, 1)
        return values + noise * 0.01


def _calibration_cfg(
    root: Path,
    *,
    perturbation: bool = False,
    output_name: str = "calibration",
    formal_two_condition_gate: bool = False,
):
    path = write_config(
        root,
        perturbation=perturbation,
        episodes=1,
        output_name=output_name,
    )
    overrides = [
        "benchmark.tasks=[0]",
        "benchmark.max_steps=6",
        "static_calibration.enabled=true",
        "static_calibration.settle_steps=0",
        "static_calibration.capture_offsets=[0,2,4]",
        "static_calibration.repeated_same_frame_encodes=3",
        "static_calibration.threshold_quantile=0.5",
        "static_calibration.save_frames=false",
    ]
    if perturbation:
        overrides.extend(
            [
                "perturbation.category=[camera_viewpoints]",
                "perturbation.level=[easy]",
            ]
        )
    if formal_two_condition_gate:
        overrides.extend(
            [
                "static_calibration.minimum_samples_for_freeze=2",
                "static_calibration.required_conditions=[clean,ood]",
                "static_calibration.minimum_samples_per_condition_for_freeze=1",
                "static_calibration.required_ood_categories=[camera_viewpoints]",
                "static_calibration.minimum_samples_per_ood_category_for_freeze=1",
            ]
        )
    return load_config(path, overrides)


def test_metric_uses_per_sample_max_encoder_noise_and_full_noop_horizon():
    repeated = np.asarray([[[0.0]], [[0.1]], [[0.2]]])
    trajectory = np.asarray([[[0.0]], [[2.0]], [[4.0]]])
    frames = {
        0: np.zeros((1, 1, 3), dtype=np.uint8),
        2: np.full((1, 1, 3), 2, dtype=np.uint8),
        4: np.full((1, 1, 3), 4, dtype=np.uint8),
    }
    metrics = compute_static_calibration_metrics(
        repeated,
        trajectory,
        capture_offsets=(0, 2, 4),
        frames_by_offset=frames,
    )
    assert metrics["same_frame_pairwise_motion_energy"] == pytest.approx(
        [0.1, 0.2, 0.1]
    )
    assert metrics["same_frame_max_motion_energy"] == pytest.approx(0.2)
    assert metrics["noop_full_horizon_motion_energy"] == pytest.approx(4.0)
    assert metrics["pixel_full_horizon_mae"] == pytest.approx(4 / 255)


def test_calibration_is_opt_in_and_does_not_change_evaluation_job_ids(tmp_path):
    path = write_config(tmp_path, episodes=1)
    baseline = load_config(path)
    assert baseline.static_calibration.enabled is False
    assert "static_calibration" not in baseline.to_dict()
    calibration = load_config(
        path,
        [
            "static_calibration.enabled=true",
            "static_calibration.capture_offsets=[0,2,4]",
        ],
    )
    assert [job.job_id for job in plan_jobs(baseline)] == [
        job.job_id for job in plan_jobs(calibration)
    ]
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_config(
            path,
            [
                "static_calibration.enabled=true",
                "static_calibration.capture_offsets=[0,2,4]",
                "diagnostics.enabled=true",
                "diagnostics.source_experiment_id=source",
            ],
        )


def test_calibration_namespace_rejects_nested_diagnostic_output(tmp_path):
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    (diagnostic / "diagnostic_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="future-diagnostic"):
        preflight_static_calibration_output(diagnostic / "nested_calibration")


def test_formal_calibration_configs_lock_exact_clean_ood_denominators():
    clean = load_config(
        Path("configs/studies/thought2_static_calibration_formal_clean.yaml")
    )
    ood = load_config(
        Path("configs/studies/thought2_static_calibration_formal_ood.yaml")
    )
    clean_jobs = plan_jobs(clean)
    ood_jobs = plan_jobs(ood)

    assert len(clean_jobs) == len({job.job_id for job in clean_jobs}) == 100
    assert len(ood_jobs) == len({job.job_id for job in ood_jobs}) == 100
    assert all(job.condition == "clean" and not job.skip_reason for job in clean_jobs)
    assert all(
        job.condition == "ood"
        and job.perturbation_level == "easy"
        and not job.skip_reason
        for job in ood_jobs
    )
    assert Counter(job.perturbation_category for job in ood_jobs) == {
        "camera_viewpoints": 20,
        "light_conditions": 20,
        "background_textures": 20,
        "robot_initial_states": 20,
        "objects_layout": 20,
    }
    assert calibration_compatibility_fingerprint(
        clean, {}
    ) == calibration_compatibility_fingerprint(ood, {})


def test_calibration_cli_dry_run_is_read_only_and_loads_no_runtime(
    tmp_path, monkeypatch, capsys
):
    path = write_config(tmp_path, episodes=1)
    output_dir = tmp_path / "mock_eval"

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run loaded a model, encoder, or environment")

    monkeypatch.setattr(cli, "_make_policy", forbidden)
    monkeypatch.setattr(cli, "_make_environment", forbidden)
    monkeypatch.setattr(cli, "FastWAMFutureProbe", forbidden)
    status = cli.main(
        [
            "calibrate-static",
            "--config",
            str(path),
            "--dry-run",
            "--set",
            "benchmark.tasks=[0]",
            "--set",
            "static_calibration.enabled=true",
            "--set",
            "static_calibration.capture_offsets=[0,2,4]",
        ]
    )
    assert status == 0
    assert json.loads(capsys.readouterr().out) == {
        "assigned": 1,
        "pending": 1,
        "completed": 0,
        "eligible_samples": 0,
        "skipped_by_resume": 0,
    }
    assert not output_dir.exists()


def test_worker_only_executes_standard_noop_and_resumes(tmp_path):
    cfg = _calibration_cfg(tmp_path / "cfg")
    job = plan_jobs(cfg)[0]
    environment = NeverSuccessEnv()
    policy = NoActPolicy()
    outcome = run_static_calibration_worker(
        cfg,
        policy=policy,
        environment=environment,
        encoder=DeterministicNoisyEncoder(),
        jobs=[job],
        provenance={"checkpoint_hash": "mock", "fastwam_commit": "mock"},
        close_resources=False,
    )
    assert outcome == {
        "assigned": 1,
        "pending": 1,
        "completed": 1,
        "eligible_samples": 1,
        "skipped_by_resume": 0,
    }
    assert policy.act_calls == 0
    assert len(environment.actions) == 4
    assert all(
        np.array_equal(action, np.asarray(STANDARD_NOOP_ACTION))
        for action in environment.actions
    )

    manifest = json.loads(
        (cfg.experiment.output_dir / "calibration_manifest.json").read_text()
    )
    assert manifest["kind"] == "static_motion_calibration"
    assert not (cfg.experiment.output_dir / "experiment_manifest.json").exists()
    assert not (cfg.experiment.output_dir / "diagnostic_manifest.json").exists()
    row = json.loads(
        (
            cfg.experiment.output_dir
            / "workers/rank_0/static_calibration_samples.jsonl"
        ).read_text()
    )
    assert row["status"] == "completed"
    assert row["eligible_for_threshold"] is True
    assert row["policy_action_sampled"] is False
    assert row["metrics"]["same_frame_max_motion_energy"] == pytest.approx(
        0.02
    )
    assert row["metrics"]["noop_full_horizon_motion_energy"] == pytest.approx(
        4.02
    )

    second_environment = NeverSuccessEnv()
    resumed = run_static_calibration_worker(
        cfg,
        policy=NoActPolicy(),
        environment=second_environment,
        encoder=DeterministicNoisyEncoder(),
        jobs=[job],
        provenance={"checkpoint_hash": "mock", "fastwam_commit": "mock"},
        close_resources=False,
    )
    assert resumed["pending"] == 0
    assert resumed["skipped_by_resume"] == 1
    assert second_environment.resets == 0

    changed = replace(
        cfg,
        static_calibration=replace(
            cfg.static_calibration,
            threshold_quantile=0.95,
        ),
    )
    with pytest.raises(RuntimeError, match="protocol changed"):
        cli._dry_run_static_calibration(
            changed,
            rank=0,
            world_size=1,
            rerun="incomplete",
        )


def test_pooling_clean_ood_freeze_gate_and_read_only_pilot_sensitivity(
    tmp_path,
):
    clean = _calibration_cfg(
        tmp_path / "clean",
        output_name="calibration_clean",
        formal_two_condition_gate=True,
    )
    ood = _calibration_cfg(
        tmp_path / "ood",
        perturbation=True,
        output_name="calibration_ood",
        formal_two_condition_gate=True,
    )
    provenance = {
        "checkpoint_hash": "mock",
        "fastwam_commit": "mock",
        "git_dirty": False,
        "fastwam_dirty": False,
        "libero_dirty": False,
        "libero_plus_dirty": False,
    }
    for cfg in (clean, ood):
        run_static_calibration_worker(
            cfg,
            policy=NoActPolicy(),
            environment=NeverSuccessEnv(),
            encoder=DeterministicNoisyEncoder(),
            provenance=provenance,
        )

    diagnostic_dir = tmp_path / "diagnostic"
    diagnostic_manifest = diagnostic_dir / "diagnostic_manifest.json"
    diagnostic_manifest.parent.mkdir(parents=True)
    diagnostic_manifest.write_text(
        json.dumps(
            {
                "experiment_id": "pilot",
                "protocol_fingerprint": "pilot-fingerprint",
                "config": {
                    "diagnostics": {"static_motion_threshold": 1.0}
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostic_row = (
        diagnostic_dir / "workers/rank_0/diagnostics.jsonl"
    )
    diagnostic_row.parent.mkdir(parents=True)
    diagnostic_row.write_text(
        json.dumps(
            {
                "diagnostic_id": "probe-1",
                "job_id": "job-1",
                "condition": "clean",
                "episode_success": True,
                "status": "completed",
                "attempt_started_ns": 1,
                "metrics": {
                    "predicted_motion_energy": 5.0,
                    "actual_motion_energy": 3.0,
                    "predicted_static": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with diagnostic_row.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "diagnostic_id": "probe-without-outcome",
                    "job_id": "job-2",
                    "condition": "clean",
                    "status": "completed",
                    "attempt_started_ns": 2,
                    "metrics": {
                        "predicted_motion_energy": 5.0,
                        "actual_motion_energy": 3.0,
                        "predicted_static": False,
                    },
                }
            )
            + "\n"
        )
    source_before = {
        path.relative_to(diagnostic_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in diagnostic_dir.rglob("*")
        if path.is_file()
    }

    destination = tmp_path / "comparison"
    summary = aggregate_static_calibration(
        destination,
        [clean.experiment.output_dir, ood.experiment.output_dir],
        [diagnostic_dir],
    )
    assert summary["eligible_sample_count"] == 2
    assert summary["threshold_status"] == "eligible_for_manual_freeze"
    assert summary["candidate_static_motion_threshold"] == pytest.approx(
        4.02
    )
    assert all(
        check["passed"] for check in summary["freeze_checks"].values()
    )
    assert all(
        source["job_manifest_sha256"]
        and source["sample_files"]
        and source["sample_files"][0]["sha256"]
        for source in summary["source_manifests"]
    )
    sensitivity = summary["diagnostic_sensitivity"]
    assert sensitivity["classified_rows"] == 2
    assert sensitivity["missing_episode_success_labels"] == 1
    assert sensitivity["groups"]["all"]["predicted_static_count"] == 0
    assert sensitivity["groups"]["all"]["actual_static_count"] == 2
    assert sensitivity["groups"]["episode_success:true"]["rows"] == 1
    assert sensitivity["groups"]["episode_success:false"]["rows"] == 0
    assert sensitivity["diagnostic_sources"][0]["diagnostic_files"][0][
        "sha256"
    ]
    assert sensitivity["source_rows_rewritten"] is False

    source_after = {
        path.relative_to(diagnostic_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in diagnostic_dir.rglob("*")
        if path.is_file()
    }
    assert source_after == source_before
    summary_path = destination / "summary/static_calibration_summary.json"
    sensitivity_path = (
        destination / "summary/static_threshold_sensitivity.json"
    )
    sensitivity_file = json.loads(sensitivity_path.read_text())
    assert sensitivity_file["calibration_summary_sha256"] == hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    report_path = generate_static_calibration_report(destination)
    assert report_path.is_file()
    assert "eligible_for_manual_freeze" in report_path.read_text()

    ood_samples_path = (
        ood.experiment.output_dir
        / "workers/rank_0/static_calibration_samples.jsonl"
    )
    ood_sample = json.loads(ood_samples_path.read_text())
    ood_sample["runtime_control_frequency_hz"] = 10.0
    ood_samples_path.write_text(json.dumps(ood_sample) + "\n")
    frequency_mismatch = aggregate_static_calibration(
        tmp_path / "frequency_mismatch",
        [clean.experiment.output_dir, ood.experiment.output_dir],
    )
    assert frequency_mismatch["threshold_status"] == "candidate_only"
    assert (
        frequency_mismatch["freeze_checks"][
            "runtime_control_frequency_consistent"
        ]["passed"]
        is False
    )

    clean_manifest_path = (
        clean.experiment.output_dir / "calibration_manifest.json"
    )
    clean_manifest = json.loads(clean_manifest_path.read_text())
    clean_manifest["provenance"]["git_dirty"] = True
    clean_manifest_path.write_text(json.dumps(clean_manifest))
    dirty_source = aggregate_static_calibration(
        tmp_path / "dirty_source",
        [clean.experiment.output_dir, ood.experiment.output_dir],
    )
    assert dirty_source["threshold_status"] == "candidate_only"
    assert (
        dirty_source["freeze_checks"]["all_source_trees_explicitly_clean"][
            "passed"
        ]
        is False
    )


def test_linear_quantile_interpolates_and_empty_is_unavailable():
    assert linear_quantile([], 0.99) is None
    assert linear_quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert higher_quantile([], 0.99) is None
    assert higher_quantile([0.0, 10.0], 0.25) == pytest.approx(10.0)
