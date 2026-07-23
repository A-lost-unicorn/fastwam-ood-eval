from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import write_config
from fastwam_ood_eval.config import DiagnosticsConfig, load_config
from fastwam_ood_eval.diagnostics.artifact_writer import (
    DiagnosticArtifactWriter,
    action_chunk_hash,
    load_all_completed_jobs,
)
from fastwam_ood_eval.diagnostics.diagnostic_runner import (
    _probe_indices,
    diagnostic_protocol_fingerprint,
    load_source_jobs,
    run_diagnostic_worker,
)
from fastwam_ood_eval.diagnostics.protocol import FutureProbeOutput
from fastwam_ood_eval.envs.mock import MockBenchmarkEnv
from fastwam_ood_eval.evaluation.episode_runner import run_episode
from fastwam_ood_eval.evaluation.jobs import plan_jobs, write_jobs
from fastwam_ood_eval.policy.mock import MockPolicy


class RecordingEnv(MockBenchmarkEnv):
    control_freq = 20.0

    def __init__(self):
        super().__init__(max_steps=2)
        self.actions = []
        self.resets = 0

    def reset(self, job):
        self.resets += 1
        observation = super().reset(job)
        self.target = 2
        return observation

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return super().step(action)


class Probe:
    def __init__(
        self,
        *,
        group_size=2,
        decoded_frames_per_group=2,
        attention_mode="per_frame_causal",
        action_horizon=4,
    ):
        self.seeds = []
        self.group_size = group_size
        self.decoded_frames_per_group = decoded_frames_per_group
        self.attention_mode = attention_mode
        self.action_horizon = action_horizon

    def validate_capability(self):
        return None

    def observation_to_model_frame(self, observation):
        value = int(observation["robot_state"][0])
        return np.full((4, 4, 3), value, dtype=np.uint8)

    def encode_frame_embeddings(self, frames):
        return np.stack(frames)

    def predict_action_conditioned_future(
        self, observation, actions, *, diagnostic_seed, num_video_frames, num_inference_steps
    ):
        self.seeds.append(diagnostic_seed)
        actions[0][0] = 999.0
        frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(5)]
        return FutureProbeOutput(
            predicted_frames=frames,
            latency_ms=1.0,
            metadata={
                "action_video_freq_ratio": 1,
                "num_video_frames": 5,
                "action_conditioning_group_size": self.group_size,
                "vae_temporal_downsample_factor": self.decoded_frames_per_group,
                "video_dit_temporal_patch_size": 1,
                "video_attention_mask_mode": self.attention_mode,
                "action_horizon": self.action_horizon,
            },
        )


class UnconditionalProbe(Probe):
    def predict_unconditional_future(
        self, observation, actions, *, diagnostic_seed, num_video_frames, num_inference_steps
    ):
        output = super().predict_action_conditioned_future(
            observation,
            actions,
            diagnostic_seed=diagnostic_seed,
            num_video_frames=num_video_frames,
            num_inference_steps=num_inference_steps,
        )
        output.metadata = {
            "future_kind": "unconditional",
            "action_conditioned": False,
            "action_video_freq_ratio": 1,
            "num_video_frames": 5,
        }
        return output


def _cfg(tmp_path):
    return load_config(
        write_config(tmp_path, episodes=1),
        [
            "benchmark.tasks=[0]", "benchmark.max_steps=2", "benchmark.control_horizon=2",
            "diagnostics.enabled=true", "diagnostics.source_experiment_id=source",
            "diagnostics.num_video_frames=5", "diagnostics.max_probes_per_episode=1",
            "diagnostics.probe_strategy=first", "diagnostics.isolate_rng=false",
            "diagnostics.save_predicted_video=false", "diagnostics.save_actual_video=false",
            "diagnostics.save_side_by_side_video=false", "diagnostics.control_frequency_hz=20",
        ],
    )


def test_shadow_probe_cannot_mutate_executed_action_and_resume_is_global(tmp_path):
    cfg = _cfg(tmp_path)
    job = plan_jobs(cfg)[0]
    env, policy, probe = RecordingEnv(), MockPolicy(2), Probe()
    outcome = run_diagnostic_worker(
        cfg, policy=policy, environment=env, probe=probe, jobs=[job], close_resources=False
    )
    assert outcome == {"assigned": 1, "pending": 1, "completed": 1, "probes": 1, "skipped_by_resume": 0}
    assert all(action[0] != 999.0 for action in env.actions)
    row = json.loads((cfg.experiment.output_dir / "workers/rank_0/diagnostics.jsonl").read_text())
    assert row["action_unchanged"] is True
    assert row["num_video_frames"] == 5
    assert row["extra"]["probe_index"] == 0
    assert row["extra"]["playback_fps"] == 20.0
    assert row["extra"]["playback_fps_status"] == "exact"
    assert row["metrics"]["future_generation_latency_ms"] == 1.0
    assert row["metrics"]["diagnostic_latency_ms"] >= 0.0
    assert "environment stepping excluded" in row["metric_metadata"]["resources"][
        "diagnostic_latency_scope"
    ]
    assert probe.seeds == [job.episode_seed + cfg.diagnostics.diagnostic_seed_offset]

    # Move the completion to another rank: changed sharding must still resume it.
    fingerprint = diagnostic_protocol_fingerprint(cfg, {})
    DiagnosticArtifactWriter(cfg.experiment.output_dir, 1).mark_job_complete(
        job_id=job.job_id, status="completed", termination_reason="success", success=True,
        probe_count=1, protocol_fingerprint=fingerprint,
    )
    second_env = RecordingEnv()
    resumed = run_diagnostic_worker(
        cfg, policy=MockPolicy(2), environment=second_env, probe=Probe(), jobs=[job],
        close_resources=False,
    )
    assert resumed["pending"] == 0
    assert second_env.resets == 0


def test_diagnostics_on_and_off_execute_identical_action_hash(tmp_path):
    cfg_on = _cfg(tmp_path / "on")
    job = plan_jobs(cfg_on)[0]

    cfg_off = replace(
        cfg_on,
        experiment=replace(
            cfg_on.experiment,
            output_dir=tmp_path / "off" / "output",
        ),
        diagnostics=DiagnosticsConfig(),
    )
    off_environment = RecordingEnv()
    run_episode(
        cfg=cfg_off,
        job=job,
        policy=MockPolicy(2),
        environment=off_environment,
        worker_rank=0,
        provenance={},
        worker_dir=tmp_path / "off" / "worker",
    )

    on_environment = RecordingEnv()
    run_diagnostic_worker(
        cfg_on,
        policy=MockPolicy(2),
        environment=on_environment,
        probe=Probe(),
        jobs=[job],
        close_resources=False,
    )

    off_hash = action_chunk_hash(np.stack(off_environment.actions))
    on_hash = action_chunk_hash(np.stack(on_environment.actions))
    assert off_hash == on_hash


def test_capability_failure_precedes_reset_and_policy_act(tmp_path):
    cfg = _cfg(tmp_path)
    job = plan_jobs(cfg)[0]
    env = RecordingEnv()

    class BadProbe(Probe):
        def validate_capability(self):
            raise RuntimeError("unsupported")

    with pytest.raises(RuntimeError, match="unsupported"):
        run_diagnostic_worker(
            cfg, policy=MockPolicy(2), environment=env, probe=BadProbe(), jobs=[job],
            close_resources=False,
        )
    assert env.resets == 0


def test_probe_error_is_retryable_by_default_resume(tmp_path):
    cfg = _cfg(tmp_path)
    job = plan_jobs(cfg)[0]

    class FailingProbe(Probe):
        def predict_action_conditioned_future(self, *args, **kwargs):
            raise RuntimeError("synthetic infer_joint failure")

    first_environment = RecordingEnv()
    first = run_diagnostic_worker(
        cfg,
        policy=MockPolicy(2),
        environment=first_environment,
        probe=FailingProbe(),
        jobs=[job],
        close_resources=False,
    )
    assert first["completed"] == 1
    fingerprint = diagnostic_protocol_fingerprint(cfg, {})
    completion = load_all_completed_jobs(cfg.experiment.output_dir)[
        (job.job_id, fingerprint)
    ]
    assert completion["status"] == "error"
    assert completion["probe_error_count"] == 1

    second_environment = RecordingEnv()
    second = run_diagnostic_worker(
        cfg,
        policy=MockPolicy(2),
        environment=second_environment,
        probe=Probe(),
        jobs=[job],
        close_resources=False,
    )
    assert second["pending"] == 1
    assert second_environment.resets == 1
    latest = load_all_completed_jobs(cfg.experiment.output_dir)[
        (job.job_id, fingerprint)
    ]
    assert latest["status"] == "completed"
    assert latest["attempt_started_ns"] >= completion["attempt_started_ns"]


def test_first_strategy_means_only_first_replan(tmp_path):
    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        diagnostics=replace(cfg.diagnostics, max_probes_per_episode=2, probe_strategy="first"),
    )
    assert _probe_indices(cfg) == (0,)


def test_incomplete_action_conditioning_group_is_excluded(tmp_path):
    cfg = _cfg(tmp_path)
    job = plan_jobs(cfg)[0]
    run_diagnostic_worker(
        cfg,
        policy=MockPolicy(2),
        environment=RecordingEnv(),
        probe=Probe(group_size=4, decoded_frames_per_group=4),
        jobs=[job],
        close_resources=False,
    )
    row = json.loads((cfg.experiment.output_dir / "workers/rank_0/diagnostics.jsonl").read_text())
    assert row["status"] == "unavailable"
    assert row["approximate_alignment"] is True
    assert row["extra"]["aligned_future_frame_count"] == 0
    assert row["metrics"]["future_latent_l1"] is None
    future_frames = [
        frame for frame in row["alignment"]["frames"] if frame["predicted_frame_index"] > 0
    ]
    assert future_frames
    assert all(frame["action_conditioning_fully_executed"] is False for frame in future_frames)


def test_unconditional_future_uses_temporal_alignment_without_action_coverage_gate(tmp_path):
    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        diagnostics=replace(cfg.diagnostics, mode="unconditional_future"),
    )
    job = plan_jobs(cfg)[0]
    run_diagnostic_worker(
        cfg,
        policy=MockPolicy(2),
        environment=RecordingEnv(),
        probe=UnconditionalProbe(),
        jobs=[job],
        close_resources=False,
    )
    row = json.loads(
        (cfg.experiment.output_dir / "workers/rank_0/diagnostics.jsonl").read_text()
    )
    assert row["status"] == "completed"
    assert row["mode"] == "unconditional_future"
    assert row["action_conditioned_verified"] is False
    assert row["alignment"]["action_conditioning_coverage_complete"] is None
    assert row["alignment"]["action_dependency_scope"] == "not_applicable_unconditional"
    assert row["extra"]["aligned_future_frame_count"] == 2


def test_first_frame_causal_uses_full_transitive_action_dependency(tmp_path):
    cfg = _cfg(tmp_path)
    job = plan_jobs(cfg)[0]
    run_diagnostic_worker(
        cfg,
        policy=MockPolicy(2),
        environment=RecordingEnv(),
        probe=Probe(
            group_size=2,
            decoded_frames_per_group=2,
            attention_mode="first_frame_causal",
            action_horizon=4,
        ),
        jobs=[job],
        close_resources=False,
    )
    row = json.loads((cfg.experiment.output_dir / "workers/rank_0/diagnostics.jsonl").read_text())
    assert row["status"] == "unavailable"
    assert row["alignment"]["action_dependency_scope"] == "all_future_groups"
    future_frames = [
        frame for frame in row["alignment"]["frames"] if frame["predicted_frame_index"] > 0
    ]
    assert future_frames
    assert all(frame["direct_action_conditioning_action_end_exclusive"] == 2 for frame in future_frames[:2])
    assert all(frame["action_dependency_end_exclusive"] == 4 for frame in future_frames)
    assert all(frame["action_conditioning_fully_executed"] is False for frame in future_frames)


def test_source_jobs_are_filtered_without_changing_source_job_ids(tmp_path):
    source = tmp_path / "source"
    cfg = load_config(
        write_config(tmp_path / "cfg", perturbation=True, episodes=2),
        [
            "perturbation.category=[camera_viewpoints]", "perturbation.level=[easy]",
            "benchmark.tasks=[0]", "diagnostics.enabled=true",
            "diagnostics.source_experiment_id=source",
            f"diagnostics.source_output_dir={source}",
        ],
    )
    wanted = plan_jobs(cfg)
    distractors = [
        replace(wanted[0], job_id="a" * 24, perturbation_category="light_conditions"),
        replace(wanted[0], job_id="b" * 24, episode_index=99),
    ]
    write_jobs(source / "job_manifest.jsonl", [*wanted, *distractors])
    (source / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "source",
                "config": cfg.to_dict(),
                "job_count": len(wanted) + len(distractors),
            }
        ),
        encoding="utf-8",
    )
    selected = load_source_jobs(cfg)
    assert [job.job_id for job in selected] == [job.job_id for job in wanted]


def test_source_protocol_mismatch_is_rejected_before_job_use(tmp_path):
    source = tmp_path / "source"
    cfg = _cfg(tmp_path / "cfg")
    source_cfg = replace(
        cfg,
        benchmark=replace(cfg.benchmark, control_horizon=1),
    )
    source.mkdir()
    (source / "experiment_manifest.json").write_text(
        json.dumps({"experiment_id": "source", "config": source_cfg.to_dict()}),
        encoding="utf-8",
    )
    write_jobs(source / "job_manifest.jsonl", plan_jobs(cfg))
    cfg = replace(cfg, diagnostics=replace(cfg.diagnostics, source_output_dir=source))

    with pytest.raises(RuntimeError, match="benchmark.control_horizon differs"):
        load_source_jobs(cfg)


def test_source_provenance_mismatch_precedes_environment_reset(tmp_path):
    source = tmp_path / "source"
    cfg = _cfg(tmp_path / "cfg")
    job = plan_jobs(cfg)[0]
    source.mkdir()
    (source / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "source",
                "config": cfg.to_dict(),
                "provenance": {
                    "checkpoint_hash": "thought1-checkpoint",
                    "fastwam_commit": "fastwam-commit",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = replace(cfg, diagnostics=replace(cfg.diagnostics, source_output_dir=source))
    environment = RecordingEnv()

    with pytest.raises(RuntimeError, match="checkpoint_hash differs"):
        run_diagnostic_worker(
            cfg,
            policy=MockPolicy(2),
            environment=environment,
            probe=Probe(),
            jobs=[job],
            provenance={
                "checkpoint_hash": "different-checkpoint",
                "fastwam_commit": "fastwam-commit",
            },
            close_resources=False,
        )
    assert environment.resets == 0


def test_source_tree_is_read_only_and_output_is_isolated(tmp_path):
    source = tmp_path / "thought1"
    cfg = _cfg(tmp_path / "cfg")
    job = plan_jobs(cfg)[0]
    source.mkdir()
    (source / "experiment_manifest.json").write_text(
        json.dumps({"experiment_id": "source", "job_count": 1}), encoding="utf-8"
    )
    write_jobs(source / "job_manifest.jsonl", [job])
    cfg = replace(
        cfg,
        diagnostics=replace(cfg.diagnostics, source_output_dir=source),
    )
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    run_diagnostic_worker(
        cfg, policy=MockPolicy(2), environment=RecordingEnv(), probe=Probe(), jobs=[job],
        close_resources=False,
    )
    after = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    assert after == before
    assert (cfg.experiment.output_dir / "source_manifest.json").is_file()
    with pytest.raises(ValueError, match="must be disjoint"):
        DiagnosticArtifactWriter(source, source_output_dir=source)


def test_three_rank_shards_form_union_without_duplicate_jobs(tmp_path):
    cfg = _cfg(tmp_path)
    template = plan_jobs(cfg)[0]
    jobs = [
        replace(
            template,
            job_id=f"{index:016x}{index:08x}",
            episode_index=index,
            episode_seed=template.episode_seed + index,
        )
        for index in range(6)
    ]
    for rank in range(3):
        run_diagnostic_worker(
            cfg,
            policy=MockPolicy(2),
            environment=RecordingEnv(),
            probe=Probe(),
            jobs=jobs,
            rank=rank,
            world_size=3,
            close_resources=False,
        )
    fingerprint = diagnostic_protocol_fingerprint(cfg, {})
    completions = load_all_completed_jobs(cfg.experiment.output_dir)
    completed_ids = {job_id for job_id, value in completions if value == fingerprint}
    assert completed_ids == {job.job_id for job in jobs}
    recorded_ids = []
    for path in (cfg.experiment.output_dir / "workers").glob("rank_*/diagnostics.jsonl"):
        recorded_ids.extend(json.loads(line)["job_id"] for line in path.read_text().splitlines())
    assert sorted(recorded_ids) == sorted(job.job_id for job in jobs)
