from __future__ import annotations

from fastwam_ood_eval.schemas.episode_result import EpisodeResult


def test_episode_result_round_trip():
    values = dict(
        experiment_id="x",
        job_id="j",
        timestamp="now",
        git_commit=None,
        fastwam_commit=None,
        libero_commit=None,
        libero_plus_commit=None,
        checkpoint=None,
        checkpoint_hash=None,
        suite="s",
        task_id=0,
        task_name="t",
        episode_index=0,
        episode_seed=1,
        condition="clean",
        perturbation_category=None,
        perturbation_level=None,
        perturbation_parameters={},
        success=True,
        steps=2,
        termination_reason="success",
        policy_latency_mean_ms=1.0,
        policy_latency_p50_ms=1.0,
        policy_latency_p95_ms=1.0,
        warmup_latency_ms=1.0,
        action_chunk_shape=[2, 7],
        observation_image_shape=[32, 32, 3],
        episode_duration_s=0.1,
        gpu_peak_memory_mb=0.0,
        gpu_memory_allocated_mb=0.0,
        gpu_memory_reserved_mb=0.0,
        video_path=None,
        error=None,
    )
    result = EpisodeResult(**values)
    assert EpisodeResult.from_dict(result.to_dict()).to_dict() == result.to_dict()

