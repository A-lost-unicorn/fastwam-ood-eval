"""Run and record one evaluation episode."""

from __future__ import annotations

import logging
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastwam_ood_eval.config import EvalConfig
from fastwam_ood_eval.envs.base import BaseBenchmarkEnv
from fastwam_ood_eval.evaluation.jobs import EvaluationJob
from fastwam_ood_eval.evaluation.success_checker import is_episode_success
from fastwam_ood_eval.policy.base import BasePolicy
from fastwam_ood_eval.recording.episode_trace import EpisodeTrace
from fastwam_ood_eval.recording.video_recorder import VideoRecorder
from fastwam_ood_eval.schemas.episode_result import EpisodeResult

LOGGER = logging.getLogger(__name__)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _robot_state(observation: dict[str, Any]) -> Any:
    if "robot_state" in observation:
        return observation["robot_state"]
    keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    return {key: observation.get(key) for key in keys if key in observation}


def run_episode(
    *,
    cfg: EvalConfig,
    job: EvaluationJob,
    policy: BasePolicy,
    environment: BaseBenchmarkEnv,
    worker_rank: int,
    provenance: dict[str, Any],
    worker_dir: Path,
) -> EpisodeResult:
    started = time.perf_counter()
    latencies: list[float] = []
    warmup_latency: float | None = None
    action_shape: list[int] | None = None
    observation_shape: list[int] | None = None
    allocated = reserved = peak = 0.0
    steps = 0
    success = False
    error: str | None = None
    termination = "max_steps"
    video_path: str | None = None
    trace_path = worker_dir / "traces" / f"{job.job_id}.jsonl"
    if not (cfg.recording.save_actions or cfg.recording.save_robot_state):
        trace_path = None
    recorder = VideoRecorder(
        enabled=cfg.experiment.save_video,
        fps=cfg.recording.fps,
        output_path=worker_dir / "videos" / f"{job.job_id}.{cfg.recording.video_format}",
    )

    if job.skip_reason:
        termination = "skipped"
        error = job.skip_reason
    else:
        try:
            observation = environment.reset(job)
            policy.reset(environment.task_description, seed=job.episode_seed)
            recorder.add_observation(observation)
            with EpisodeTrace(trace_path) as trace:
                for _ in range(cfg.benchmark.num_steps_wait):
                    step_result = environment.step([0, 0, 0, 0, 0, 0, -1])
                    observation = step_result.observation
                    steps += 1
                    recorder.add_observation(observation)
                    if is_episode_success(environment, step_result):
                        success = True
                        termination = "success"
                        break
                policy_steps = 0
                while not success and policy_steps < cfg.benchmark.max_steps:
                    output = policy.act(observation)
                    latencies.append(output.latency_ms)
                    if warmup_latency is None and output.warmup_latency_ms is not None:
                        warmup_latency = output.warmup_latency_ms
                    action_shape = output.action_chunk_shape
                    observation_shape = output.observation_image_shape
                    allocated = max(allocated, output.gpu_memory_allocated_mb)
                    reserved = max(reserved, output.gpu_memory_reserved_mb)
                    actions = list(output.actions)[: cfg.benchmark.control_horizon]
                    if not actions:
                        raise RuntimeError("Policy returned an empty action chunk")
                    for action in actions:
                        if policy_steps >= cfg.benchmark.max_steps:
                            break
                        trace.record(
                            steps,
                            action=action if cfg.recording.save_actions else None,
                            robot_state=_robot_state(observation) if cfg.recording.save_robot_state else None,
                        )
                        step_result = environment.step(action)
                        observation = step_result.observation
                        steps += 1
                        policy_steps += 1
                        recorder.add_observation(observation)
                        if is_episode_success(environment, step_result):
                            success = True
                            termination = "success"
                            break
            peak_method = getattr(policy, "peak_memory_mb", None)
            peak = float(peak_method()) if callable(peak_method) else allocated
        except Exception as exc:  # noqa: BLE001 - episode errors must become durable results
            termination = "exception"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
            LOGGER.exception("Job %s failed", job.job_id)

    should_save = cfg.experiment.save_video and (
        not cfg.experiment.save_failure_video_only or not success
    )
    if should_save:
        try:
            video_path = recorder.save()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not save video for %s: %s", job.job_id, exc)
            if error is None:
                error = f"video_error: {type(exc).__name__}: {exc}"
    else:
        recorder.discard()

    return EpisodeResult(
        experiment_id=job.experiment_id,
        job_id=job.job_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=provenance.get("git_commit"),
        fastwam_commit=provenance.get("fastwam_commit"),
        libero_commit=provenance.get("libero_commit"),
        libero_plus_commit=provenance.get("libero_plus_commit"),
        checkpoint=provenance.get("checkpoint"),
        checkpoint_hash=provenance.get("checkpoint_hash"),
        suite=job.suite,
        task_id=job.task_id,
        task_name=job.task_name,
        episode_index=job.episode_index,
        episode_seed=job.episode_seed,
        condition=job.condition,
        perturbation_category=job.perturbation_category,
        perturbation_level=job.perturbation_level,
        perturbation_parameters=job.perturbation_parameters,
        success=success,
        steps=steps,
        termination_reason=termination,
        policy_latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        policy_latency_p50_ms=percentile(latencies, 0.50),
        policy_latency_p95_ms=percentile(latencies, 0.95),
        warmup_latency_ms=warmup_latency,
        action_chunk_shape=action_shape,
        observation_image_shape=observation_shape,
        episode_duration_s=time.perf_counter() - started,
        gpu_peak_memory_mb=peak,
        gpu_memory_allocated_mb=allocated,
        gpu_memory_reserved_mb=reserved,
        video_path=video_path,
        error=error,
        worker_rank=worker_rank,
        status="skipped" if termination == "skipped" else "completed",
    )

