"""Independent shadow runner for action-conditioned future diagnostics.

The ordinary policy action is copied and hashed *before* the optional probe is
called.  Only that protected copy is sent to the environment; probe outputs can
never replace it.  This module deliberately does not call or modify the Thought
1 episode runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.config import EvalConfig
from fastwam_ood_eval.diagnostics.artifact_writer import (
    DiagnosticArtifactWriter,
    action_chunk_hash,
    clone_action_chunk,
    diagnostic_id,
    load_all_completed_jobs,
)
from fastwam_ood_eval.diagnostics.future_probe import frame_to_rgb_uint8
from fastwam_ood_eval.diagnostics.metrics import build_metric_metadata, compute_future_metrics
from fastwam_ood_eval.diagnostics.protocol import FutureProbeOutput, SupportsFutureProbe
from fastwam_ood_eval.diagnostics.rng_isolation import RngIsolation
from fastwam_ood_eval.diagnostics.temporal_alignment import (
    TemporalAlignment,
    build_temporal_alignment,
)
from fastwam_ood_eval.envs.base import BaseBenchmarkEnv
from fastwam_ood_eval.evaluation.jobs import EvaluationJob, read_jobs, shard_jobs
from fastwam_ood_eval.evaluation.success_checker import is_episode_success
from fastwam_ood_eval.policy.base import BasePolicy
from fastwam_ood_eval.schemas.future_diagnostic_result import FutureDiagnosticResult

LOGGER = logging.getLogger(__name__)


def _as_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _as_jsonable(value.to_dict())
        return _as_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def diagnostic_protocol_fingerprint(
    cfg: EvalConfig,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint every setting that changes diagnostic identity or semantics."""

    diagnostic_config = _as_jsonable(cfg.diagnostics)
    source_files: dict[str, str] = {}
    if cfg.diagnostics.source_output_dir is not None:
        for name in ("experiment_manifest.json", "job_manifest.jsonl"):
            path = Path(cfg.diagnostics.source_output_dir) / name
            if path.is_file():
                source_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema": "thought2-shadow-diagnostic-v1",
        "diagnostics": diagnostic_config,
        "checkpoint": {
            "path": str(cfg.checkpoint.path) if cfg.checkpoint.path is not None else None,
            "model_name": cfg.checkpoint.model_name,
            "config_path": (
                str(cfg.checkpoint.config_path) if cfg.checkpoint.config_path is not None else None
            ),
            "dataset_stats_path": (
                str(cfg.checkpoint.dataset_stats_path)
                if cfg.checkpoint.dataset_stats_path is not None
                else None
            ),
            "hash": (provenance or {}).get("checkpoint_hash"),
        },
        "policy": _as_jsonable(cfg.policy),
        "benchmark": {
            "backend": cfg.benchmark.backend,
            "suite": cfg.benchmark.suite,
            "max_steps": cfg.benchmark.max_steps,
            "num_steps_wait": cfg.benchmark.num_steps_wait,
            "control_horizon": cfg.benchmark.control_horizon,
            "image_size": list(cfg.benchmark.image_size),
        },
        "fastwam_commit": (provenance or {}).get("fastwam_commit"),
        "source_files_sha256": source_files,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_source_jobs(cfg: EvalConfig) -> list[EvaluationJob]:
    """Read, but never create or modify, the source Thought 1 job manifest."""

    source = cfg.diagnostics.source_output_dir
    if source is None:
        raise ValueError("diagnostics.source_output_dir is required when jobs are not supplied")
    _validate_source_manifest_compatibility(cfg, Path(source))
    manifest = Path(source) / "job_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Source job manifest does not exist: {manifest}")
    jobs = read_jobs(manifest)
    tasks = None if cfg.benchmark.tasks is None else set(cfg.benchmark.tasks)
    categories = set(cfg.perturbation.categories)
    levels = set(cfg.perturbation.levels)
    expected_condition = "ood" if cfg.perturbation.enabled else "clean"
    selected = [
        job
        for job in jobs
        if job.suite == cfg.benchmark.suite
        and (tasks is None or job.task_id in tasks)
        and 0 <= job.episode_index < cfg.benchmark.episodes_per_task
        and job.condition == expected_condition
        and (
            not cfg.perturbation.enabled
            or (
                job.perturbation_category in categories
                and job.perturbation_level in levels
            )
        )
    ]
    if not selected:
        raise RuntimeError(
            "No source jobs match the diagnostic suite/task/episode/condition/perturbation filters"
        )
    return selected


def _validate_source_manifest_compatibility(cfg: EvalConfig, source: Path) -> None:
    """Fail closed when a source experiment cannot represent this diagnostic subset.

    A smoke diagnostic may intentionally shorten ``max_steps`` or select fewer
    tasks/episodes/perturbations, but it may not silently change the checkpoint,
    seed, observation/control protocol, or policy identity that produced the
    source jobs.
    """

    path = source / "experiment_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source experiment manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid source experiment manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid source experiment manifest object: {path}")
    expected_id = cfg.diagnostics.source_experiment_id
    if expected_id is not None and str(payload.get("experiment_id")) != expected_id:
        raise RuntimeError(
            "Configured source_experiment_id does not match source manifest: "
            f"configured={expected_id}, manifest={payload.get('experiment_id')}"
        )
    source_cfg = payload.get("config")
    if not isinstance(source_cfg, Mapping):
        raise RuntimeError(
            f"Source manifest lacks the full config required for compatibility checks: {path}"
        )

    errors: list[str] = []

    def section(name: str) -> Mapping[str, Any]:
        value = source_cfg.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"source config is missing section {name}")
            return {}
        return value

    source_experiment = section("experiment")
    source_checkpoint = section("checkpoint")
    source_policy = section("policy")
    source_benchmark = section("benchmark")
    source_perturbation = section("perturbation")

    def require_equal(label: str, source_value: Any, diagnostic_value: Any) -> None:
        if source_value != diagnostic_value:
            errors.append(
                f"{label} differs (source={source_value!r}, diagnostic={diagnostic_value!r})"
            )

    require_equal("experiment.seed", source_experiment.get("seed"), cfg.experiment.seed)
    for key, diagnostic_value in (
        ("path", str(cfg.checkpoint.path) if cfg.checkpoint.path is not None else None),
        ("model_name", cfg.checkpoint.model_name),
        (
            "config_path",
            str(cfg.checkpoint.config_path) if cfg.checkpoint.config_path is not None else None,
        ),
        (
            "dataset_stats_path",
            str(cfg.checkpoint.dataset_stats_path)
            if cfg.checkpoint.dataset_stats_path is not None
            else None,
        ),
    ):
        require_equal(f"checkpoint.{key}", source_checkpoint.get(key), diagnostic_value)
    for key, diagnostic_value in cfg.policy.__dict__.items():
        require_equal(f"policy.{key}", source_policy.get(key), diagnostic_value)
    for key, diagnostic_value in (
        ("backend", cfg.benchmark.backend),
        ("suite", cfg.benchmark.suite),
        ("suite_config", str(cfg.benchmark.suite_config)),
        ("num_steps_wait", cfg.benchmark.num_steps_wait),
        ("control_horizon", cfg.benchmark.control_horizon),
        ("image_size", list(cfg.benchmark.image_size)),
    ):
        require_equal(f"benchmark.{key}", source_benchmark.get(key), diagnostic_value)

    source_max_steps = source_benchmark.get("max_steps")
    if not isinstance(source_max_steps, int) or source_max_steps < cfg.benchmark.max_steps:
        errors.append(
            "benchmark.max_steps must cover the diagnostic subset "
            f"(source={source_max_steps!r}, diagnostic={cfg.benchmark.max_steps})"
        )
    source_episodes = source_benchmark.get("episodes_per_task")
    if not isinstance(source_episodes, int) or source_episodes < cfg.benchmark.episodes_per_task:
        errors.append(
            "benchmark.episodes_per_task must cover the diagnostic subset "
            f"(source={source_episodes!r}, diagnostic={cfg.benchmark.episodes_per_task})"
        )
    source_tasks = source_benchmark.get("tasks")
    if cfg.benchmark.tasks is not None and source_tasks not in (None, "all"):
        try:
            missing_tasks = set(cfg.benchmark.tasks) - {int(value) for value in source_tasks}
        except (TypeError, ValueError):
            missing_tasks = set(cfg.benchmark.tasks)
        if missing_tasks:
            errors.append(f"source benchmark.tasks does not cover {sorted(missing_tasks)}")

    require_equal(
        "perturbation.enabled",
        bool(source_perturbation.get("enabled", False)),
        cfg.perturbation.enabled,
    )
    require_equal(
        "perturbation.variant_selection",
        source_perturbation.get("variant_selection", "sample"),
        cfg.perturbation.variant_selection,
    )
    if cfg.perturbation.enabled:
        source_categories = set(source_perturbation.get("category", ()))
        source_levels = set(source_perturbation.get("level", ()))
        missing_categories = set(cfg.perturbation.categories) - source_categories
        missing_levels = set(cfg.perturbation.levels) - source_levels
        if missing_categories:
            errors.append(
                f"source perturbation categories do not cover {sorted(missing_categories)}"
            )
        if missing_levels:
            errors.append(f"source perturbation levels do not cover {sorted(missing_levels)}")

    if errors:
        raise RuntimeError(
            "Diagnostic/source protocol mismatch; source outputs remain read-only:\n- "
            + "\n- ".join(errors)
        )


def _validate_source_provenance(
    cfg: EvalConfig,
    provenance: Mapping[str, Any],
) -> None:
    """Require the live checkpoint/upstream identity to match recorded Thought 1 provenance."""

    source = cfg.diagnostics.source_output_dir
    if source is None or not provenance:
        return
    path = Path(source) / "experiment_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot verify source provenance from {path}") from exc
    source_provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    if not isinstance(source_provenance, Mapping):
        raise RuntimeError(f"Source manifest lacks provenance required for diagnostics: {path}")
    errors: list[str] = []
    for key in ("checkpoint_hash", "fastwam_commit"):
        source_value = source_provenance.get(key)
        current_value = provenance.get(key)
        if source_value in (None, "") or current_value in (None, ""):
            errors.append(
                f"{key} is unavailable (source={source_value!r}, current={current_value!r})"
            )
        elif source_value != current_value:
            errors.append(
                f"{key} differs (source={source_value!r}, current={current_value!r})"
            )
    if errors:
        raise RuntimeError(
            "Diagnostic/source provenance mismatch; refusing to compare different model identities:\n- "
            + "\n- ".join(errors)
        )


def validate_source_provenance(
    cfg: EvalConfig,
    provenance: Mapping[str, Any],
) -> None:
    """Public pre-environment wrapper for the read-only source identity gate."""

    _validate_source_provenance(cfg, provenance)


def _probe_indices(cfg: EvalConfig) -> tuple[int, ...]:
    diagnostics = cfg.diagnostics
    count = diagnostics.max_probes_per_episode
    if diagnostics.probe_strategy == "explicit_replan_indices":
        return tuple(diagnostics.explicit_replan_indices[:count])
    estimated_replans = max(
        1,
        math.ceil(cfg.benchmark.max_steps / max(1, cfg.benchmark.control_horizon)),
    )
    if diagnostics.probe_strategy == "first":
        return (0,)
    if count == 1:
        return (0,)
    if diagnostics.probe_strategy == "evenly_spaced":
        if estimated_replans == 1:
            return (0,)
        selected = {
            round(index * (estimated_replans - 1) / max(1, count - 1))
            for index in range(count)
        }
        return tuple(sorted(selected))
    raise ValueError(f"Unsupported diagnostic probe strategy: {diagnostics.probe_strategy}")


def _frame_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    if len(shape) == 5 and shape[0] == 1:  # [1,C,T,H,W]
        return [value[0, :, index] for index in range(shape[2])]
    if len(shape) == 4:
        if shape[0] in (1, 3, 4):  # [C,T,H,W]
            return [value[:, index] for index in range(shape[1])]
        return [value[index] for index in range(shape[0])]  # [T,C,H,W] or [T,H,W,C]
    return [value]


def _runtime_control_frequency(
    environment: BaseBenchmarkEnv,
    configured_hz: float | None,
) -> tuple[float | None, bool]:
    values: list[float] = []
    adapter_env = getattr(environment, "env", None)
    candidates = (
        environment,
        adapter_env,
        getattr(adapter_env, "env", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "control_freq", None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0 and numeric not in values:
            values.append(numeric)
    if len(values) > 1 and any(not math.isclose(values[0], item) for item in values[1:]):
        raise RuntimeError(f"Conflicting runtime environment control frequencies: {values}")
    runtime_hz = values[0] if values else None
    if configured_hz is not None and runtime_hz is not None and not math.isclose(
        float(configured_hz), runtime_hz, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError(
            "Configured diagnostic control frequency conflicts with runtime environment: "
            f"configured={configured_hz}, runtime={runtime_hz}"
        )
    if runtime_hz is not None:
        return runtime_hz, True
    return configured_hz, False


def _action_shape(actions: Any) -> list[int] | None:
    shape = getattr(actions, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    try:
        rows = list(actions)
    except TypeError:
        return None
    if not rows:
        return [0]
    try:
        return [len(rows), len(rows[0])]
    except TypeError:
        return [len(rows)]


def _diagnostic_seed(cfg: EvalConfig, job: EvaluationJob, probe_index: int) -> int:
    """Use probe ordinal, not sparse replanning index, for stable diagnostic seeds."""

    return int(job.episode_seed + cfg.diagnostics.diagnostic_seed_offset + probe_index)


def _action_video_frequency_ratio(
    cfg: EvalConfig,
    probe: SupportsFutureProbe,
    output: FutureProbeOutput | None,
) -> tuple[int | None, str | None]:
    metadata = getattr(output, "metadata", {}) or {}
    raw = metadata.get("action_video_freq_ratio")
    if raw is None:
        raw = getattr(probe, "action_video_freq_ratio", None)
    if raw is None:
        if cfg.benchmark.backend == "mock":
            return 1, None
        return None, "Future probe did not expose an upstream-derived action_video_freq_ratio"
    try:
        ratio = int(raw)
    except (TypeError, ValueError):
        return None, f"Invalid action_video_freq_ratio from future probe: {raw!r}"
    if isinstance(raw, bool) or ratio <= 0:
        return None, f"Invalid action_video_freq_ratio from future probe: {raw!r}"
    return ratio, None


def _action_conditioning_geometry(
    cfg: EvalConfig,
    probe: SupportsFutureProbe,
    output: FutureProbeOutput | None,
    *,
    ratio: int | None,
) -> tuple[int | None, int | None, str | None, int | None, str | None]:
    metadata = getattr(output, "metadata", {}) or {}
    group_raw = metadata.get(
        "action_conditioning_group_size",
        getattr(probe, "action_conditioning_group_size", None),
    )
    vae_raw = metadata.get(
        "vae_temporal_downsample_factor",
        getattr(probe, "vae_temporal_downsample_factor", None),
    )
    patch_raw = metadata.get(
        "video_dit_temporal_patch_size",
        getattr(probe, "video_dit_temporal_patch_size", None),
    )
    attention_mode_raw = metadata.get(
        "video_attention_mask_mode",
        getattr(probe, "video_attention_mask_mode", None),
    )
    action_horizon_raw = metadata.get(
        "action_horizon",
        getattr(probe, "action_horizon", None),
    )
    if group_raw is None or vae_raw is None or patch_raw is None:
        if cfg.benchmark.backend == "mock" and ratio is not None:
            return ratio, 1, "per_frame_causal", None, None
        return None, None, None, None, (
            "Future probe did not expose action-conditioning group size, VAE temporal factor, "
            "and video DiT temporal patch size"
        )
    try:
        group_size, vae_factor, patch_size = int(group_raw), int(vae_raw), int(patch_raw)
    except (TypeError, ValueError):
        return None, None, None, None, "Invalid action-conditioning geometry in probe metadata"
    if any(value <= 0 for value in (group_size, vae_factor, patch_size)):
        return None, None, None, None, "Action-conditioning geometry values must be positive"
    if patch_size != 1:
        return None, None, None, None, (
            "Temporal DiT patch size other than one has no proven decoded-frame dependency mapping"
        )
    attention_mode = str(attention_mode_raw or "")
    if not attention_mode and cfg.benchmark.backend == "mock":
        attention_mode = "per_frame_causal"
    if attention_mode not in {"per_frame_causal", "first_frame_causal", "bidirectional"}:
        return None, None, None, None, (
            f"Unsupported or missing video attention dependency mode: {attention_mode!r}"
        )
    action_horizon: int | None = None
    if action_horizon_raw is not None:
        try:
            action_horizon = int(action_horizon_raw)
        except (TypeError, ValueError):
            return None, None, None, None, "Invalid action_horizon in probe metadata"
        if action_horizon <= 0:
            return None, None, None, None, "action_horizon must be positive"
    if attention_mode in {"first_frame_causal", "bidirectional"} and action_horizon is None:
        return None, None, None, None, (
            f"{attention_mode} requires action_horizon to close transitive action dependencies"
        )
    return group_size, vae_factor, attention_mode, action_horizon, None


def _annotate_action_coverage(
    alignment: TemporalAlignment,
    *,
    executed_action_count: int,
    group_size: int,
    decoded_frames_per_group: int,
    video_attention_mask_mode: str,
    action_horizon: int | None,
) -> tuple[dict[str, Any], set[int]]:
    payload = alignment.to_dict()
    covered_indices: set[int] = {0}
    incomplete_aligned = False
    for frame in payload["frames"]:
        predicted_index = int(frame["predicted_frame_index"])
        if predicted_index == 0:
            frame.update(
                {
                    "action_conditioning_group": None,
                    "action_conditioning_action_start": None,
                    "action_conditioning_action_end_exclusive": None,
                    "direct_action_conditioning_action_start": None,
                    "direct_action_conditioning_action_end_exclusive": None,
                    "action_dependency_start": None,
                    "action_dependency_end_exclusive": None,
                    "action_conditioning_fully_executed": True,
                }
            )
            continue
        group = (predicted_index - 1) // decoded_frames_per_group
        direct_start = group * group_size
        direct_end = direct_start + group_size
        if video_attention_mask_mode == "per_frame_causal":
            dependency_start = 0
            dependency_end = direct_end
        else:
            if action_horizon is None:
                raise RuntimeError(
                    f"{video_attention_mask_mode} requires the complete action horizon"
                )
            dependency_start = 0
            dependency_end = action_horizon
        fully_executed = executed_action_count >= dependency_end
        frame.update(
            {
                "action_conditioning_group": group,
                "action_conditioning_action_start": dependency_start,
                "action_conditioning_action_end_exclusive": dependency_end,
                "direct_action_conditioning_action_start": direct_start,
                "direct_action_conditioning_action_end_exclusive": direct_end,
                "action_dependency_start": dependency_start,
                "action_dependency_end_exclusive": dependency_end,
                "action_conditioning_fully_executed": fully_executed,
            }
        )
        if fully_executed:
            covered_indices.add(predicted_index)
        else:
            incomplete_aligned = True
    payload.update(
        {
            "action_conditioning_group_size": group_size,
            "decoded_frames_per_action_conditioning_group": decoded_frames_per_group,
            "video_attention_mask_mode": video_attention_mask_mode,
            "action_dependency_scope": (
                "causal_prefix"
                if video_attention_mask_mode == "per_frame_causal"
                else "all_future_groups"
            ),
            "action_horizon": action_horizon,
            "required_executed_actions_for_first_future": (
                group_size
                if video_attention_mask_mode == "per_frame_causal"
                else action_horizon
            ),
            "action_conditioning_coverage_complete": not incomplete_aligned,
            "metric_aligned_frame_count": len(covered_indices & {
                int(frame["predicted_frame_index"]) for frame in payload["frames"]
            }),
        }
    )
    return payload, covered_indices


def _model_frame(probe: SupportsFutureProbe, observation: dict[str, Any], *, mock: bool) -> Any:
    converter = getattr(probe, "observation_to_model_frame", None)
    if callable(converter):
        return converter(observation)
    if not mock:
        raise RuntimeError(
            "A real future probe must implement observation_to_model_frame(); "
            "agent-view-only fallback is scientifically invalid"
        )
    panels: dict[str, Any] = {}
    for source, label in (
        ("agentview_image", "image"),
        ("robot0_eye_in_hand_image", "wrist_image"),
    ):
        frame = observation.get(source)
        if frame is None:
            continue
        try:
            panels[label] = frame[::-1, ::-1]
        except (TypeError, IndexError):
            panels[label] = frame
    if not panels:
        raise RuntimeError("Observation contains no image usable by the mock diagnostic probe")
    return panels if len(panels) > 1 else next(iter(panels.values()))


def _encode_embeddings(
    probe: SupportsFutureProbe,
    frames: Sequence[Any],
    *,
    mock: bool,
) -> tuple[Any, str]:
    encoder = getattr(probe, "encode_frame_embeddings", None)
    if callable(encoder):
        return encoder(frames), "reencoded_frame_embedding_without_temporal_context"
    if not mock:
        raise RuntimeError(
            "A real future probe must implement encode_frame_embeddings() for paired latent metrics"
        )
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Mock future metrics require numpy") from exc
    return np.stack([frame_to_rgb_uint8(frame) for frame in frames], axis=0), "mock_rgb_proxy"


def _aligned_metrics(
    *,
    cfg: EvalConfig,
    probe: SupportsFutureProbe,
    predicted_frames: Sequence[Any],
    actual_frames: Sequence[Any],
    latency_ms: float | None,
    peak_memory_mb: float | None,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    metadata = build_metric_metadata(static_motion_threshold=cfg.diagnostics.static_motion_threshold)
    metadata["causal_interpretation_allowed"] = False
    if len(predicted_frames) < 2 or len(actual_frames) < 2:
        metrics = {
            "future_latent_l1": None,
            "future_latent_cosine_distance": None,
            "predicted_motion_energy": None,
            "actual_motion_energy": None,
            "motion_energy_ratio": None,
            "motion_direction_cosine": None,
            "predicted_static": None,
            "actual_static": None,
            "static_future_flag": None,
            "future_generation_latency_ms": latency_ms,
            "future_generation_peak_memory_mb": peak_memory_mb,
            "diagnostic_latency_ms": latency_ms,
            "diagnostic_peak_memory_mb": peak_memory_mb,
        }
        metadata["availability"] = "unavailable_no_aligned_future_frame"
        return metrics, metadata, None, None
    predicted_embeddings, encoding_status = _encode_embeddings(
        probe, predicted_frames, mock=cfg.benchmark.backend == "mock"
    )
    actual_embeddings, actual_encoding_status = _encode_embeddings(
        probe, actual_frames, mock=cfg.benchmark.backend == "mock"
    )
    metrics = compute_future_metrics(
        predicted_embeddings,
        actual_embeddings,
        static_motion_threshold=cfg.diagnostics.static_motion_threshold,
        epsilon=cfg.diagnostics.motion_epsilon,
        generation_latency_ms=latency_ms,
        generation_peak_memory_mb=peak_memory_mb,
    )
    metrics.setdefault("static_future_flag", metrics.get("predicted_static"))
    metrics.setdefault("diagnostic_latency_ms", metrics.get("future_generation_latency_ms"))
    metrics.setdefault("diagnostic_peak_memory_mb", metrics.get("future_generation_peak_memory_mb"))
    metadata["predicted_encoding"] = encoding_status
    metadata["actual_encoding"] = actual_encoding_status
    return metrics, metadata, predicted_embeddings, actual_embeddings


def _call_probe(
    *,
    cfg: EvalConfig,
    probe: SupportsFutureProbe,
    observation: dict[str, Any],
    actions: Any,
    seed: int,
) -> FutureProbeOutput:
    context = RngIsolation(seed) if cfg.diagnostics.isolate_rng else nullcontext()
    with context:
        return probe.predict_action_conditioned_future(
            observation,
            actions,
            diagnostic_seed=seed,
            num_video_frames=cfg.diagnostics.num_video_frames,
            num_inference_steps=cfg.diagnostics.num_inference_steps,
        )


def validate_probe_capability(cfg: EvalConfig, probe: SupportsFutureProbe) -> bool:
    """Run the semantic gate before resetting an environment or asking for actions."""

    validator = getattr(probe, "validate_capability", None)
    if callable(validator):
        validator()
        if cfg.benchmark.backend != "mock":
            verification = getattr(probe, "checkpoint_verification", None)
            if not isinstance(verification, Mapping) or not (
                verification.get("action_conditioning_parameters_loaded_verified") is True
                and verification.get("action_conditioned_training_provenance_verified") is True
            ):
                raise RuntimeError(
                    "A real future probe must prove both checkpoint parameter loading and "
                    "matched action-conditioned training provenance"
                )
        return True
    if cfg.benchmark.backend != "mock":
        raise RuntimeError("A real future probe must provide validate_capability()")
    return False


def _probe_payload(
    *,
    cfg: EvalConfig,
    job: EvaluationJob,
    worker_rank: int,
    provenance: Mapping[str, Any],
    protocol_fingerprint: str,
    probe_index: int,
    replan_index: int,
    origin_env_step: int,
    seed: int,
    action_hash_before: str,
    action_hash_after: str,
    full_actions: Any,
    executed_actions: Sequence[Any],
    alignment: TemporalAlignment | Mapping[str, Any],
    metrics: Mapping[str, Any],
    metric_metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    latency_ms: float | None,
    peak_memory_mb: float | None,
    action_conditioned_verified: bool,
    status: str,
    error: str | None,
    probe_metadata: Mapping[str, Any] | None,
    generated_num_video_frames: int | None,
    aligned_future_frame_count: int,
    playback_fps: float | None,
    playback_fps_verified: bool,
    attempt_id: str,
    attempt_started_ns: int,
) -> dict[str, Any]:
    alignment_dict = alignment.to_dict() if hasattr(alignment, "to_dict") else dict(alignment)
    approximate = bool(
        not alignment_dict.get("exact_step_mapping", False)
        or alignment_dict.get("timestamp_status") != "exact"
        or not alignment_dict.get("action_conditioning_coverage_complete", False)
    )
    probe_id = diagnostic_id(
        f"{job.job_id}:{replan_index}:{seed}",
        protocol_fingerprint,
    )
    unchanged = action_hash_before == action_hash_after
    return {
        "schema_version": 1,
        "diagnostic_id": probe_id,
        "probe_id": probe_id,
        "experiment_id": cfg.experiment.name,
        "source_experiment_id": str(cfg.diagnostics.source_experiment_id),
        "job_id": job.job_id,
        "probe_index": probe_index,
        "replan_index": replan_index,
        "origin_env_step": origin_env_step,
        "environment_step": origin_env_step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "attempt_started_ns": attempt_started_ns,
        "worker_rank": worker_rank,
        "suite": job.suite,
        "task_id": job.task_id,
        "task_name": job.task_name,
        "episode_index": job.episode_index,
        "episode_seed": job.episode_seed,
        "initial_state_index": job.initial_state_index,
        "condition": job.condition,
        "perturbation_category": job.perturbation_category,
        "perturbation_level": job.perturbation_level,
        "perturbation_parameters": job.perturbation_parameters,
        "checkpoint": provenance.get("checkpoint"),
        "checkpoint_hash": provenance.get("checkpoint_hash"),
        "fastwam_commit": provenance.get("fastwam_commit"),
        "mode": cfg.diagnostics.mode,
        "action_conditioned_verified": bool(action_conditioned_verified),
        "causal_interpretation_allowed": False,
        "inference_seed": job.episode_seed,
        "diagnostic_seed": seed,
        "num_video_frames": generated_num_video_frames,
        "num_inference_steps": cfg.diagnostics.num_inference_steps,
        "action_hash": action_hash_before,
        "action_hash_before": action_hash_before,
        "action_hash_after": action_hash_after,
        "action_unchanged": unchanged,
        "action_chunk_shape": _action_shape(full_actions),
        "predicted_actions": _as_jsonable(full_actions),
        "executed_actions": _as_jsonable(list(executed_actions)),
        "executed_action_count": len(executed_actions),
        "alignment": alignment_dict,
        "approximate_alignment": approximate,
        "metrics": dict(metrics),
        "metric_metadata": dict(metric_metadata),
        "static_future_flag": metrics.get("static_future_flag", metrics.get("predicted_static")),
        "predicted_video_path": artifacts.get("predicted_video_path"),
        "actual_video_path": artifacts.get("actual_video_path"),
        "side_by_side_video_path": artifacts.get("side_by_side_video_path"),
        "latent_path": artifacts.get("latent_path"),
        "artifact_paths": dict(artifacts),
        "generation_latency_ms": latency_ms,
        "generation_peak_memory_mb": peak_memory_mb,
        "status": status,
        "error": error,
        "extra": {
            "protocol_fingerprint": protocol_fingerprint,
            "probe_index": probe_index,
            "probe_metadata": _as_jsonable(probe_metadata or {}),
            "aligned_future_frame_count": int(aligned_future_frame_count),
            "playback_fps": playback_fps,
            "playback_fps_status": (
                "exact" if playback_fps_verified else (
                    "approximate" if playback_fps is not None else "unavailable"
                )
            ),
        },
    }


def run_diagnostic_episode(
    *,
    cfg: EvalConfig,
    job: EvaluationJob,
    policy: BasePolicy,
    environment: BaseBenchmarkEnv,
    probe: SupportsFutureProbe,
    writer: DiagnosticArtifactWriter,
    worker_rank: int,
    provenance: Mapping[str, Any],
    protocol_fingerprint: str,
    action_conditioned_verified: bool,
) -> dict[str, Any]:
    """Execute one episode while buffering one durable row per selected probe."""

    attempt_started_ns = time.time_ns()
    attempt_id = hashlib.sha256(
        f"{job.job_id}:{worker_rank}:{attempt_started_ns}".encode("utf-8")
    ).hexdigest()[:24]

    if job.skip_reason:
        writer.mark_job_complete(
            job_id=job.job_id,
            status="skipped",
            termination_reason="skipped",
            success=False,
            probe_count=0,
            diagnostic_id_value=diagnostic_id(job.job_id, protocol_fingerprint),
            protocol_fingerprint=protocol_fingerprint,
            error=job.skip_reason,
            attempt_id=attempt_id,
            attempt_started_ns=attempt_started_ns,
        )
        return {"job_id": job.job_id, "success": False, "termination_reason": "skipped", "probes": 0}

    selected_replans = _probe_indices(cfg)
    probe_ordinals = {replan: index for index, replan in enumerate(selected_replans)}
    buffered: list[dict[str, Any]] = []
    steps = 0
    policy_steps = 0
    success = False
    termination = "max_steps"
    episode_error: str | None = None
    observation: dict[str, Any] | None = None
    control_frequency_hz: float | None = cfg.diagnostics.control_frequency_hz
    control_frequency_verified = False

    try:
        observation = environment.reset(job)
        control_frequency_hz, control_frequency_verified = _runtime_control_frequency(
            environment, cfg.diagnostics.control_frequency_hz
        )
        policy.reset(environment.task_description, seed=job.episode_seed)
        for _ in range(cfg.benchmark.num_steps_wait):
            step_result = environment.step([0, 0, 0, 0, 0, 0, -1])
            observation = step_result.observation
            steps += 1
            if is_episode_success(environment, step_result):
                success = True
                termination = "success"
                break

        replan_index = 0
        while not success and policy_steps < cfg.benchmark.max_steps:
            assert observation is not None
            output = policy.act(observation)
            execution_chunk = clone_action_chunk(output.actions)
            action_hash_before = action_chunk_hash(execution_chunk)
            all_actions = list(execution_chunk)
            if not all_actions:
                raise RuntimeError("Policy returned an empty action chunk")
            origin_env_step = steps
            probe_index = probe_ordinals.get(replan_index)
            selected = (
                probe_index is not None
                and len(buffered) < cfg.diagnostics.max_probes_per_episode
            )
            probe_output: FutureProbeOutput | None = None
            probe_error: str | None = None
            predicted_frames: list[Any] = []
            actual_by_offset: dict[int, Any] = {}
            diagnostic_elapsed_ms = 0.0
            seed = _diagnostic_seed(cfg, job, int(probe_index or 0))
            if selected:
                diagnostic_phase_started = time.perf_counter()
                try:
                    actual_by_offset[0] = _model_frame(
                        probe, observation, mock=cfg.benchmark.backend == "mock"
                    )
                    probe_output = _call_probe(
                        cfg=cfg,
                        probe=probe,
                        observation=observation,
                        actions=clone_action_chunk(execution_chunk),
                        seed=seed,
                    )
                    predicted_value = probe_output.model_space_predicted_frames
                    if predicted_value is None:
                        predicted_value = probe_output.predicted_frames
                    predicted_frames = _frame_sequence(predicted_value)
                    if not predicted_frames:
                        raise RuntimeError("Future probe returned no predicted frames")
                except Exception as exc:  # probe failure must never change control execution
                    probe_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
                    LOGGER.exception("Future probe failed for job=%s replan=%d", job.job_id, replan_index)
                finally:
                    diagnostic_elapsed_ms += (
                        time.perf_counter() - diagnostic_phase_started
                    ) * 1000.0

            ratio, ratio_error = _action_video_frequency_ratio(cfg, probe, probe_output)
            if selected and ratio_error is not None:
                probe_error = probe_error or ratio_error
            (
                group_size,
                decoded_frames_per_group,
                video_attention_mask_mode,
                action_horizon,
                geometry_error,
            ) = (
                _action_conditioning_geometry(cfg, probe, probe_output, ratio=ratio)
            )
            if selected and geometry_error is not None:
                probe_error = probe_error or geometry_error

            action_hash_after = action_chunk_hash(execution_chunk)
            if action_hash_after != action_hash_before:
                raise RuntimeError(
                    "Shadow future probe mutated the protected action chunk; refusing to execute it"
                )

            executed_actions: list[Any] = []
            segment_exception: Exception | None = None
            for action in all_actions[: cfg.benchmark.control_horizon]:
                if policy_steps >= cfg.benchmark.max_steps:
                    break
                try:
                    step_result = environment.step(action)
                except Exception as exc:  # retain any completed diagnostic segment for audit
                    segment_exception = exc
                    break
                executed_actions.append(clone_action_chunk(action))
                observation = step_result.observation
                steps += 1
                policy_steps += 1
                if selected and ratio is not None and len(executed_actions) % ratio == 0:
                    diagnostic_phase_started = time.perf_counter()
                    try:
                        actual_by_offset[len(executed_actions)] = _model_frame(
                            probe, observation, mock=cfg.benchmark.backend == "mock"
                        )
                    except Exception as exc:
                        if probe_error is None:
                            probe_error = (
                                f"actual_frame_error: {type(exc).__name__}: {exc}\n"
                                f"{traceback.format_exc(limit=12)}"
                            )
                    finally:
                        diagnostic_elapsed_ms += (
                            time.perf_counter() - diagnostic_phase_started
                        ) * 1000.0
                if is_episode_success(environment, step_result):
                    success = True
                    termination = "success"
                    break

            if selected:
                diagnostic_phase_started = time.perf_counter()
                latency_ms = (
                    float(probe_output.latency_ms) if probe_output is not None else None
                )
                peak_memory_mb = (
                    float(probe_output.gpu_peak_memory_mb) if probe_output is not None else None
                )
                if (
                    predicted_frames
                    and ratio is not None
                    and group_size is not None
                    and decoded_frames_per_group is not None
                    and video_attention_mask_mode is not None
                ):
                    base_alignment = build_temporal_alignment(
                        origin_env_step=origin_env_step,
                        action_video_freq_ratio=ratio,
                        predicted_frame_count=len(predicted_frames),
                        executed_action_count=len(executed_actions),
                        control_frequency_hz=control_frequency_hz,
                        control_frequency_verified=control_frequency_verified,
                    )
                    alignment, covered_indices = _annotate_action_coverage(
                        base_alignment,
                        executed_action_count=len(executed_actions),
                        group_size=group_size,
                        decoded_frames_per_group=decoded_frames_per_group,
                        video_attention_mask_mode=video_attention_mask_mode,
                        action_horizon=action_horizon,
                    )
                    aligned_predicted: list[Any] = []
                    aligned_actual: list[Any] = []
                    metric_frame_indices: list[int] = []
                    for frame in base_alignment.frames:
                        if frame.predicted_frame_index not in covered_indices:
                            continue
                        actual = actual_by_offset.get(frame.actual_env_step_offset)
                        if actual is None:
                            probe_error = probe_error or (
                                "Missing model-space actual observation for exact action offset "
                                f"{frame.actual_env_step_offset}"
                            )
                            continue
                        aligned_predicted.append(predicted_frames[frame.predicted_frame_index])
                        aligned_actual.append(actual)
                        metric_frame_indices.append(frame.predicted_frame_index)
                    alignment["metric_predicted_frame_indices"] = metric_frame_indices
                    alignment["metric_aligned_frame_count"] = len(metric_frame_indices)
                else:
                    alignment = {
                        "origin_env_step": origin_env_step,
                        "action_video_freq_ratio": ratio,
                        "predicted_frame_count": len(predicted_frames),
                        "executed_action_count": len(executed_actions),
                        "frames": [],
                        "aligned_frame_count": 0,
                        "has_aligned_future": False,
                        "exact_step_mapping": False,
                        "timestamp_status": "unavailable",
                        "action_conditioning_coverage_complete": False,
                        "action_conditioning_group_size": group_size,
                        "decoded_frames_per_action_conditioning_group": decoded_frames_per_group,
                        "video_attention_mask_mode": video_attention_mask_mode,
                        "action_horizon": action_horizon,
                        "truncated": True,
                        "unaligned_tail_action_steps": len(executed_actions),
                        "metric_predicted_frame_indices": [],
                    }
                    aligned_predicted = []
                    aligned_actual = []
                predicted_embeddings = actual_embeddings = None
                try:
                    metrics, metric_metadata, predicted_embeddings, actual_embeddings = _aligned_metrics(
                        cfg=cfg,
                        probe=probe,
                        predicted_frames=aligned_predicted,
                        actual_frames=aligned_actual,
                        latency_ms=latency_ms,
                        peak_memory_mb=peak_memory_mb,
                    )
                except Exception as exc:
                    probe_error = probe_error or (
                        f"metric_error: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
                    )
                    metrics = {
                        "future_latent_l1": None,
                        "future_latent_cosine_distance": None,
                        "predicted_motion_energy": None,
                        "actual_motion_energy": None,
                        "motion_energy_ratio": None,
                        "motion_direction_cosine": None,
                        "predicted_static": None,
                        "actual_static": None,
                        "static_future_flag": None,
                        "diagnostic_latency_ms": latency_ms,
                        "diagnostic_peak_memory_mb": peak_memory_mb,
                    }
                    metric_metadata = {
                        "availability": "error",
                        "reason": str(exc),
                        "causal_interpretation_allowed": False,
                    }
                try:
                    playback_fps = (
                        float(control_frequency_hz) / ratio
                        if control_frequency_hz is not None and ratio is not None
                        else None
                    )
                    artifacts = writer.write_probe_artifacts(
                        job_id=job.job_id,
                        replan_index=replan_index,
                        predicted_frames=predicted_frames,
                        actual_frames=aligned_actual,
                        side_by_side_predicted_frames=aligned_predicted,
                        fps=playback_fps,
                        predicted_latents=predicted_embeddings,
                        actual_latents=actual_embeddings,
                        save_predicted=cfg.diagnostics.save_predicted_video,
                        save_actual=cfg.diagnostics.save_actual_video,
                        save_side_by_side=cfg.diagnostics.save_side_by_side_video,
                        save_latents=cfg.diagnostics.save_latents,
                    )
                except Exception as exc:
                    probe_error = probe_error or (
                        f"artifact_error: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
                    )
                    artifacts = {
                        "predicted_video_path": None,
                        "actual_video_path": None,
                        "side_by_side_video_path": None,
                        "latent_path": None,
                    }
                diagnostic_elapsed_ms += (
                    time.perf_counter() - diagnostic_phase_started
                ) * 1000.0
                metrics["diagnostic_latency_ms"] = diagnostic_elapsed_ms
                resources = metric_metadata.setdefault("resources", {})
                if isinstance(resources, dict):
                    resources["diagnostic_latency_scope"] = (
                        "shadow preprocessing, future probe, actual-frame preprocessing, "
                        "paired metrics, and artifact writing; environment stepping excluded"
                    )
                payload = _probe_payload(
                    cfg=cfg,
                    job=job,
                    worker_rank=worker_rank,
                    provenance=provenance,
                    protocol_fingerprint=protocol_fingerprint,
                    probe_index=int(probe_index),
                    replan_index=replan_index,
                    origin_env_step=origin_env_step,
                    seed=seed,
                    action_hash_before=action_hash_before,
                    action_hash_after=action_hash_after,
                    full_actions=execution_chunk,
                    executed_actions=executed_actions,
                    alignment=alignment,
                    metrics=metrics,
                    metric_metadata=metric_metadata,
                    artifacts=artifacts,
                    latency_ms=latency_ms,
                    peak_memory_mb=peak_memory_mb,
                    action_conditioned_verified=action_conditioned_verified,
                    status=(
                        "error"
                        if probe_error
                        else ("completed" if len(aligned_actual) > 1 else "unavailable")
                    ),
                    error=probe_error,
                    probe_metadata=(probe_output.metadata if probe_output is not None else {}),
                    generated_num_video_frames=(len(predicted_frames) if predicted_frames else None),
                    aligned_future_frame_count=max(0, len(aligned_actual) - 1),
                    playback_fps=playback_fps,
                    playback_fps_verified=control_frequency_verified,
                    attempt_id=attempt_id,
                    attempt_started_ns=attempt_started_ns,
                )
                buffered.append(payload)

            if segment_exception is not None:
                raise segment_exception
            replan_index += 1
    except Exception as exc:  # episode failures become durable diagnostic outcomes
        termination = "exception"
        episode_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
        LOGGER.exception("Diagnostic job %s failed", job.job_id)

    for payload in buffered:
        payload["episode_success"] = bool(success)
        payload["success"] = bool(success)
        payload["termination_reason"] = termination
        if episode_error:
            payload["extra"]["episode_error"] = episode_error
            payload["error"] = "\n".join(
                value for value in (payload.get("error"), episode_error) if value
            )
            payload["status"] = "exception"
        result = FutureDiagnosticResult.from_dict(payload)
        writer.append_diagnostic(result)
    probe_error_count = sum(
        payload.get("status") in {"error", "exception"} for payload in buffered
    )
    completion_status = (
        "exception"
        if termination == "exception"
        else ("error" if probe_error_count else "completed")
    )
    completion_error = episode_error
    if completion_error is None and probe_error_count:
        completion_error = f"{probe_error_count} diagnostic probe(s) failed"
    writer.mark_job_complete(
        job_id=job.job_id,
        status=completion_status,
        termination_reason=termination,
        success=success,
        probe_count=len(buffered),
        diagnostic_id_value=diagnostic_id(job.job_id, protocol_fingerprint),
        protocol_fingerprint=protocol_fingerprint,
        error=completion_error,
        attempt_id=attempt_id,
        attempt_started_ns=attempt_started_ns,
        probe_error_count=probe_error_count,
    )
    return {
        "job_id": job.job_id,
        "success": success,
        "termination_reason": termination,
        "probes": len(buffered),
        "error": episode_error,
    }


def run_diagnostic_worker(
    cfg: EvalConfig,
    *,
    policy: BasePolicy,
    environment: BaseBenchmarkEnv,
    probe: SupportsFutureProbe,
    jobs: Iterable[EvaluationJob] | None = None,
    rank: int = 0,
    world_size: int = 1,
    provenance: Mapping[str, Any] | None = None,
    writer: DiagnosticArtifactWriter | None = None,
    close_resources: bool = True,
    rerun: str = "incomplete",
) -> dict[str, int]:
    """Run one deterministic diagnostic shard with per-job completion resume."""

    if not cfg.diagnostics.enabled:
        raise ValueError("Diagnostics are disabled in this configuration")
    if rerun not in {"incomplete", "failed", "all"}:
        raise ValueError("rerun must be incomplete, failed, or all")
    source_output = cfg.diagnostics.source_output_dir
    artifact_writer = writer or DiagnosticArtifactWriter(
        cfg.experiment.output_dir,
        rank,
        source_output_dir=source_output,
        fps=cfg.recording.fps,
    )
    prov = dict(provenance or {})
    fingerprint = diagnostic_protocol_fingerprint(cfg, prov)
    all_jobs = list(jobs) if jobs is not None else load_source_jobs(cfg)
    assigned = shard_jobs(all_jobs, rank, world_size)
    completed_count = 0
    probe_count = 0
    capability_verified = False
    try:
        # This gate intentionally precedes environment.reset() and policy.act().
        capability_verified = validate_probe_capability(cfg, probe)
        _validate_source_provenance(cfg, prov)
        artifact_writer.prepare_manifest(
            protocol_fingerprint=fingerprint,
            config=cfg.to_dict(),
            experiment_id=cfg.experiment.name,
            source_experiment_id=cfg.diagnostics.source_experiment_id,
            source_output_dir=source_output,
            resume=cfg.experiment.resume,
            overwrite=cfg.experiment.overwrite,
            provenance=prov,
            write=rank == 0,
            planned_job_count=len(all_jobs),
        )
        completed = (
            load_all_completed_jobs(artifact_writer.output_dir)
            if cfg.experiment.resume
            else {}
        )

        def should_run(job: EvaluationJob) -> bool:
            if cfg.experiment.overwrite or rerun == "all":
                return True
            previous = completed.get((job.job_id, fingerprint))
            if previous is None:
                return True
            if rerun == "failed":
                return previous.get("status") in {"error", "exception"} or previous.get(
                    "termination_reason"
                ) in {"exception", "max_steps"}
            return previous.get("status") not in {"completed", "skipped"}

        pending = [job for job in assigned if should_run(job)]
        pending_count = len(pending)
        for job in pending:
            outcome = run_diagnostic_episode(
                cfg=cfg,
                job=job,
                policy=policy,
                environment=environment,
                probe=probe,
                writer=artifact_writer,
                worker_rank=rank,
                provenance=prov,
                protocol_fingerprint=fingerprint,
                action_conditioned_verified=capability_verified,
            )
            completed_count += 1
            probe_count += int(outcome["probes"])
    finally:
        if close_resources:
            environment.close()
            policy.close()
    return {
        "assigned": len(assigned),
        "pending": pending_count,
        "completed": completed_count,
        "probes": probe_count,
        "skipped_by_resume": len(assigned) - pending_count,
    }


__all__ = [
    "diagnostic_protocol_fingerprint",
    "load_source_jobs",
    "run_diagnostic_episode",
    "run_diagnostic_worker",
    "validate_source_provenance",
    "validate_probe_capability",
]
