"""Real frozen-policy geometry-subspace intervention execution."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought4.action_intervention import (
    ActionSeedIdentity,
    DonorCandidate,
    build_deterministic_derangement,
    compare_action_chunks,
    donor_manifest,
    geometry_shuffle_hidden,
    validate_seed_identity,
)
from fastwam_ood_eval.thought4.config import (
    FP32_SUBSPACE_ARITHMETIC,
    Thought4Config,
)
from fastwam_ood_eval.thought4.feature_hooks import (
    ScopedVideoKVCacheCapture,
    ScopedVideoKVCacheReplacement,
    VideoKVCacheSpec,
)
from fastwam_ood_eval.thought4.geometry_subspace import (
    correct_reconstruction,
    geometry_coordinates,
    subspace_from_linear_weight,
)
from fastwam_ood_eval.thought4.probe_evaluation import episode_grouped_bootstrap
from fastwam_ood_eval.thought4.probe_models import linear_weight
from fastwam_ood_eval.thought4.real_runtime import (
    FrozenFastWAMRuntime,
    RenderedProbeSample,
)
from fastwam_ood_eval.thought4.schemas import sha256_canonical, sha256_file
from fastwam_ood_eval.thought4.video_feature_extractor import tensor_sha256


class InterventionRuntimeError(RuntimeError):
    """Raised when a real intervention loses its matched controls."""


def geometry_coordinate_condition_shift(
    probe_examples: Sequence[Any],
    *,
    selection: Mapping[str, Any],
    subspace: Any,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Measure held-out paired condition shifts inside the selected subspace."""

    import torch

    selected = [
        value
        for value in probe_examples
        if value.source == "A"
        and value.split == "test"
        and value.feature_key == selection["feature_key"]
    ]
    by_sample: dict[str, dict[str, Any]] = {}
    episode_by_sample: dict[str, str] = {}
    for value in selected:
        conditions = by_sample.setdefault(value.sample_id, {})
        if value.condition in conditions:
            raise InterventionRuntimeError(
                "duplicate selected feature for a sample/condition"
            )
        feature = torch.as_tensor(value.feature).detach().float().reshape(1, -1)
        conditions[value.condition] = geometry_coordinates(feature, subspace)[0]
        episode_by_sample[value.sample_id] = value.episode_id
    if not by_sample:
        raise InterventionRuntimeError(
            "selected intervention feature has no held-out examples"
        )
    required = {"clean", "camera", "lighting", "robot_init"}
    if any(set(values) != required for values in by_sample.values()):
        raise InterventionRuntimeError(
            "coordinate-shift panel lacks a frozen held-out condition"
        )
    condition_rows: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in ("camera", "lighting", "robot_init")
    }
    for sample_id, values in sorted(by_sample.items()):
        clean = values["clean"]
        clean_norm = float(clean.norm().item())
        for condition in condition_rows:
            delta = values[condition] - clean
            distance = float(delta.norm().item())
            condition_rows[condition].append(
                {
                    "sample_id": sample_id,
                    "episode_id": episode_by_sample[sample_id],
                    "coordinate_l2": distance,
                    "coordinate_shift_clean_ratio": distance
                    / max(clean_norm, 1e-12),
                }
            )
    summaries: dict[str, Any] = {}
    for offset, (condition, rows) in enumerate(condition_rows.items()):
        values = [float(row["coordinate_l2"]) for row in rows]
        episodes = [str(row["episode_id"]) for row in rows]
        summaries[condition] = {
            "exact_state_pair": condition in {"camera", "lighting"},
            "coordinate_l2_grouped_bootstrap": asdict(
                episode_grouped_bootstrap(
                    values,
                    episodes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + offset,
                )
            ),
            "coordinate_shift_clean_ratio_mean": sum(
                float(row["coordinate_shift_clean_ratio"]) for row in rows
            )
            / len(rows),
            "rows": rows,
        }
    camera_by_id = {
        row["sample_id"]: float(row["coordinate_l2"])
        for row in condition_rows["camera"]
    }
    lighting_by_id = {
        row["sample_id"]: float(row["coordinate_l2"])
        for row in condition_rows["lighting"]
    }
    sample_ids = sorted(by_sample)
    camera_minus_lighting = [
        camera_by_id[sample_id] - lighting_by_id[sample_id]
        for sample_id in sample_ids
    ]
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.geometry_coordinate_shift.v1",
        "feature_key": selection["feature_key"],
        "selection_split": "development",
        "evaluation_split": "test",
        "condition_summaries": summaries,
        "camera_minus_lighting_paired_grouped_bootstrap": asdict(
            episode_grouped_bootstrap(
                camera_minus_lighting,
                [episode_by_sample[sample_id] for sample_id in sample_ids],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 17,
            )
        ),
        "robot_init_exact_state_pair": False,
    }
    payload["result_sha256"] = sha256_canonical(payload)
    return payload


def _cache_spec_from_path(
    module_path: str, *, expected_calls: int
) -> VideoKVCacheSpec:
    parts = module_path.split(".")
    if (
        len(parts) != 4
        or parts[:2] != ["mot", "video_kv_cache"]
        or parts[3] not in {"k", "v"}
    ):
        raise InterventionRuntimeError(
            f"unsupported actual-cache replacement boundary: {module_path}"
        )
    try:
        layer_index = int(parts[2])
    except ValueError as exc:
        raise InterventionRuntimeError(
            f"invalid cache layer in boundary: {module_path}"
        ) from exc
    return VideoKVCacheSpec(layer_index, parts[3], expected_calls)


def _run_action(
    cfg: Thought4Config,
    runtime: FrozenFastWAMRuntime,
    sample: RenderedProbeSample,
    *,
    seed: int,
    replacement: tuple[VideoKVCacheSpec, Any] | None = None,
) -> Any:
    import torch

    runtime.upstream_cfg.seed = int(seed)

    def infer() -> Any:
        action, _images, future_frames = runtime.official._predict_action_chunk(
            obs=dict(sample.observation),
            task_description=sample.task_description,
            model=runtime.model,
            processor=runtime.processor,
            cfg=runtime.upstream_cfg,
            action_horizon=runtime.action_horizon,
            input_w=runtime.input_width,
            input_h=runtime.input_height,
            model_device=cfg.runtime.device,
        )
        if future_frames is not None:
            raise InterventionRuntimeError(
                "intervention inference unexpectedly returned future RGB"
            )
        return torch.as_tensor(action).detach().cpu()

    with torch.inference_mode():
        if replacement is None:
            return infer()
        spec, transform = replacement
        with ScopedVideoKVCacheReplacement(runtime.model.mot, spec, transform):
            return infer()


def _capture_raw_feature(
    cfg: Thought4Config,
    runtime: FrozenFastWAMRuntime,
    sample: RenderedProbeSample,
    *,
    spec: VideoKVCacheSpec,
    seed: int,
) -> Any:
    runtime.upstream_cfg.seed = int(seed)
    with ScopedVideoKVCacheCapture(
        runtime.model.mot, (spec,), to_cpu=True
    ) as capture:
        _run_action(cfg, runtime, sample, seed=seed)
    values = capture.captured[spec.name]
    if len(values) != 1:
        raise InterventionRuntimeError(
            f"source-A actual-cache capture produced {len(values)} tensors"
        )
    return values[0]


def _require_fp32_subspace_protocol(cfg: Thought4Config) -> None:
    if cfg.intervention.subspace_arithmetic != FP32_SUBSPACE_ARITHMETIC:
        raise InterventionRuntimeError(
            "the repaired intervention requires the frozen FP32/single-cast "
            "subspace arithmetic contract"
        )


def _bitwise_correct_reconstruction(
    hidden: Any,
    subspace: Any,
) -> tuple[Any, dict[str, Any]]:
    """Require FP32 reconstruction to recover captured BF16 exactly."""

    import torch

    if hidden.dtype != torch.bfloat16:
        raise InterventionRuntimeError(
            f"correct-control input must be BF16, got {hidden.dtype}"
        )
    correct = correct_reconstruction(hidden, subspace)
    bitwise_equal = (
        correct.output.shape == hidden.shape
        and correct.output.dtype == hidden.dtype
        and correct.output.device == hidden.device
        and torch.equal(correct.output, hidden)
    )
    metadata: dict[str, Any] = {
        "schema_version": "thought4.phase4.bitwise_correct_reconstruction.v1",
        "subspace_arithmetic": FP32_SUBSPACE_ARITHMETIC,
        "input_shape": list(hidden.shape),
        "input_dtype": str(hidden.dtype),
        "compute_dtype": str(torch.float32),
        "output_dtype": str(correct.output.dtype),
        "single_output_cast": True,
        "residual_reconstruction_max_abs": (
            correct.residual_reconstruction_error
        ),
        "input_sha256": tensor_sha256(hidden),
        "output_sha256": tensor_sha256(correct.output),
        "bitwise_equal_after_output_cast": bitwise_equal,
        "passed": bitwise_equal,
    }
    if not bitwise_equal:
        raise InterventionRuntimeError(
            "correct geometry reconstruction is not bitwise equal after the "
            "single BF16 output cast: "
            f"max_abs={correct.residual_reconstruction_error:.9g}, "
            f"input_dtype={hidden.dtype}, output_dtype={correct.output.dtype}"
        )
    return correct, metadata


def run_identity_replacement_smoke(
    cfg: Thought4Config,
    runtime: FrozenFastWAMRuntime,
    sample: RenderedProbeSample,
) -> dict[str, Any]:
    """Exercise the real source-A replacement path without a scientific effect."""

    import torch

    _require_fp32_subspace_protocol(cfg)
    # Exercise the direct action-consumed source-A replacement boundary that
    # formal Phase 4-C is allowed to select.
    module_path = f"mot.video_kv_cache.{cfg.backbone.video_layers[0]}.v"
    spec = _cache_spec_from_path(
        module_path,
        expected_calls=cfg.runtime.action_denoise_steps,
    )
    seed = int(cfg.intervention.action_seeds[0])
    raw = _capture_raw_feature(
        cfg,
        runtime,
        sample,
        spec=spec,
        seed=seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        int(cfg.experiment.seed) + 808
    )
    technical_weight = torch.randn(
        (3, raw.shape[-1]), generator=generator, dtype=torch.float32
    )
    technical_subspace = subspace_from_linear_weight(
        technical_weight,
        energy_threshold=cfg.intervention.rank_energy_threshold,
        max_rank=min(3, cfg.intervention.max_rank),
    )
    reconstructed, reconstruction_contract = (
        _bitwise_correct_reconstruction(raw, technical_subspace)
    )
    reconstruction_contract.update(
        {
            "technical_weight_sha256": tensor_sha256(technical_weight),
            "basis_sha256": tensor_sha256(technical_subspace.basis),
            "basis_dtype": str(technical_subspace.basis.dtype),
            "subspace_rank": technical_subspace.rank,
        }
    )
    reference = _run_action(cfg, runtime, sample, seed=seed)
    replay_rows = [
        compare_action_chunks(
            reference,
            _run_action(cfg, runtime, sample, seed=seed),
        )
        for _ in range(1, cfg.intervention.replay_floor_repeats)
    ]
    if not replay_rows:
        raise InterventionRuntimeError(
            "identity replacement smoke needs at least two replay runs"
        )
    replay_floor = max(
        replay_rows, key=lambda value: float(value["action_l2"])
    )
    replaced = _run_action(
        cfg,
        runtime,
        sample,
        seed=seed,
        replacement=(
            spec,
            lambda original, value=raw: value.to(
                device=original.device, dtype=original.dtype
            ),
        ),
    )
    identity_metrics = compare_action_chunks(reference, replaced)
    reconstructed_action = _run_action(
        cfg,
        runtime,
        sample,
        seed=seed,
        replacement=(
            spec,
            lambda original, value=reconstructed.output: value.to(
                device=original.device, dtype=original.dtype
            ),
        ),
    )
    reconstruction_metrics = compare_action_chunks(
        reference, reconstructed_action
    )
    replay_l2 = float(replay_floor["action_l2"])
    identity_l2 = float(identity_metrics["action_l2"])
    # Exact replay is expected.  The small absolute allowance only covers
    # nondeterministic low-level kernels when the measured replay floor is zero.
    tolerance = max(1e-6, replay_l2 * 2.0 + 1e-8)
    reconstruction_l2 = float(reconstruction_metrics["action_l2"])
    reconstruction_contract["action_replacement"] = reconstruction_metrics
    reconstruction_contract["allowed_action_l2"] = tolerance
    reconstruction_contract["action_replacement_passed"] = (
        reconstruction_l2 <= tolerance
    )
    reconstruction_contract["passed"] = (
        reconstruction_contract["passed"]
        and reconstruction_contract["action_replacement_passed"]
    )
    reconstruction_contract["contract_sha256"] = sha256_canonical(
        reconstruction_contract
    )
    passed = (
        identity_l2 <= tolerance and reconstruction_contract["passed"]
    )
    if not passed:
        raise InterventionRuntimeError(
            "identity or FP32-reconstructed source-A replacement exceeds the "
            "measured replay floor"
        )
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.identity_replacement_smoke.v2",
        "module_path": module_path,
        "hook_location": "forward_action_with_video_cache argument",
        "action_seed_identity": asdict(
            _seed_identity(cfg, runtime, sample, seed)
        ),
        "captured_feature_shape": list(raw.shape),
        "captured_feature_dtype": str(raw.dtype),
        "captured_feature_sha256": tensor_sha256(raw),
        "replay_floor": replay_floor,
        "identity_replacement": identity_metrics,
        "bf16_fp32_subspace_reconstruction": reconstruction_contract,
        "subspace_arithmetic": FP32_SUBSPACE_ARITHMETIC,
        "allowed_action_l2": tolerance,
        "passed": passed,
        "scientific_result": False,
    }
    payload["result_sha256"] = sha256_canonical(payload)
    return payload


def _seed_identity(
    cfg: Thought4Config,
    runtime: FrozenFastWAMRuntime,
    sample: RenderedProbeSample,
    seed: int,
) -> ActionSeedIdentity:
    import numpy as np
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (1, runtime.action_horizon, runtime.model.action_expert.action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype=runtime.model.torch_dtype)
    observation_hashes = {
        key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        for key, value in sorted(sample.observation.items())
        if hasattr(value, "shape")
    }
    scheduler = runtime.model.infer_action_scheduler
    raw_scheduler_config = getattr(scheduler, "config", {})
    try:
        scheduler_config = dict(raw_scheduler_config)
    except (TypeError, ValueError) as exc:
        raise InterventionRuntimeError(
            "action scheduler config cannot be frozen"
        ) from exc
    return ActionSeedIdentity(
        seed=int(seed),
        initial_noise_sha256=tensor_sha256(noise),
        denoise_schedule_sha256=sha256_canonical(
            {
                "steps": cfg.runtime.action_denoise_steps,
                "scheduler": type(scheduler).__name__,
                "scheduler_config": scheduler_config,
                "sigma_shift": runtime.upstream_cfg.EVALUATION.get(
                    "sigma_shift"
                ),
                "action_horizon": runtime.action_horizon,
                "rand_device": str(
                    runtime.upstream_cfg.EVALUATION.get("rand_device", "cpu")
                ),
            }
        ),
        checkpoint_sha256=cfg.backbone.checkpoint_sha256,
        preprocessing_sha256=sha256_canonical(
            {
                "observation_hashes": observation_hashes,
                "task_description": sample.task_description,
                "input_hw": [runtime.input_height, runtime.input_width],
                "dataset_stats_sha256": sha256_file(
                    cfg.backbone.dataset_stats_path
                ),
            }
        ),
    )


def run_geometry_subspace_intervention(
    cfg: Thought4Config,
    runtime: FrozenFastWAMRuntime,
    samples: Sequence[RenderedProbeSample],
    *,
    selection: Mapping[str, Any],
    linear_probe: Any,
    probe_examples: Sequence[Any],
) -> dict[str, Any]:
    """Intervene only on held-out Clean states with deterministic donors."""

    import torch

    _require_fp32_subspace_protocol(cfg)

    eligible = [
        sample
        for sample in samples
        if sample.condition == "clean" and sample.plan.identity.split == "test"
    ]
    if len(eligible) < 2:
        raise InterventionRuntimeError(
            "intervention needs at least two held-out Clean episodes"
        )
    by_id = {sample.plan.identity.sample_id: sample for sample in eligible}
    candidates = [
        DonorCandidate(
            sample_id=sample.plan.identity.sample_id,
            task_id=sample.plan.identity.task_id,
            episode_id=sample.plan.identity.episode_id,
            progress_bin=int(sample.plan.frame_index // 25),
        )
        for sample in eligible
    ]
    pairs = build_deterministic_derangement(
        candidates, seed=cfg.intervention.donor_seed
    )
    module_path = str(selection["module_path"])
    spec = _cache_spec_from_path(
        module_path,
        expected_calls=cfg.runtime.action_denoise_steps,
    )
    subspace = subspace_from_linear_weight(
        linear_weight(linear_probe),
        energy_threshold=cfg.intervention.rank_energy_threshold,
        max_rank=cfg.intervention.max_rank,
    )
    coordinate_shift = geometry_coordinate_condition_shift(
        probe_examples,
        selection=selection,
        subspace=subspace,
        bootstrap_replicates=cfg.probe.bootstrap_replicates,
        bootstrap_seed=cfg.probe.bootstrap_seed + 500,
    )
    capture_seed = cfg.intervention.action_seeds[0]
    raw_features = {
        sample_id: _capture_raw_feature(
            cfg,
            runtime,
            sample,
            spec=spec,
            seed=capture_seed,
        )
        for sample_id, sample in by_id.items()
    }
    rows: list[dict[str, Any]] = []
    above_floor = 0
    for pair in pairs:
        target = by_id[pair.target_sample_id]
        target_hidden = raw_features[pair.target_sample_id].to(
            cfg.runtime.device
        )
        donor_hidden = raw_features[pair.donor_sample_id].to(
            cfg.runtime.device
        )
        shuffled_hidden, intervention_metadata = geometry_shuffle_hidden(
            target_hidden, donor_hidden, subspace
        )
        if (
            abs(intervention_metadata["coordinate_norm_ratio"] - 1.0)
            > cfg.intervention.norm_ratio_tolerance
        ):
            raise InterventionRuntimeError(
                "norm-matched donor coordinate ratio exceeds tolerance"
            )
        correct, correct_contract = _bitwise_correct_reconstruction(
            target_hidden, subspace
        )
        for seed in cfg.intervention.action_seeds:
            identity_correct = _seed_identity(
                cfg, runtime, target, int(seed)
            )
            identity_shuffle = _seed_identity(
                cfg, runtime, target, int(seed)
            )
            validate_seed_identity(identity_correct, identity_shuffle)
            replay_first = _run_action(cfg, runtime, target, seed=int(seed))
            floor_rows = [
                compare_action_chunks(
                    replay_first,
                    _run_action(cfg, runtime, target, seed=int(seed)),
                )
                for _ in range(1, cfg.intervention.replay_floor_repeats)
            ]
            if not floor_rows:
                raise InterventionRuntimeError(
                    "replay_floor_repeats must be at least two"
                )
            floor_metrics = max(
                floor_rows, key=lambda value: float(value["action_l2"])
            )
            correct_action = _run_action(
                cfg,
                runtime,
                target,
                seed=int(seed),
                replacement=(
                    spec,
                    lambda original, value=correct.output: value.to(
                        device=original.device, dtype=original.dtype
                    ),
                ),
            )
            shuffled_action = _run_action(
                cfg,
                runtime,
                target,
                seed=int(seed),
                replacement=(
                    spec,
                    lambda original, value=shuffled_hidden: value.to(
                        device=original.device, dtype=original.dtype
                    ),
                ),
            )
            correct_replay = compare_action_chunks(replay_first, correct_action)
            correct_shuffle = compare_action_chunks(
                correct_action, shuffled_action
            )
            floor_l2 = float(floor_metrics["action_l2"])
            action_l2_threshold = max(1e-6, floor_l2 * 2.0 + 1e-8)
            if float(correct_replay["action_l2"]) > action_l2_threshold:
                raise InterventionRuntimeError(
                    "correct geometry reconstruction exceeds replay tolerance"
                )
            exceeds = (
                float(correct_shuffle["action_l2"]) > action_l2_threshold
            )
            above_floor += int(exceeds)
            row = {
                "target_sample_id": pair.target_sample_id,
                "donor_sample_id": pair.donor_sample_id,
                "target_episode_id": pair.target_episode_id,
                "donor_episode_id": pair.donor_episode_id,
                "action_seed_identity": asdict(identity_correct),
                "replay_floor": floor_metrics,
                "replay_floor_repeats": cfg.intervention.replay_floor_repeats,
                "correct_vs_unhooked": correct_replay,
                "correct_vs_shuffle": correct_shuffle,
                "correct_shuffle_exceeds_floor": exceeds,
                "action_l2_replay_tolerance": action_l2_threshold,
                "correct_reconstruction": correct_contract,
                **intervention_metadata,
            }
            row["row_sha256"] = sha256_canonical(row)
            rows.append(row)
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.geometry_intervention.v1",
        "selection": dict(selection),
        "donor_manifest": donor_manifest(
            pairs, seed=cfg.intervention.donor_seed
        ),
        "subspace": {
            "rank": subspace.rank,
            "explained_weight_energy": subspace.explained_weight_energy,
            "basis_sha256": tensor_sha256(subspace.basis),
            "basis_dtype": str(subspace.basis.dtype),
            "arithmetic": FP32_SUBSPACE_ARITHMETIC,
        },
        "geometry_coordinate_condition_shift": coordinate_shift,
        "rows": rows,
        "correct_shuffle_above_floor_count": above_floor,
        "comparison_count": len(rows),
        "eef_trajectory_change_available": False,
        "eef_trajectory_change_unavailable_reason": (
            "Phase 4 does not execute action chunks in the environment; OSC action "
            "deltas are not treated as recovered EEF trajectories."
        ),
        "all_backbone_parameters_frozen": True,
        "success_outcome_read": False,
        "future_rgb_read": False,
    }
    payload["result_sha256"] = sha256_canonical(payload)
    return payload
