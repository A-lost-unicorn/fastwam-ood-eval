"""Confirmed real-model Phase 5 smoke and guarded pilot/formal entry points."""

from __future__ import annotations

import gc
import json
import os
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought5.artifacts import write_status_transition
from fastwam_ood_eval.thought5.camera_rays import two_camera_token_rays
from fastwam_ood_eval.thought5.checkpointing import (
    frozen_parameter_sha256,
    geoeq_state_dict,
    restore_geoeq_state,
    save_geoeq_checkpoint,
    tensor_state_sha256,
)
from fastwam_ood_eval.thought5.config import Thought5Config
from fastwam_ood_eval.thought5.geo_equiv_model import GeoEqAttachment
from fastwam_ood_eval.thought5.geo_targets import build_geometry_targets
from fastwam_ood_eval.thought5.pose_transforms import (
    pose_embedding_12,
    relative_clean_to_camera,
)
from fastwam_ood_eval.thought5.schemas import clean_project_commit, object_sha256
from fastwam_ood_eval.thought5.trainer import (
    GeoEqForwardBatch,
    gradient_report,
    matched_optimizer,
    paired_training_loss,
    weights_for_variant,
)


class Thought5RuntimeError(RuntimeError):
    pass


def _progress(stage: str, **values: Any) -> None:
    from datetime import datetime, timezone

    print(
        json.dumps(
            {
                "phase": "5",
                "stage": stage,
                "time": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _require_single_visible_cuda(cfg: Thought5Config) -> None:
    import torch

    if cfg.runtime.device != "cuda:0":
        raise Thought5RuntimeError("real Phase5 process requires logical cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise Thought5RuntimeError("CUDA is unavailable in the confirmed window")
    torch.cuda.set_device("cuda:0")


def _thought4_smoke_config(cfg: Thought5Config) -> Any:
    from fastwam_ood_eval.thought4.config import load_thought4_config

    source = load_thought4_config("configs/thought4/phase4_geometry_action_smoke_v8.yaml")
    return replace(
        source,
        experiment=replace(
            source.experiment,
            name=f"{cfg.experiment.name}_paired_render",
            output_dir=cfg.experiment.output_dir / "runtime" / "paired_render",
            seed=cfg.experiment.seed,
        ),
        runtime=replace(source.runtime, device=cfg.runtime.device),
        backbone=replace(
            source.backbone,
            checkpoint_path=cfg.backbone.checkpoint_path,
            checkpoint_sha256=cfg.backbone.checkpoint_sha256,
            dataset_stats_path=cfg.backbone.dataset_stats_path,
            fastwam_commit=cfg.backbone.fastwam_commit,
            frozen_parameter_sha256=cfg.backbone.frozen_parameter_sha256,
        ),
        cohort=replace(
            source.cohort,
            dataset_root=cfg.cohort.dataset_root,
            dataset_revision=cfg.cohort.dataset_revision,
            split_seed=cfg.cohort.split_seed,
        ),
    )


def _render_two_exact_pairs(cfg: Thought5Config) -> tuple[Any, list[Any]]:
    from fastwam_ood_eval.thought4.cohort import plan_base_states
    from fastwam_ood_eval.thought4.real_runtime import render_probe_samples

    t4_cfg = _thought4_smoke_config(cfg)
    plans = plan_base_states(t4_cfg.cohort, horizon=t4_cfg.probe.horizon)[:2]
    _progress("paired_render_started", base_states=len(plans))
    samples, _states = render_probe_samples(t4_cfg, plans)
    grouped: dict[str, dict[str, Any]] = {}
    for sample in samples:
        grouped.setdefault(sample.plan.identity.sample_id, {})[sample.condition] = sample
    if len(grouped) != 2:
        raise Thought5RuntimeError("smoke did not render exactly two base states")
    for values in grouped.values():
        if set(values) != {"clean", "camera", "lighting", "robot_init"}:
            raise Thought5RuntimeError("smoke condition panel is incomplete")
        clean_hash = values["clean"].rendered.record.simulator_state_sha256
        for condition in ("camera", "lighting"):
            if values[condition].rendered.record.simulator_state_sha256 != clean_hash:
                raise Thought5RuntimeError(f"{condition} is not exact-state paired")
            if not values[condition].rendered.record.exact_state_pair:
                raise Thought5RuntimeError(f"{condition} exact-state flag is false")
        if values["robot_init"].rendered.record.exact_state_pair:
            raise Thought5RuntimeError("robot-init incorrectly marked exact-state")
    return t4_cfg, samples


def _padded_actions(episode: Any, frame_index: int) -> tuple[Any, Any]:
    import torch

    values = np.asarray(episode.actions[frame_index : frame_index + 32], dtype=np.float32)
    output = np.zeros((32, 7), dtype=np.float32)
    count = min(32, len(values))
    if count:
        output[:count] = values[:count]
    padding = np.ones(32, dtype=bool)
    padding[:count] = False
    if count == 0:
        raise Thought5RuntimeError("smoke action target is empty")
    return torch.from_numpy(output), torch.from_numpy(padding)


def _target_mapping(target: Any, *, device: str) -> dict[str, Any]:
    import torch

    tokens = len(target.depth)
    relation = np.repeat(target.eef_object_translation.reshape(1, 3), tokens, axis=0)
    return {
        "depth": torch.from_numpy(target.depth).reshape(1, tokens, 1).to(device),
        "depth_relation": torch.from_numpy(target.depth_relation)
        .reshape(1, tokens, 1)
        .to(device),
        "point_camera": torch.from_numpy(target.points_camera).unsqueeze(0).to(device),
        "point_world": torch.from_numpy(target.points_world).unsqueeze(0).to(device),
        "eef_object_world": torch.from_numpy(relation).unsqueeze(0).to(device),
    }


def _build_smoke_batch(
    cfg: Thought5Config,
    t4_cfg: Any,
    runtime: Any,
    samples: Sequence[Any],
    *,
    noise_seed_offset: int = 0,
) -> GeoEqForwardBatch:
    import torch
    from fastwam_ood_eval.thought3.phase_c_smoke import _sample_training_t_on_cpu
    from fastwam_ood_eval.thought3.real_training import preprocess_current_action_target
    from fastwam_ood_eval.thought4.real_runtime import load_demonstration_episode

    official = runtime.official
    by_pair: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if sample.condition in {"clean", "camera"}:
            by_pair.setdefault(sample.plan.identity.sample_id, {})[sample.condition] = sample
    clean_latents: list[Any] = []
    camera_latents: list[Any] = []
    contexts: list[Any] = []
    masks: list[Any] = []
    actions: list[Any] = []
    action_pads: list[Any] = []
    clean_rays: list[Any] = []
    camera_rays: list[Any] = []
    clean_poses: list[Any] = []
    camera_poses: list[Any] = []
    clean_extrinsics: list[Any] = []
    camera_extrinsics: list[Any] = []
    clean_targets: dict[str, list[Any]] = {
        key: []
        for key in (
            "depth",
            "depth_relation",
            "point_camera",
            "point_world",
            "eef_object_world",
        )
    }
    camera_targets = {key: [] for key in clean_targets}
    valid_masks: list[Any] = []
    pair_ids: list[str] = []
    for pair_id, values in sorted(by_pair.items()):
        if set(values) != {"clean", "camera"}:
            raise Thought5RuntimeError("smoke Clean/Camera pair is incomplete")
        clean, camera = values["clean"], values["camera"]
        episode = load_demonstration_episode(
            cfg.cohort.dataset_root, clean.plan.episode_index
        )
        raw_action, action_pad = _padded_actions(episode, clean.plan.frame_index)
        raw_state = torch.from_numpy(official._extract_sim_state(dict(clean.observation)))
        target_action, proprio, processed_pad = preprocess_current_action_target(
            raw_action, raw_state, action_pad, processor=runtime.processor
        )
        prompt = official.DEFAULT_PROMPT.format(task=clean.task_description)
        with torch.no_grad():
            context, context_mask = runtime.model.encode_prompt(prompt)
            context, context_mask = runtime.model._append_proprio_to_context(
                context,
                context_mask,
                proprio.to(device=cfg.runtime.device, dtype=runtime.model.torch_dtype),
            )
        pair_latents: dict[str, Any] = {}
        pair_grids: dict[str, Any] = {}
        pair_targets: dict[str, Any] = {}
        pair_extrinsics: dict[str, np.ndarray] = {}
        for condition, sample in values.items():
            image, _proprio, _imgs = official._obs_to_model_input(
                dict(sample.observation),
                cfg=runtime.upstream_cfg,
                processor=runtime.processor,
                width=runtime.input_width,
                height=runtime.input_height,
                device=cfg.runtime.device,
                dtype=runtime.model.torch_dtype,
            )
            with torch.no_grad():
                latent = runtime.model._encode_input_image_latents_tensor(image)
            if tuple(latent.shape) != (1, 48, 1, 14, 28):
                raise Thought5RuntimeError("current latent shape changed")
            metadata = sample.rendered.record.camera
            grid = two_camera_token_rays(
                metadata.intrinsic,
                image_height=224,
                camera_width=224,
                token_height=7,
                tokens_per_camera_width=7,
            )
            geometry = sample.rendered.geometry_state
            extrinsic = np.asarray(metadata.extrinsic_camera_to_world, dtype=np.float64)
            target = build_geometry_targets(
                depth_map=sample.rendered.depth,
                ray_grid=grid,
                camera_to_world=extrinsic,
                eef_position_world=geometry["eef_position_world"],
                object_position_world=geometry["object_position_world"],
            )
            pair_latents[condition] = latent
            pair_grids[condition] = grid
            pair_targets[condition] = target
            pair_extrinsics[condition] = extrinsic
        clean_latents.append(pair_latents["clean"][0])
        camera_latents.append(pair_latents["camera"][0])
        contexts.append(context[0])
        masks.append(context_mask[0])
        actions.append(target_action)
        action_pads.append(processed_pad)
        clean_rays.append(torch.from_numpy(pair_grids["clean"].rays_camera))
        camera_rays.append(torch.from_numpy(pair_grids["camera"].rays_camera))
        identity = np.eye(4, dtype=np.float32)
        clean_poses.append(torch.from_numpy(pose_embedding_12(identity)))
        relative = relative_clean_to_camera(
            pair_extrinsics["clean"], pair_extrinsics["camera"]
        )
        camera_poses.append(torch.from_numpy(pose_embedding_12(relative)))
        clean_extrinsics.append(torch.from_numpy(pair_extrinsics["clean"].astype(np.float32)))
        camera_extrinsics.append(torch.from_numpy(pair_extrinsics["camera"].astype(np.float32)))
        for key, value in _target_mapping(pair_targets["clean"], device="cpu").items():
            clean_targets[key].append(value[0])
        for key, value in _target_mapping(pair_targets["camera"], device="cpu").items():
            camera_targets[key].append(value[0])
        valid_masks.append(
            torch.from_numpy(
                pair_targets["clean"].valid_mask & pair_targets["camera"].valid_mask
            )
        )
        pair_ids.append(pair_id)
    device = cfg.runtime.device
    dtype = runtime.model.torch_dtype
    target_action = torch.stack(actions).to(device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(
        cfg.experiment.seed + 101 + int(noise_seed_offset)
    )
    action_noise = torch.randn(
        target_action.shape, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    timestep = _sample_training_t_on_cpu(
        runtime.model.train_action_scheduler,
        cfg.experiment.seed + 102 + int(noise_seed_offset),
        device,
        dtype,
    ).repeat(len(pair_ids))
    noisy_action = runtime.model.train_action_scheduler.add_noise(
        target_action, action_noise, timestep
    )
    velocity = runtime.model.train_action_scheduler.training_target(
        target_action, action_noise, timestep
    )
    weight = runtime.model.train_action_scheduler.training_weight(timestep)
    return GeoEqForwardBatch(
        pair_ids=pair_ids,
        clean_current_latent=torch.stack(clean_latents).to(device=device, dtype=dtype),
        camera_current_latent=torch.stack(camera_latents).to(device=device, dtype=dtype),
        context=torch.stack(contexts).to(device=device, dtype=dtype),
        context_mask=torch.stack(masks).to(device=device, dtype=torch.bool),
        noisy_action=noisy_action,
        timestep_action=timestep,
        velocity_target=velocity,
        action_is_pad=torch.stack(action_pads).to(device=device, dtype=torch.bool),
        action_weight=weight,
        clean_rays=torch.stack(clean_rays).to(device=device, dtype=torch.float32),
        camera_rays=torch.stack(camera_rays).to(device=device, dtype=torch.float32),
        clean_pose_12=torch.stack(clean_poses).to(device=device, dtype=torch.float32),
        camera_pose_12=torch.stack(camera_poses).to(device=device, dtype=torch.float32),
        clean_camera_to_world=torch.stack(clean_extrinsics).to(device=device),
        camera_camera_to_world=torch.stack(camera_extrinsics).to(device=device),
        clean_geometry_target={
            key: torch.stack(values).to(device=device) for key, values in clean_targets.items()
        },
        camera_geometry_target={
            key: torch.stack(values).to(device=device) for key, values in camera_targets.items()
        },
        valid_mask=torch.stack(valid_masks).to(device=device, dtype=torch.bool),
    )


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _train_smoke_tracks(
    cfg: Thought5Config, attachment: GeoEqAttachment, batch: GeoEqForwardBatch
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    initial = _clone_state(geoeq_state_dict(attachment))
    states: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    steps = min(2, cfg.training.max_steps)
    for variant in ("B1", "G3"):
        restore_geoeq_state(attachment, initial)
        optimizer = matched_optimizer(
            attachment,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        rows: list[dict[str, Any]] = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, components = paired_training_loss(
                attachment,
                batch,
                variant=variant,
                weights=weights_for_variant(cfg.training.lambda_by_variant[variant]),
            )
            if not bool(torch.isfinite(loss)):
                raise Thought5RuntimeError(f"{variant} produced non-finite loss")
            loss.backward()
            gradients = gradient_report(attachment)
            if not all(group["finite"] for group in gradients.values()):
                raise Thought5RuntimeError(f"{variant} produced non-finite gradients")
            if gradients["lora"]["nonzero_elements"] <= 0:
                raise Thought5RuntimeError(f"{variant} LoRA gradient is disconnected")
            if variant == "B1" and any(
                gradients[group]["gradient_tensor_count"] != 0
                for group in ("geo_projector", "ray_pose_encoder")
            ):
                raise Thought5RuntimeError(
                    "B1 zero-weight auxiliary path unexpectedly received gradients"
                )
            if variant == "G3" and any(
                gradients[group]["nonzero_elements"] <= 0
                for group in ("geo_projector", "ray_pose_encoder")
            ):
                raise Thought5RuntimeError(
                    "G3 geometry or ray/pose gradient is disconnected"
                )
            optimizer.step()
            torch.cuda.synchronize(cfg.runtime.device)
            rows.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "components": {
                        key: float(value.detach().cpu()) for key, value in components.items()
                    },
                    "gradients": gradients,
                    "peak_memory_mib": torch.cuda.max_memory_allocated(
                        cfg.runtime.device
                    )
                    / 2**20,
                }
            )
            del loss, components
        states[variant] = _clone_state(geoeq_state_dict(attachment))
        results[variant] = {
            "steps": steps,
            "rows": rows,
            "final_state_sha256": tensor_state_sha256(states[variant]),
        }
        del optimizer
        torch.cuda.empty_cache()
        _progress("smoke_track_complete", variant=variant, steps=steps)
    return results, states


def _official_smoke_action(
    cfg: Thought5Config,
    runtime: Any,
    sample: Any,
    *,
    seed: int,
    scope: Any,
) -> tuple[Any, float, bool]:
    import torch

    runtime.upstream_cfg.seed = int(seed)
    started = time.perf_counter()
    with torch.inference_mode(), scope:
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
    torch.cuda.synchronize(cfg.runtime.device)
    value = torch.as_tensor(action).detach().float().cpu().contiguous()
    if tuple(value.shape) != (32, 7) or not bool(torch.isfinite(value).all()):
        raise Thought5RuntimeError("smoke action inference returned invalid action")
    return value, (time.perf_counter() - started) * 1000.0, future_frames is None


def run_real_smoke(cfg: Thought5Config, *, resume: bool) -> dict[str, Any]:
    import torch
    from fastwam_ood_eval.thought4.real_runtime import (
        load_frozen_fastwam,
        release_fastwam,
    )

    _require_single_visible_cuda(cfg)
    project_commit = clean_project_commit()
    output = cfg.experiment.output_dir
    status_path = output / "run_status.json"
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            raise Thought5RuntimeError("completed smoke output is immutable")
        if prior.get("project_commit") not in {None, project_commit}:
            raise Thought5RuntimeError(
                "partial smoke belongs to a different project commit"
            )
        if not resume:
            raise Thought5RuntimeError("partial smoke exists; inspect and pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_status_transition(
        status_path,
        {
            "schema_version": "thought5.phase5.run_status.v1",
            "status": "running",
            "stage": "smoke",
            "config_fingerprint": cfg.fingerprint,
            "project_commit": project_commit,
            "scientific_result": False,
        },
    )
    started = time.perf_counter()
    runtime = None
    attachment = None
    try:
        t4_cfg, samples = _render_two_exact_pairs(cfg)
        _progress("model_load_started", device=cfg.runtime.device)
        torch.cuda.reset_peak_memory_stats(cfg.runtime.device)
        runtime = load_frozen_fastwam(t4_cfg)
        load_peak = torch.cuda.max_memory_allocated(cfg.runtime.device) / 2**20
        release_parameter_sha256 = frozen_parameter_sha256(
            runtime.model.named_parameters()
        )
        if release_parameter_sha256 != cfg.backbone.frozen_parameter_sha256:
            raise Thought5RuntimeError(
                "loaded release parameter SHA differs before GeoEq attachment"
            )
        clean_sample = next(sample for sample in samples if sample.condition == "clean")
        action_seed = cfg.experiment.seed + 8801
        baseline_action, baseline_latency_ms, baseline_no_future = (
            _official_smoke_action(
                cfg,
                runtime,
                clean_sample,
                seed=action_seed,
                scope=nullcontext(),
            )
        )
        attachment = GeoEqAttachment(
            runtime.model,
            lora_targets=cfg.method.lora_targets,
            lora_rank=cfg.method.lora_rank,
            lora_alpha=cfg.method.lora_alpha,
            lora_dropout=cfg.method.lora_dropout,
            projector_hidden_dim=cfg.method.geo_projector_hidden_dim,
            ray_pose_hidden_dim=cfg.method.ray_pose_hidden_dim,
        )
        wrapped_action, wrapped_latency_ms, wrapped_no_future = (
            _official_smoke_action(
                cfg,
                runtime,
                clean_sample,
                seed=action_seed,
                scope=nullcontext(),
            )
        )
        if not torch.equal(baseline_action, wrapped_action):
            raise Thought5RuntimeError(
                "zero-init GeoEq attachment changed baseline action output"
            )
        parameter_manifest = attachment.parameter_manifest()
        write_status_transition(
            output / "trainable_parameter_manifest.json", parameter_manifest
        )
        frozen_before = frozen_parameter_sha256(runtime.model.named_parameters())
        batch = _build_smoke_batch(cfg, t4_cfg, runtime, samples)
        torch.cuda.reset_peak_memory_stats(cfg.runtime.device)
        training_results, states = _train_smoke_tracks(cfg, attachment, batch)
        frozen_after = frozen_parameter_sha256(runtime.model.named_parameters())
        if frozen_before != frozen_after:
            raise Thought5RuntimeError("frozen Fast-WAM parameters changed")
        checkpoints: dict[str, str] = {}
        for variant in ("B1", "G3"):
            restore_geoeq_state(attachment, states[variant])
            path = save_geoeq_checkpoint(
                output / "checkpoints" / variant.lower(),
                attachment=attachment,
                variant=variant,
                global_step=training_results[variant]["steps"],
                config_fingerprint=cfg.fingerprint,
                cohort_fingerprint=object_sha256(sorted(batch.pair_ids)),
                backbone_checkpoint_sha256=cfg.backbone.checkpoint_sha256,
                frozen_before_sha256=frozen_before,
                frozen_after_sha256=frozen_after,
            )
            checkpoints[variant] = str(path)
        restore_geoeq_state(attachment, states["G3"])
        if "geo_projector" in attachment.inference_modules():
            raise Thought5RuntimeError("GeoProjector leaked into inference graph")
        camera_sample = next(
            sample
            for sample in samples
            if sample.condition == "camera"
            and sample.plan.identity.sample_id
            == clean_sample.plan.identity.sample_id
        )
        from fastwam_ood_eval.thought5.future_runtime import current_conditioning

        rays, pose = current_conditioning(camera_sample, clean_sample)
        projector_calls = 0

        def count_projector(_module: Any, _inputs: Any, _output: Any) -> None:
            nonlocal projector_calls
            projector_calls += 1

        projector_handle = attachment.geo_projector.register_forward_hook(
            count_projector
        )
        try:
            geoeq_action, geoeq_latency_ms, geoeq_no_future = (
                _official_smoke_action(
                    cfg,
                    runtime,
                    camera_sample,
                    seed=action_seed,
                    scope=attachment.conditioning(
                        rays=rays,
                        camera_pose_12=pose,
                        enable_ray_pose=True,
                    ),
                )
            )
        finally:
            projector_handle.remove()
        if projector_calls != 0 or not geoeq_no_future:
            raise Thought5RuntimeError(
                "training-only projector/future RGB leaked into inference"
            )
        result = {
            "schema_version": "thought5.phase5.smoke_result.v1",
            "status": "complete",
            "scientific_result": False,
            "project_commit": project_commit,
            "base_states": 2,
            "exact_state_pairs": 2,
            "load_peak_mib": load_peak,
            "training_peak_mib": torch.cuda.max_memory_allocated(cfg.runtime.device)
            / 2**20,
            "frozen_parameter_sha256_before": frozen_before,
            "frozen_parameter_sha256_after": frozen_after,
            "release_parameter_sha256_before_attachment": (
                release_parameter_sha256
            ),
            "trainable_parameter_count": parameter_manifest[
                "trainable_parameter_count"
            ],
            "matched_parameter_budget": True,
            "training": training_results,
            "checkpoints": checkpoints,
            "inference_inputs": ["current_rgb", "proprio", "intrinsic", "extrinsic"],
            "inference_uses_gt_depth": False,
            "auxiliary_projector_removed": True,
            "inference_smoke": {
                "baseline_zero_lora_bitwise": True,
                "baseline_action_sha256": object_sha256(
                    baseline_action.tolist()
                ),
                "wrapped_action_sha256": object_sha256(
                    wrapped_action.tolist()
                ),
                "geoeq_action_sha256": object_sha256(
                    geoeq_action.tolist()
                ),
                "baseline_latency_ms": baseline_latency_ms,
                "zero_lora_wrapped_latency_ms": wrapped_latency_ms,
                "geoeq_camera_latency_ms": geoeq_latency_ms,
                "ray_pose_added_latency_ms": geoeq_latency_ms
                - wrapped_latency_ms,
                "future_rgb_absent": bool(
                    baseline_no_future and wrapped_no_future and geoeq_no_future
                ),
                "geo_projector_forward_calls": projector_calls,
                "gt_depth_read": False,
            },
            "elapsed_s": time.perf_counter() - started,
            "pilot_unlocked": True,
        }
        write_status_transition(output / "smoke_result.json", result)
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.run_status.v1",
                "status": "complete",
                "stage": "smoke",
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "scientific_result": False,
                "result": str(output / "smoke_result.json"),
            },
        )
        _progress("smoke_complete", result=str(output / "smoke_result.json"))
        return result
    except BaseException as exc:
        write_status_transition(
            status_path,
            {
                "schema_version": "thought5.phase5.run_status.v1",
                "status": "error",
                "stage": "smoke",
                "config_fingerprint": cfg.fingerprint,
                "project_commit": project_commit,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if attachment is not None:
            attachment.close()
        if runtime is not None:
            release_fastwam(runtime)
        gc.collect()


def run_real_stage(cfg: Thought5Config, *, resume: bool = False) -> dict[str, Any]:
    if cfg.experiment.stage == "smoke":
        return run_real_smoke(cfg, resume=resume)
    if cfg.experiment.stage == "pilot":
        from fastwam_ood_eval.thought5.panel_runtime import run_pilot

        return run_pilot(cfg, resume=resume)
    if cfg.experiment.stage == "formal":
        from fastwam_ood_eval.thought5.panel_runtime import run_formal

        return run_formal(cfg, resume=resume)
    raise Thought5RuntimeError(f"unsupported real stage: {cfg.experiment.stage}")
