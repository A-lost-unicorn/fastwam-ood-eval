"""Protocol freezing, CPU contract dry-run, and NOT-RUN artifact scaffold."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fastwam_ood_eval.thought5.artifacts import (
    build_artifact_manifest,
    execution_integrity,
    write_status_transition,
)
from fastwam_ood_eval.thought5.camera_rays import (
    assert_fastwam_token_layout,
    two_camera_token_rays,
)
from fastwam_ood_eval.thought5.config import Thought5Config, config_summary
from fastwam_ood_eval.thought5.future_geometry_eval import (
    not_run_future_geometry_result,
)
from fastwam_ood_eval.thought5.future_runtime import _future_probe_projection
from fastwam_ood_eval.thought5.future_utility_eval import (
    not_run_future_utility_result,
)
from fastwam_ood_eval.thought5.geo_equiv_model import GeoEqAttachment
from fastwam_ood_eval.thought5.geo_targets import build_geometry_targets
from fastwam_ood_eval.thought5.paired_geometry_data import (
    assert_formal_exclusion,
    cohort_manifest,
)
from fastwam_ood_eval.thought5.representation_eval import (
    not_run_representation_result,
)
from fastwam_ood_eval.thought5.rollout_eval import not_run_rollout_result
from fastwam_ood_eval.thought5.schemas import (
    object_sha256,
    validate_full_object_seal,
    write_json_once,
    write_text_once,
)


def frozen_protocol_candidate(cfg: Thought5Config, cohort_sha256: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "thought5.phase5.frozen_protocol.v1",
        "status": (
            "frozen" if cfg.experiment.protocol_frozen else "candidate_not_frozen"
        ),
        "config_fingerprint": cfg.fingerprint,
        "cohort_sha256": cohort_sha256,
        "hypotheses": {
            "H1": "G3 reduces exact-state Camera geometry gap by >=25% vs B1 with both CIs below zero",
            "H2": "A1-G3 beats A0-G3 and AS-G3, with utility gain over B1",
            "H3": "G3 improves paired Camera success vs B1 while Clean is noninferior",
        },
        "controls": {
            "B0": "original checkpoint, no training",
            "B1": "matched LoRA/data/steps/seed; auxiliary lambdas zero",
            "G1": "Geo-REPA only",
            "G2": "pose/ray equivariance only",
            "G3": "Geo-REPA plus pose/ray equivariance",
            "G4": "shuffled geometry correspondence control, pilot/reduced budget",
        },
        "fixed_layer": "mot.video_kv_cache.15.v",
        "lora_targets": list(cfg.method.lora_targets),
        "statistics": {
            "episode_grouped_bootstrap_replicates": cfg.evaluation.bootstrap_replicates,
            "episode_grouped_bootstrap_seed": cfg.evaluation.bootstrap_seed,
            "task_cluster_bootstrap_seed": cfg.evaluation.task_bootstrap_seed,
            "confidence_level": 0.95,
            "h1_min_gap_reduction": cfg.evaluation.h1_min_gap_reduction,
            "clean_noninferiority_margin": cfg.evaluation.clean_noninferiority_margin,
            "g4_equivalence_fraction": cfg.evaluation.g4_equivalence_fraction,
        },
        "checkpoint_rule": cfg.training.checkpoint_rule,
        "thought3_future_adapter_recipe": {
            "steps": cfg.evaluation.thought3_adapter_steps,
            "learning_rate": cfg.evaluation.thought3_adapter_lr,
            "seed": cfg.evaluation.thought3_adapter_seed,
            "variants": ["A0", "A1", "AS"],
        },
        "future_geometry_probe": {
            "model": cfg.evaluation.future_probe_model,
            "source": "mot.video_kv_cache.15.v K=1 future tokens",
            "random_projection_dim": cfg.evaluation.future_probe_projection_dim,
            "random_projection_seed": cfg.evaluation.future_probe_projection_seed,
            "ridge_alphas": list(cfg.evaluation.future_probe_ridge_alphas),
            "fit_split": "train",
            "selection_split": "development",
            "formal_read_for_selection": False,
            "same_capacity_and_rule_all_backbones": True,
        },
        "rollout_protocol": {
            "max_steps": cfg.evaluation.rollout_max_steps,
            "wait_steps": cfg.evaluation.rollout_wait_steps,
            "control_horizon": cfg.evaluation.rollout_control_horizon,
            "image_size": list(cfg.evaluation.rollout_image_size),
            "save_failure_videos": cfg.evaluation.rollout_save_failure_videos,
            "checkpoint_selection_reads_rollout": False,
        },
        "stop_rules": {
            "hook_or_transform_debugs_remaining": 1,
            "lambda_adjustments_remaining": 1,
            "pilot_recipes_remaining": 1,
            "formal_recipe_mutation_allowed": False,
        },
        "formal_unlock": (
            "requires completed pilot direction signal and a separately sealed "
            "formal_protocol_frozen.json"
        ),
    }
    payload["protocol_sha256"] = object_sha256(payload)
    return payload


def _mock_model() -> Any:
    import torch

    class SelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = torch.nn.Linear(3072, 3072)
            self.v = torch.nn.Linear(3072, 3072)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SelfAttention()

    class EmptyBlock(torch.nn.Module):
        pass

    class Video(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([EmptyBlock() for _ in range(15)] + [Block()])

    class Mock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.video_expert = Video()

    torch.manual_seed(5507)
    return Mock()


def cpu_contract_dry_run(cfg: Thought5Config) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    manifest = cohort_manifest(cfg.cohort)
    assert_formal_exclusion(manifest)
    intrinsic = np.asarray(
        [[210.0, 0.0, 111.5], [0.0, 210.0, 111.5], [0.0, 0.0, 1.0]]
    )
    grid = two_camera_token_rays(intrinsic)
    assert_fastwam_token_layout(grid, 98)
    depth = np.full((224, 224), 1.25, dtype=np.float32)
    targets = build_geometry_targets(
        depth_map=depth,
        ray_grid=grid,
        camera_to_world=np.eye(4),
        eef_position_world=[0.0, 0.0, 0.5],
        object_position_world=[0.1, -0.2, 0.7],
    )
    if not targets.detached or targets.packed.shape != (98, 11):
        raise RuntimeError("CPU geometry target contract failed")
    probe_projection = _future_probe_projection(
        input_dim=cfg.backbone.hidden_dim,
        output_dim=cfg.evaluation.future_probe_projection_dim,
        seed=cfg.evaluation.future_probe_projection_seed,
    )
    repeated_projection = _future_probe_projection(
        input_dim=cfg.backbone.hidden_dim,
        output_dim=cfg.evaluation.future_probe_projection_dim,
        seed=cfg.evaluation.future_probe_projection_seed,
    )
    model = _mock_model()
    original_v = model.video_expert.blocks[15].self_attn.v
    hidden = torch.randn(1, 98, 3072)
    baseline = original_v(hidden).detach()
    attachment = GeoEqAttachment(
        model,
        lora_targets=cfg.method.lora_targets,
        lora_rank=cfg.method.lora_rank,
        lora_alpha=cfg.method.lora_alpha,
        lora_dropout=cfg.method.lora_dropout,
        projector_hidden_dim=cfg.method.geo_projector_hidden_dim,
        ray_pose_hidden_dim=cfg.method.ray_pose_hidden_dim,
    )
    wrapped = model.video_expert.blocks[15].self_attn.v(hidden).detach()
    if not torch.equal(baseline, wrapped):
        raise RuntimeError("zero-init LoRA changed baseline output")
    rays = torch.from_numpy(grid.rays_camera).unsqueeze(0)
    pose = torch.eye(4)[:3].reshape(1, 12)
    with attachment.conditioning(rays=rays, camera_pose_12=pose, enable_ray_pose=True):
        selected = model.video_expert.blocks[15].self_attn.v(hidden)
        if attachment.captured_value is not selected:
            raise RuntimeError("selected Video V capture identity mismatch")
        prediction = attachment.geometry_prediction()
    loss = prediction.square().mean() + selected.square().mean()
    loss.backward()
    parameter_manifest = attachment.parameter_manifest()
    if parameter_manifest["trainable_parameter_count"] <= 0:
        raise RuntimeError("mock trainable parameter count is empty")
    inference = attachment.inference_modules()
    if "geo_projector" in inference or set(inference) != {"backbone", "ray_pose_encoder"}:
        raise RuntimeError("training-only projector leaked into inference modules")
    latency_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            attachment.ray_pose_encoder(rays, pose)
    ray_pose_latency_ms = (time.perf_counter() - latency_started) * 1000 / 3
    attachment.close()
    result = {
        "schema_version": "thought5.phase5.cpu_dry_run.v1",
        "status": "complete",
        "scientific_result": False,
        "config_fingerprint": cfg.fingerprint,
        "cohort_manifest_sha256": manifest["manifest_sha256"],
        "checks": {
            "config_valid": True,
            "historical_exclusions_valid": True,
            "formal_multitask": len(cfg.cohort.formal_tasks) >= 2,
            "ray_layout_98x3": list(grid.rays_camera.shape) == [98, 3],
            "geometry_target_98x11": list(targets.packed.shape) == [98, 11],
            "future_probe_projection_3072x128": list(probe_projection.shape)
            == [3072, 128],
            "future_probe_projection_deterministic": bool(
                np.array_equal(probe_projection, repeated_projection)
            ),
            "missing_wrist_metadata_masked": int(grid.valid_mask.sum()) == 49,
            "zero_lora_baseline_bitwise": True,
            "selected_v_hook_fired": True,
            "backward_finite": bool(torch.isfinite(loss)),
            "trainable_whitelist_valid": True,
            "projector_removed_at_inference": True,
            "no_gt_depth_at_inference": True,
        },
        "mock_trainable_parameter_count": parameter_manifest[
            "trainable_parameter_count"
        ],
        "ray_pose_cpu_latency_ms": ray_pose_latency_ms,
        "elapsed_s": time.perf_counter() - started,
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"CPU dry-run checks failed: {result['checks']}")
    return result | {"_cohort_manifest": manifest, "_parameter_manifest": parameter_manifest}


def _placeholder_report() -> str:
    return """# Phase 5 — Camera-Equivariant Geometry Alignment

## Evidence chain

Thought1 found the Camera OOD failure; Thought2 found OOD future-consistency
degradation; Thought3 showed future-content sensitivity without held-out utility;
Thought4 localized an action-consumed Camera Equivariance Gap. Phase 5 now
intervenes with Geo-REPA plus relative pose/camera-ray conditioning.

## Current state

The audit and CPU contract dry-run are complete. Real GPU smoke, pilot,
representation evaluation, future geometry, future utility, and rollout are
**NOT RUN**. Therefore H1, H2, H3, and the final mechanism classification are
undetermined. No scientific claim is made from this scaffold.

## Permitted final interpretation

Only after the frozen B1/G1/G2/G3 controls, G4 shuffled control, expanded
held-out utility panel, and paired rollouts are complete may this report select
one preregistered mechanism classification. Even full support would establish
an important mechanism, not a unique or sufficient cause and not a claim about
all world-action models.
"""


def materialize_dry_run(cfg: Thought5Config) -> dict[str, Any]:
    root = cfg.experiment.output_dir
    dry_root = Path("outputs/thought5/phase5_cpu_dry_run_v2")
    dry_result_path = dry_root / "dry_run_result.json"
    if dry_result_path.is_file():
        prior = json.loads(dry_result_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            if prior.get("config_fingerprint") != cfg.fingerprint:
                raise RuntimeError(
                    "completed CPU dry-run belongs to a different config fingerprint"
                )
            return {
                "status": "complete",
                "scientific_result": False,
                "dry_run_result": str(dry_result_path),
                "output_root": str(root),
                "formal_status": "NOT RUN",
                "idempotent_reuse": True,
                "summary": config_summary(cfg),
            }
    result = cpu_contract_dry_run(cfg)
    cohort = result.pop("_cohort_manifest")
    parameter_manifest = result.pop("_parameter_manifest")
    dry_root.mkdir(parents=True, exist_ok=True)
    write_status_transition(dry_result_path, result)
    protocol = frozen_protocol_candidate(cfg, cohort["manifest_sha256"])
    write_status_transition(root / "frozen_protocol.json", protocol)
    write_status_transition(root / "cohort_manifest.json", cohort)
    candidate_manifest = dict(parameter_manifest)
    candidate_manifest["status"] = "mock_shape_verified_real_model_NOT_RUN"
    write_status_transition(root / "trainable_parameter_manifest.json", candidate_manifest)
    placeholders = {
        "training_results.json": {
            "schema_version": "thought5.phase5.training_results.v1",
            "status": "NOT RUN",
            "variants": list(cfg.training.variants),
        },
        "representation_results.json": not_run_representation_result(),
        "future_geometry_results.json": not_run_future_geometry_result(),
        "future_utility_results.json": not_run_future_utility_result(),
        "rollout_results.json": not_run_rollout_result(),
        "mechanism_evidence.json": {
            "schema_version": "thought5.phase5.mechanism_evidence.v1",
            "status": "NOT RUN",
            "H1": None,
            "H2": None,
            "H3": None,
        },
        "mechanism_classification.json": {
            "schema_version": "thought5.phase5.mechanism_classification.v1",
            "status": "NOT RUN",
            "classification": None,
        },
    }
    for name, value in placeholders.items():
        write_status_transition(root / name, value)
    write_status_transition(
        root / "run_status.json",
        {
            "schema_version": "thought5.phase5.run_status.v1",
            "status": "NOT RUN",
            "audit": "complete",
            "cpu_dry_run": "complete",
            "gpu_smoke": "NOT RUN",
            "pilot": "NOT RUN",
            "formal": "NOT RUN",
        },
    )
    write_text_once(root / "report.md", _placeholder_report(), allow_identical=True)
    integrity = execution_integrity(
        config_fingerprint=cfg.fingerprint,
        cohort_sha256=cohort["manifest_sha256"],
        stage_status={
            "audit": "complete",
            "cpu_dry_run": "complete",
            "gpu_smoke": "NOT RUN",
            "pilot": "NOT RUN",
            "formal": "NOT RUN",
        },
        checkpoints={},
        immutable_inputs={
            "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
            "dataset_stats_sha256": cfg.backbone.dataset_stats_sha256,
            "fastwam_commit": cfg.backbone.fastwam_commit,
        },
        status="NOT RUN",
    )
    if not validate_full_object_seal(integrity):
        raise RuntimeError("execution integrity seal regression")
    write_status_transition(root / "execution_integrity.json", integrity)
    names = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    artifact_manifest = build_artifact_manifest(root, names=names, status="NOT RUN")
    write_status_transition(root / "artifact_manifest.json", artifact_manifest)
    return {
        "status": "complete",
        "scientific_result": False,
        "dry_run_result": str(dry_root / "dry_run_result.json"),
        "output_root": str(root),
        "formal_status": "NOT RUN",
        "summary": config_summary(cfg),
    }
