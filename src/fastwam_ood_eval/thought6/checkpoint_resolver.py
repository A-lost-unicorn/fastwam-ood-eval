"""Fail-closed resolution of the one authoritative Thought3 K=1 Adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought6.config import Thought6Config
from fastwam_ood_eval.thought6.schemas import Thought6Error, file_sha256, object_sha256


def resolve_adapter_checkpoint(cfg: Thought6Config) -> dict[str, Any]:
    result_path = cfg.authoritative_phase2_result
    if not result_path.is_file():
        raise Thought6Error(f"authoritative Thought3 Phase 2 result is absent: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != "thought3.phase2.full_28_4.result.v1"
        or result.get("status") != "complete"
        or result.get("classification") != "training_valid_dev_direction_not_observed"
        or result.get("phase3_unlocked") is not False
        or result.get("hard_checks", {}).get("fixed_step_200_primary") is not True
    ):
        raise Thought6Error("Thought3 authoritative Phase 2 result is not the frozen valid negative")
    tracks = result.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != {"A0", "A1"}:
        raise Thought6Error("Thought3 Phase 2 track set is ambiguous")
    candidate = Path(str(tracks["A1"].get("checkpoint", ""))).resolve()
    configured = cfg.adapter_checkpoint_path.resolve()
    if candidate != configured:
        raise Thought6Error(
            f"configured Adapter differs from the sole fixed A1 primary: {candidate}"
        )
    if candidate.name != "step_00000200" or not candidate.is_dir():
        raise Thought6Error("authoritative A1 checkpoint is not fixed step 200")
    manifest_path = cfg.adapter_manifest_path.resolve()
    if manifest_path != candidate / "manifest.json" or not manifest_path.is_file():
        raise Thought6Error("Adapter manifest path is absent or inconsistent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_path = candidate / "adapter.safetensors"
    adapter_sha = file_sha256(adapter_path)
    expected_checks = {
        "schema_version": "thought3.adapter_checkpoint.v1",
        "variant": "A1",
        "k": 1,
        "global_step": 200,
        "trainable_parameter_count": cfg.adapter_parameter_count,
        "backbone_checkpoint_sha256": cfg.backbone_checkpoint_sha256,
        "frozen_parameter_sha256": cfg.backbone_frozen_parameter_sha256,
        "fastwam_commit": cfg.fastwam_commit,
    }
    for key, expected in expected_checks.items():
        if manifest.get(key) != expected:
            raise Thought6Error(f"Adapter manifest {key} mismatch")
    extra = manifest.get("extra", {})
    if (
        extra.get("primary_checkpoint_rule") != "fixed_step_200_no_selection_no_fallback"
        or extra.get("checkpoint_kind") != "adapter_only"
        or extra.get("contains_backbone") is not False
        or extra.get("files_sha256", {}).get("adapter.safetensors") != adapter_sha
        or extra.get("adapter_state_sha256") != cfg.adapter_state_sha256
        or adapter_sha != cfg.adapter_file_sha256
        or tracks["A1"].get("adapter_state_sha256") != cfg.adapter_state_sha256
    ):
        raise Thought6Error("Adapter checkpoint file/state provenance mismatch")
    thought3_cfg = load_thought3_config(cfg.thought3_config_path)
    if thought3_cfg.adapter_structural_fingerprint != manifest.get("adapter_fingerprint"):
        raise Thought6Error("Thought3 Adapter structural config fingerprint mismatch")
    if thought3_cfg.sampler.active_k != 1 or thought3_cfg.sampler.shift != 5.0:
        raise Thought6Error("Thought3 future sampler contract mismatch")
    payload = {
        "schema_version": "thought6.adapter_checkpoint_manifest.v1",
        "status": "resolved_unique",
        "selection_rule": "authoritative_phase2_A1_fixed_step_200_no_selection_no_fallback",
        "authoritative_result_path": str(result_path),
        "authoritative_result_sha256": file_sha256(result_path),
        "checkpoint_path": str(candidate),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "adapter_path": str(adapter_path),
        "adapter_file_sha256": adapter_sha,
        "adapter_state_sha256": cfg.adapter_state_sha256,
        "adapter_fingerprint": manifest["adapter_fingerprint"],
        "adapter_config": thought3_cfg.adapter_structural_payload,
        "trainable_parameter_count": cfg.adapter_parameter_count,
        "backbone_checkpoint_sha256": cfg.backbone_checkpoint_sha256,
        "backbone_frozen_parameter_sha256": cfg.backbone_frozen_parameter_sha256,
        "future_sampler": {
            "k": 1,
            "shift": thought3_cfg.sampler.shift,
            "num_train_timesteps": thought3_cfg.sampler.num_train_timesteps,
            "rand_device": thought3_cfg.sampler.rand_device,
            "latent_shape": list(thought3_cfg.sampler.latent_shape),
            "cache_dtype": thought3_cfg.sampler.cache_dtype,
        },
        "ambiguity_count": 0,
        "loss_or_recency_used_for_selection": False,
    }
    payload["resolution_sha256"] = object_sha256(payload)
    return payload


def load_frozen_adapter(cfg: Thought6Config, *, device: str) -> tuple[Any, dict[str, Any]]:
    """Construct, load, and freeze the resolved Adapter without an optimizer."""

    from fastwam_ood_eval.thought3.checkpointing import load_adapter_checkpoint
    from fastwam_ood_eval.thought3.real_training import build_real_adapter

    resolved = resolve_adapter_checkpoint(cfg)
    thought3_cfg = load_thought3_config(cfg.thought3_config_path)
    adapter = build_real_adapter(thought3_cfg, device=device)
    manifest = load_adapter_checkpoint(
        resolved["checkpoint_path"],
        adapter=adapter,
        expected={
            "adapter_fingerprint": resolved["adapter_fingerprint"],
            "backbone_checkpoint_sha256": cfg.backbone_checkpoint_sha256,
            "config_fingerprint": json.loads(
                cfg.adapter_manifest_path.read_text(encoding="utf-8")
            )["config_fingerprint"],
            "dataset_stats_sha256": cfg.dataset_stats_sha256,
            "fastwam_commit": cfg.fastwam_commit,
            "frozen_parameter_sha256": cfg.backbone_frozen_parameter_sha256,
            "k": 1,
            "split_fingerprint": json.loads(
                cfg.adapter_manifest_path.read_text(encoding="utf-8")
            )["split_fingerprint"],
            "variant": "A1",
        },
    )
    adapter.requires_grad_(False)
    adapter.eval()
    if any(parameter.requires_grad or parameter.grad is not None for parameter in adapter.parameters()):
        raise Thought6Error("loaded Phase 6 Adapter is not fully frozen")
    return adapter, {**resolved, "loaded_global_step": manifest.global_step}
