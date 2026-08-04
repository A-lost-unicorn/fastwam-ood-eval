from __future__ import annotations

import torch

from fastwam_ood_eval.thought5.checkpointing import (
    frozen_parameter_sha256,
    load_geoeq_checkpoint,
    save_geoeq_checkpoint,
)
from fastwam_ood_eval.thought5.geo_equiv_model import GeoEqAttachment
from fastwam_ood_eval.thought5.lora_targets import make_lora_linear
from fastwam_ood_eval.thought5.pipeline import _mock_model


def test_zero_initialized_lora_preserves_baseline_bitwise() -> None:
    torch.manual_seed(1)
    base = torch.nn.Linear(4, 5)
    wrapper = make_lora_linear(base, rank=2, alpha=2, dropout=0, module_path="x")
    value = torch.randn(3, 4)
    assert torch.equal(base(value), wrapper(value))


def test_only_preregistered_modules_are_trainable_and_budget_is_fixed() -> None:
    attachment = GeoEqAttachment(_mock_model())
    manifest_b1 = attachment.parameter_manifest()
    manifest_g3 = attachment.parameter_manifest()
    assert manifest_b1["trainable_parameter_count"] == 1_335_320
    assert manifest_b1["trainable_parameter_count"] == manifest_g3["trainable_parameter_count"]
    assert all(
        ".lora_" in name
        or name.startswith("geo_projector.")
        or name.startswith("ray_pose_encoder.")
        for name in manifest_b1["trainable_parameter_names"]
    )
    attachment.close()


def test_auxiliary_projector_is_removed_from_inference_graph() -> None:
    attachment = GeoEqAttachment(_mock_model())
    modules = attachment.inference_modules()
    assert set(modules) == {"backbone", "ray_pose_encoder"}
    assert modules["ray_pose_encoder"].requires_depth is False
    attachment.close()


def test_ray_pose_injection_repeats_in_frame_major_future_tokens() -> None:
    model = _mock_model()
    attachment = GeoEqAttachment(model)
    hidden = torch.randn(1, 294, 3072)
    rays = torch.randn(1, 98, 3)
    pose = torch.eye(4)[:3].reshape(1, 12)
    with attachment.conditioning(
        rays=rays,
        camera_pose_12=pose,
        enable_ray_pose=True,
    ):
        model.video_expert.blocks[15].self_attn.v(hidden)
        injected = attachment.injected_encoding
    assert tuple(injected.shape) == (1, 294, 3072)
    assert torch.equal(injected[:, :98], injected[:, 98:196])
    assert torch.equal(injected[:, :98], injected[:, 196:])
    attachment.close()


def test_adapter_only_checkpoint_roundtrip_and_frozen_sha(tmp_path) -> None:
    attachment = GeoEqAttachment(_mock_model())
    before = frozen_parameter_sha256(attachment.model.named_parameters())
    checkpoint = save_geoeq_checkpoint(
        tmp_path / "ckpt",
        attachment=attachment,
        variant="G3",
        global_step=2,
        config_fingerprint="config",
        cohort_fingerprint="cohort",
        backbone_checkpoint_sha256="backbone",
        frozen_before_sha256=before,
        frozen_after_sha256=before,
    )
    manifest = load_geoeq_checkpoint(
        checkpoint,
        attachment=attachment,
        expected={"variant": "G3", "config_fingerprint": "config"},
    )
    assert manifest["contains_backbone"] is False
    assert frozen_parameter_sha256(attachment.model.named_parameters()) == before
    attachment.close()
