from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fastwam_ood_eval.thought4.config import (
    Thought4ConfigError,
    load_thought4_config,
    validate_config,
)
from fastwam_ood_eval.thought4.io_utils import (
    Thought4ArtifactError,
    atomic_write_json,
    ensure_run_mutable,
)
from fastwam_ood_eval.thought4.phase4 import (
    Phase4ExecutionError,
    _require_confirmation,
    _verify_formal_smoke_gate,
    dry_run_payload,
)
from fastwam_ood_eval.thought4.schemas import SampleIdentity, sha256_canonical
from fastwam_ood_eval.thought4.video_feature_extractor import (
    ExtractedFeature,
    FeatureShardWriter,
    read_feature_shard,
)


def test_frozen_configs_validate_and_formal_cohort_is_64() -> None:
    smoke = load_thought4_config(
        "configs/thought4/phase4_geometry_action_smoke_v3.yaml"
    )
    formal = load_thought4_config(
        "configs/thought4/phase4_geometry_action_diagnosis_v1.yaml"
    )
    assert smoke.experiment.mode == "smoke"
    assert smoke.backbone.video_layers == (15,)
    assert smoke.backbone.action_hooks == (
        "action_expert.blocks.15.norm1",
    )
    assert smoke.cohort.conditions == ("clean", "camera", "lighting")
    assert formal.experiment.mode == "formal"
    assert formal.cohort.frames_per_episode == 2
    condition_ids = dict(formal.cohort.condition_task_ids)
    assert len(condition_ids["camera"]) == 5
    assert len(condition_ids["lighting"]) == 5
    assert len(condition_ids["robot_init"]) == 5
    assert (
        formal.cohort.train_base_states
        + formal.cohort.development_base_states
        + formal.cohort.test_base_states
        == 64
    )
    with pytest.raises(Thought4ConfigError, match="outputs/thought4"):
        validate_config(
            replace(
                smoke,
                experiment=replace(
                    smoke.experiment, output_dir=Path("outputs/thought3/bad")
                ),
            )
        )


def test_dry_run_is_read_only_and_does_not_import_torch(tmp_path: Path) -> None:
    cfg = load_thought4_config(
        "configs/thought4/phase4_geometry_action_smoke_v3.yaml"
    )
    before = set(Path("outputs/thought4").rglob("*")) if Path("outputs/thought4").exists() else set()
    had_torch = "torch" in sys.modules
    payload = dry_run_payload(cfg, stage="smoke")
    after = set(Path("outputs/thought4").rglob("*")) if Path("outputs/thought4").exists() else set()
    assert payload["would_load_torch"] is False
    assert payload["would_load_gpu_model"] is False
    assert payload["would_write"] is False
    assert before == after
    assert ("torch" in sys.modules) == had_torch


def test_completed_outputs_are_immutable_and_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = Path("outputs/thought4/test_run")
    run.mkdir(parents=True)
    atomic_write_json(run / "value.json", {"x": 1})
    with pytest.raises(Thought4ArtifactError, match="overwrite"):
        atomic_write_json(run / "value.json", {"x": 2})
    atomic_write_json(run / "run_status.json", {"status": "complete"})
    with pytest.raises(Thought4ArtifactError, match="immutable"):
        ensure_run_mutable(run)


def test_runner_scripts_require_explicit_confirmation() -> None:
    smoke = Path("scripts/run_thought4_phase4_smoke.sh").read_text()
    formal = Path("scripts/run_thought4_phase4_diagnosis.sh").read_text()
    assert "CONFIRM_THOUGHT4_PHASE4_SMOKE" in smoke
    assert "CONFIRM_THOUGHT4_PHASE4_FORMAL" in formal
    assert "THOUGHT4_GPU_ID" in smoke and "THOUGHT4_GPU_ID" in formal
    assert "CUDA_VISIBLE_DEVICES" in smoke and "CUDA_VISIBLE_DEVICES" in formal
    expected_egl = 'export MUJOCO_EGL_DEVICE_ID="${physical_gpu_id}"'
    assert expected_egl in smoke and expected_egl in formal
    assert "export MUJOCO_EGL_DEVICE_ID=0" not in smoke
    assert "export MUJOCO_EGL_DEVICE_ID=0" not in formal


def test_confirmation_requires_physical_egl_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIRM_THOUGHT4_PHASE4_SMOKE", "YES")
    monkeypatch.setenv("THOUGHT4_GPU_ID", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    with pytest.raises(Phase4ExecutionError, match="MUJOCO_EGL_DEVICE_ID"):
        _require_confirmation("smoke")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    _require_confirmation("smoke")


def test_feature_shard_resume_is_checksum_validated_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    run = Path("outputs/thought4/shard_resume")
    identity = SampleIdentity(
        task_id="0",
        episode_id="episode_000001",
        frame_index=3,
        split="train",
        timestamp=0.15,
        label_identity="demo:t3",
    )
    feature = ExtractedFeature(
        identity=identity,
        condition="clean",
        source="A",
        module_path="video_expert.blocks.15.norm1",
        layer_index=15,
        denoise_step_index=None,
        pooling="spatial_mean",
        tensor=torch.arange(8, dtype=torch.float32),
    )
    writer = FeatureShardWriter(run, source="A", shard_index=0)
    records = writer.write([feature])
    original_sha = records[0].shard_sha256

    resumed = FeatureShardWriter(
        run, source="A", shard_index=0, resume=True
    ).write([feature])
    loaded, loaded_records = read_feature_shard(
        writer.path, expected_source="A"
    )
    assert resumed[0].shard_sha256 == original_sha
    assert loaded_records[0].shard_sha256 == original_sha
    assert torch.equal(loaded[0].tensor, feature.tensor)

    writer.checksum_path.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(Thought4ArtifactError, match="checksum mismatch"):
        read_feature_shard(writer.path, expected_source="A")


def test_formal_requires_sha_valid_completed_real_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_path = Path(
        "configs/thought4/phase4_geometry_action_smoke_v3.yaml"
    ).resolve()
    formal_path = Path(
        "configs/thought4/phase4_geometry_action_diagnosis_v1.yaml"
    ).resolve()
    smoke = load_thought4_config(smoke_path)
    formal = load_thought4_config(formal_path)
    monkeypatch.chdir(tmp_path)
    root = smoke.experiment.output_dir
    root.mkdir(parents=True)
    atomic_write_json(
        root / "run_status.json",
        {
            "status": "complete",
            "stage": "smoke",
            "config_fingerprint": smoke.fingerprint,
        },
    )
    result = {
        "status": "passed",
        "scientific_result": False,
        "formal_unlocked": True,
        "config_fingerprint": smoke.fingerprint,
        "backbone_parameter_sha256_before": (
            formal.backbone.frozen_parameter_sha256
        ),
        "backbone_parameter_sha256_after": (
            formal.backbone.frozen_parameter_sha256
        ),
        "identity_replacement": {
            "passed": True,
            "module_path": "mot.video_kv_cache.15.v",
            "hook_location": "forward_action_with_video_cache argument",
        },
        "future_rgb_read": False,
        "success_outcome_read": False,
    }
    result["result_sha256"] = sha256_canonical(result)
    atomic_write_json(root / "smoke_result.json", result)
    gate = _verify_formal_smoke_gate(
        formal, smoke_config_path=smoke_path
    )
    assert gate["passed"] is True

    result["result_sha256"] = "0" * 64
    atomic_write_json(
        root / "smoke_result.json", result, overwrite=True
    )
    with pytest.raises(Phase4ExecutionError, match="hard checks failed"):
        _verify_formal_smoke_gate(formal, smoke_config_path=smoke_path)
