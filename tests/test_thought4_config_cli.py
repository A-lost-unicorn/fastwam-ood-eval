from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastwam_ood_eval.thought4.config import (
    EXACT_STATE_TRAJECTORY_PAIRING,
    FP32_SUBSPACE_ARITHMETIC,
    ROBOT_INIT_TRAJECTORY_PAIRING,
    SIMULATOR_TRAJECTORY_LABEL_SOURCE,
    Thought4ConfigError,
    config_to_dict,
    load_thought4_config,
    validate_config,
)
from fastwam_ood_eval.thought4.cohort import PlannedBaseState
from fastwam_ood_eval.thought4.io_utils import (
    Thought4ArtifactError,
    atomic_write_json,
    ensure_run_mutable,
    write_or_verify_json,
)
from fastwam_ood_eval.thought4.phase4 import (
    Phase4ExecutionError,
    _alignment_audit_payload,
    _require_confirmation,
    _verify_formal_smoke_gate,
    _write_probe_stage_artifacts,
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
        "configs/thought4/phase4_geometry_action_smoke_v8.yaml"
    )
    formal = load_thought4_config(
        "configs/thought4/phase4_geometry_action_diagnosis_v6.yaml"
    )
    assert smoke.experiment.mode == "smoke"
    assert smoke.backbone.video_layers == (15,)
    assert smoke.backbone.action_hooks == (
        "action_expert.blocks.15.norm1",
    )
    assert smoke.cohort.conditions == (
        "clean",
        "camera",
        "lighting",
        "robot_init",
    )
    previous_smoke = load_thought4_config(
        "configs/thought4/phase4_geometry_action_smoke_v7.yaml"
    )
    assert replace(
        smoke,
        experiment=previous_smoke.experiment,
        probe=replace(
            smoke.probe,
            trajectory_label_source=(
                previous_smoke.probe.trajectory_label_source
            ),
        ),
        intervention=replace(
            smoke.intervention,
            subspace_arithmetic=(
                previous_smoke.intervention.subspace_arithmetic
            ),
        ),
    ) == previous_smoke
    historical_smoke = load_thought4_config(
        "configs/thought4/phase4_geometry_action_smoke_v3.yaml"
    )
    assert historical_smoke.cohort.conditions == (
        "clean",
        "camera",
        "lighting",
    )
    assert replace(
        smoke,
        experiment=historical_smoke.experiment,
        cohort=replace(
            smoke.cohort,
            conditions=historical_smoke.cohort.conditions,
        ),
        probe=replace(
            smoke.probe,
            trajectory_label_source=(
                historical_smoke.probe.trajectory_label_source
            ),
        ),
        intervention=replace(
            smoke.intervention,
            subspace_arithmetic=(
                historical_smoke.intervention.subspace_arithmetic
            ),
        ),
    ) == historical_smoke
    previous_formal = load_thought4_config(
        "configs/thought4/phase4_geometry_action_diagnosis_v5.yaml"
    )
    assert replace(
        formal,
        experiment=previous_formal.experiment,
        probe=replace(
            formal.probe,
            trajectory_label_source=(
                previous_formal.probe.trajectory_label_source
            ),
        ),
        intervention=replace(
            formal.intervention,
            subspace_arithmetic=(
                previous_formal.intervention.subspace_arithmetic
            ),
        ),
    ) == previous_formal
    historical_formal = load_thought4_config(
        "configs/thought4/phase4_geometry_action_diagnosis_v1.yaml"
    )
    assert replace(
        formal,
        experiment=historical_formal.experiment,
        probe=replace(
            formal.probe,
            trajectory_label_source=(
                historical_formal.probe.trajectory_label_source
            ),
        ),
        intervention=replace(
            formal.intervention,
            subspace_arithmetic=(
                historical_formal.intervention.subspace_arithmetic
            ),
        ),
    ) == historical_formal
    assert (
        smoke.probe.trajectory_label_source
        == SIMULATOR_TRAJECTORY_LABEL_SOURCE
    )
    assert (
        formal.probe.trajectory_label_source
        == SIMULATOR_TRAJECTORY_LABEL_SOURCE
    )
    assert smoke.intervention.subspace_arithmetic == FP32_SUBSPACE_ARITHMETIC
    assert formal.intervention.subspace_arithmetic == FP32_SUBSPACE_ARITHMETIC
    assert smoke.fingerprint == (
        "81d3885ccb5b58806c1a729e509c039f6e1cb33a34ff242f8fa16785796149d7"
    )
    assert formal.fingerprint == (
        "3b14a7d7fd09deda9253bb1cd9950d9c4b5bd0cdf9f124a4dfede22add5c24f6"
    )
    assert dry_run_payload(formal, stage="formal")[
        "planned_cohort_sha256"
    ] == (
        "9af7cf7c1933fb1e5574099361f6d7dcc7500727480ecb4bbf010089f28d8f04"
    )
    assert previous_smoke.fingerprint == (
        "72eeaed07fb5f2b8457106dd3d5cd89333a47dca19bd4533bb9c6a90b13ebb90"
    )
    assert previous_formal.fingerprint == (
        "7b2a8e7ba6a51fe5246599324b09983d8df19824bfc906d7e8fd3932276fbb3a"
    )
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
        "configs/thought4/phase4_geometry_action_smoke_v8.yaml"
    )
    before = set(Path("outputs/thought4").rglob("*")) if Path("outputs/thought4").exists() else set()
    had_torch = "torch" in sys.modules
    payload = dry_run_payload(cfg, stage="smoke")
    after = set(Path("outputs/thought4").rglob("*")) if Path("outputs/thought4").exists() else set()
    assert payload["would_load_torch"] is False
    assert payload["would_load_gpu_model"] is False
    assert payload["would_write"] is False
    assert payload["trajectory_label_source"] == SIMULATOR_TRAJECTORY_LABEL_SOURCE
    assert payload["demonstration_alignment_policy"] == (
        "disclosure_only_3cm_15deg"
    )
    assert payload["subspace_arithmetic"] == FP32_SUBSPACE_ARITHMETIC
    assert payload["planned_cohort_sha256"] == (
        "a67ff85321dc684a80b853b58ab133905232a275e6d71255fd1c966b9a3d6c12"
    )
    assert before == after
    assert ("torch" in sys.modules) == had_torch


def test_config_artifact_is_json_roundtrip_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_thought4_config(
        "configs/thought4/phase4_geometry_action_smoke_v8.yaml"
    )
    payload = config_to_dict(cfg)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["cohort"]["conditions"] == [
        "clean",
        "camera",
        "lighting",
        "robot_init",
    ]
    assert payload["backbone"]["video_layers"] == [15]
    assert payload["probe"]["trajectory_label_source"] == (
        SIMULATOR_TRAJECTORY_LABEL_SOURCE
    )
    assert payload["intervention"]["subspace_arithmetic"] == (
        FP32_SUBSPACE_ARITHMETIC
    )
    monkeypatch.chdir(tmp_path)
    path = Path("outputs/thought4/config_roundtrip/config.json")
    atomic_write_json(path, payload)
    assert write_or_verify_json(path, payload) == path


def test_alignment_audit_discloses_failures_without_selecting_rows() -> None:
    cfg = load_thought4_config(
        "configs/thought4/phase4_geometry_action_smoke_v8.yaml"
    )
    plans = tuple(
        PlannedBaseState(
            task_id="0",
            task_index=0,
            episode_id=f"episode_{index:06d}",
            episode_index=index,
            task_local_episode_index=index,
            frame_index=10 + index,
            split="train" if index == 0 else "development",
            timestamp=(10 + index) / 20,
            replay_action_count=10 + index,
        )
        for index in range(2)
    )
    samples = []
    for index, plan in enumerate(plans):
        samples.append(
            SimpleNamespace(
                plan=plan,
                condition="clean",
                trajectory_label_source=SIMULATOR_TRAJECTORY_LABEL_SOURCE,
                trajectory_label_pairing=EXACT_STATE_TRAJECTORY_PAIRING,
                demonstration_state_alignment={
                    "applicable": True,
                    "translation_error_m": 0.01 if index == 0 else 0.04,
                    "rotation_geodesic_error_degrees": 2.0,
                    "translation_limit_m": 0.03,
                    "rotation_limit_degrees": 15.0,
                    "enforcement": "disclosure_only_3cm_15deg",
                    "passed": index == 0,
                },
            )
        )
    result = _alignment_audit_payload(cfg, plans, samples)
    assert result["base_state_count"] == 2
    assert result["pass_count"] == 1
    assert result["failure_count"] == 1
    assert result["split_counts"]["development"]["failed"] == 1
    assert result["selection_effect"] == (
        "none_all_planned_base_states_retained"
    )
    supplied = result["audit_sha256"]
    unhashed = dict(result)
    unhashed.pop("audit_sha256")
    assert supplied == sha256_canonical(unhashed)


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
    assert "phase4_geometry_action_smoke_v8.yaml" in smoke
    assert "phase4_geometry_action_diagnosis_v6.yaml" in formal


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


def test_probe_stage_artifacts_commit_before_intervention_and_resume_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_thought4_config(
        Path(__file__).parents[1]
        / "configs/thought4/phase4_geometry_action_diagnosis_v6.yaml"
    )
    output = Path("outputs/thought4/probe_stage")
    video = {"schema_version": "video", "rows": []}
    action = {"schema_version": "action", "rows": []}
    selection = {
        "module_path": "mot.video_kv_cache.15.v",
        "feature_key": "mot.video_kv_cache.15.v|spatial_mean",
        "target": "eef_object_translation_camera",
    }
    layer, stage, paths = _write_probe_stage_artifacts(
        output,
        cfg,
        video_result=video,
        action_result=action,
        selection=selection,
        resume=False,
    )
    assert len(paths) == 4 and all(path.is_file() for path in paths)
    assert stage["status"] == "complete_before_intervention"
    assert stage["frozen_intervention_selection"] == selection
    assert stage["test_used_for_selection"] is False
    assert layer["rows"] == []
    supplied = stage["result_sha256"]
    assert supplied == sha256_canonical(
        {key: value for key, value in stage.items() if key != "result_sha256"}
    )
    _write_probe_stage_artifacts(
        output,
        cfg,
        video_result=video,
        action_result=action,
        selection=selection,
        resume=True,
    )
    with pytest.raises(Thought4ArtifactError, match="differs during resume"):
        _write_probe_stage_artifacts(
            output,
            cfg,
            video_result=video,
            action_result={"schema_version": "changed", "rows": []},
            selection=selection,
            resume=True,
        )


def test_formal_requires_sha_valid_completed_real_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_path = Path(
        "configs/thought4/phase4_geometry_action_smoke_v8.yaml"
    ).resolve()
    formal_path = Path(
        "configs/thought4/phase4_geometry_action_diagnosis_v6.yaml"
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
    alignment_audit = {
        "schema_version": "thought4.phase4.alignment_audit.v1",
        "config_fingerprint": smoke.fingerprint,
        "trajectory_label_source": SIMULATOR_TRAJECTORY_LABEL_SOURCE,
        "alignment_policy": "disclosure_only_3cm_15deg",
        "selection_effect": "none_all_planned_base_states_retained",
        "exact_state_trajectory_pairing": EXACT_STATE_TRAJECTORY_PAIRING,
        "robot_init_trajectory_pairing": ROBOT_INIT_TRAJECTORY_PAIRING,
        "future_rgb_read": False,
        "success_outcome_read": False,
        "base_state_count": 2,
        "pass_count": 2,
        "failure_count": 0,
        "rows": [
            {
                "sample_id": "sample-a",
                "enforcement": "disclosure_only_3cm_15deg",
                "translation_limit_m": 0.03,
                "rotation_limit_degrees": 15.0,
                "passed_3cm_15deg": True,
            },
            {
                "sample_id": "sample-b",
                "enforcement": "disclosure_only_3cm_15deg",
                "translation_limit_m": 0.03,
                "rotation_limit_degrees": 15.0,
                "passed_3cm_15deg": True,
            },
        ],
    }
    alignment_audit["audit_sha256"] = sha256_canonical(alignment_audit)
    atomic_write_json(root / "alignment_audit.json", alignment_audit)
    reconstruction = {
        "schema_version": "thought4.phase4.bitwise_correct_reconstruction.v1",
        "subspace_arithmetic": FP32_SUBSPACE_ARITHMETIC,
        "input_shape": [1, 98, 3072],
        "input_dtype": "torch.bfloat16",
        "compute_dtype": "torch.float32",
        "output_dtype": "torch.bfloat16",
        "single_output_cast": True,
        "residual_reconstruction_max_abs": 0.0,
        "input_sha256": "c" * 64,
        "output_sha256": "c" * 64,
        "bitwise_equal_after_output_cast": True,
        "passed": True,
        "technical_weight_sha256": "d" * 64,
        "basis_sha256": "e" * 64,
        "basis_dtype": "torch.float32",
        "subspace_rank": 3,
        "action_replacement": {"action_l2": 0.0},
        "allowed_action_l2": 1e-6,
        "action_replacement_passed": True,
    }
    reconstruction["contract_sha256"] = sha256_canonical(reconstruction)
    result = {
        "status": "passed",
        "scientific_result": False,
        "formal_unlocked": True,
        "config_fingerprint": smoke.fingerprint,
        "base_state_count": 2,
        "condition_count": 8,
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
            "bf16_fp32_subspace_reconstruction": reconstruction,
        },
        "robot_init_input_state_check": {
            "passed": True,
            "sample_count": 2,
            "reset_matches_clean_count": 2,
            "reset_differs_clean_count": 0,
            "input_matches_clean_count": 0,
            "input_differs_clean_count": 2,
            "simulator_state_differs_clean_count": 2,
            "same_object_layout_count": 2,
            "validation_time": "model_input_t_after_demonstration_prefix",
            "reset_state_is_disclosure_only": True,
        },
        "future_rgb_read": False,
        "success_outcome_read": False,
        "trajectory_label_source": SIMULATOR_TRAJECTORY_LABEL_SOURCE,
        "alignment_audit_sha256": alignment_audit["audit_sha256"],
        "alignment_audit_pass_count": 2,
        "alignment_audit_failure_count": 0,
        "alignment_selection_effect": (
            "none_all_planned_base_states_retained"
        ),
        "exact_state_trajectory_pairing": EXACT_STATE_TRAJECTORY_PAIRING,
        "robot_init_trajectory_pairing": ROBOT_INIT_TRAJECTORY_PAIRING,
        "subspace_arithmetic": FP32_SUBSPACE_ARITHMETIC,
    }
    result["result_sha256"] = sha256_canonical(result)
    atomic_write_json(root / "smoke_result.json", result)
    gate = _verify_formal_smoke_gate(
        formal, smoke_config_path=smoke_path
    )
    assert gate["passed"] is True

    non_bitwise = json.loads(json.dumps(result))
    non_bitwise_contract = non_bitwise["identity_replacement"][
        "bf16_fp32_subspace_reconstruction"
    ]
    non_bitwise_contract["bitwise_equal_after_output_cast"] = False
    non_bitwise_contract["passed"] = False
    non_bitwise_contract["contract_sha256"] = sha256_canonical(
        {
            key: value
            for key, value in non_bitwise_contract.items()
            if key != "contract_sha256"
        }
    )
    non_bitwise["result_sha256"] = sha256_canonical(
        {
            key: value
            for key, value in non_bitwise.items()
            if key != "result_sha256"
        }
    )
    atomic_write_json(
        root / "smoke_result.json", non_bitwise, overwrite=True
    )
    with pytest.raises(
        Phase4ExecutionError, match="bf16_fp32_reconstruction_valid"
    ):
        _verify_formal_smoke_gate(formal, smoke_config_path=smoke_path)
    atomic_write_json(root / "smoke_result.json", result, overwrite=True)

    without_robot_check = dict(result)
    without_robot_check.pop("robot_init_input_state_check")
    without_robot_check["result_sha256"] = sha256_canonical(
        {
            key: value
            for key, value in without_robot_check.items()
            if key != "result_sha256"
        }
    )
    atomic_write_json(
        root / "smoke_result.json",
        without_robot_check,
        overwrite=True,
    )
    with pytest.raises(
        Phase4ExecutionError, match="robot_init_input_state_valid"
    ):
        _verify_formal_smoke_gate(formal, smoke_config_path=smoke_path)

    result["result_sha256"] = "0" * 64
    atomic_write_json(
        root / "smoke_result.json", result, overwrite=True
    )
    with pytest.raises(Phase4ExecutionError, match="hard checks failed"):
        _verify_formal_smoke_gate(formal, smoke_config_path=smoke_path)

    result["result_sha256"] = sha256_canonical(
        {
            key: value
            for key, value in result.items()
            if key != "result_sha256"
        }
    )
    atomic_write_json(root / "smoke_result.json", result, overwrite=True)
    corrupted_alignment = dict(alignment_audit)
    corrupted_alignment["failure_count"] = 1
    atomic_write_json(
        root / "alignment_audit.json",
        corrupted_alignment,
        overwrite=True,
    )
    with pytest.raises(
        Phase4ExecutionError, match="alignment_audit_valid"
    ):
        _verify_formal_smoke_gate(formal, smoke_config_path=smoke_path)
