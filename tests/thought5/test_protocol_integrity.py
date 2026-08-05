from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from fastwam_ood_eval.thought5.artifacts import (
    build_artifact_manifest,
    execution_integrity,
    validate_artifact_manifest,
    write_report_transition,
    write_status_transition,
)
from fastwam_ood_eval.thought5.config import load_thought5_config
from fastwam_ood_eval.thought5.mechanism_decision import (
    MechanismEvidence,
    classify_mechanism,
)
from fastwam_ood_eval.thought5.paired_geometry_data import (
    assert_formal_exclusion,
    cohort_manifest,
)
from fastwam_ood_eval.thought5.panel_runtime import (
    Phase5PanelError,
    _validate_worker_import_preflight,
    _worker_environment,
    _parallel_waves,
    _pilot_direction_and_freeze,
    _validated_execution_schedule,
    parallel_schedule,
)
from fastwam_ood_eval.thought5.schemas import (
    Thought5ArtifactError,
    clean_project_commit,
    object_sha256,
    seal_full_object,
    validate_full_object_seal,
    write_json_once,
)


def formal_config():
    return load_thought5_config("configs/thought5/phase5_formal_v2.yaml")


def test_formal_cohort_excludes_thought3_and_thought4() -> None:
    manifest = cohort_manifest(formal_config().cohort)
    assert_formal_exclusion(manifest)
    formal = {(r["task_index"], r["episode_index"]) for r in manifest["rows"] if r["split"] == "formal"}
    excluded = {tuple(value) for value in manifest["historical_exclusions"]["all"]}
    assert not formal & excluded


def test_formal_is_multitask_and_split_by_task() -> None:
    cfg = formal_config()
    assert cfg.cohort.formal_tasks == (8, 9)
    assert not set(cfg.cohort.train_tasks) & set(cfg.cohort.formal_tasks)


def test_inference_configuration_forbids_gt_depth() -> None:
    cfg = formal_config()
    assert cfg.method.use_gt_depth_at_inference is False
    assert cfg.method.train_action_dit is False


def test_full_object_integrity_hash_covers_late_fields() -> None:
    sealed = seal_full_object({"a": 1, "late": {"field": 2}})
    assert validate_full_object_seal(sealed)
    sealed["late"]["field"] = 3
    assert not validate_full_object_seal(sealed)


def test_execution_integrity_uses_full_object_seal() -> None:
    value = execution_integrity(
        config_fingerprint="a",
        cohort_sha256="b",
        stage_status={"formal": "NOT RUN"},
        checkpoints={},
        immutable_inputs={"backbone": "c"},
    )
    assert validate_full_object_seal(value)
    assert value["all_fields_final_before_hash"] is True
    assert value["status"] == "NOT RUN"


def test_completed_output_cannot_be_overwritten(tmp_path) -> None:
    path = tmp_path / "status.json"
    write_status_transition(path, {"status": "complete", "value": 1})
    with pytest.raises(Thought5ArtifactError, match="finalized"):
        write_status_transition(path, {"status": "complete", "value": 2})


def test_completed_output_allows_only_identical_idempotent_replay(tmp_path) -> None:
    path = tmp_path / "status.json"
    value = {"status": "complete", "value": 1}
    write_status_transition(path, value)
    assert write_status_transition(path, value) == path


def test_not_run_output_can_transition_once(tmp_path) -> None:
    path = tmp_path / "status.json"
    write_status_transition(path, {"status": "NOT RUN"})
    write_status_transition(path, {"status": "complete"})
    assert json.loads(path.read_text())["status"] == "complete"


def test_mock_manifest_can_transition_to_real_once(tmp_path) -> None:
    path = tmp_path / "trainable.json"
    write_status_transition(
        path,
        {"status": "mock_shape_verified_real_model_NOT_RUN"},
    )
    write_status_transition(path, {"status": "complete"})
    with pytest.raises(Thought5ArtifactError, match="finalized"):
        write_status_transition(path, {"status": "complete", "changed": True})


def test_report_only_transitions_from_not_run_scaffold(tmp_path) -> None:
    path = tmp_path / "report.md"
    path.write_text("result is **NOT RUN**\n", encoding="utf-8")
    write_report_transition(path, "formal result\n")
    with pytest.raises(Thought5ArtifactError, match="finalized"):
        write_report_transition(path, "post-hoc rewrite\n")


def test_write_once_rejects_different_artifact(tmp_path) -> None:
    path = tmp_path / "one.json"
    write_json_once(path, {"a": 1})
    with pytest.raises(Thought5ArtifactError, match="overwrite"):
        write_json_once(path, {"a": 2})


def test_nan_fails_closed() -> None:
    with pytest.raises(Thought5ArtifactError, match="non-finite"):
        object_sha256({"bad": float("nan")})


def test_real_stage_project_commit_requires_clean_worktree(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Thought5 Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "frozen"],
        check=True,
    )
    assert len(clean_project_commit(tmp_path)) == 40
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(Thought5ArtifactError, match="clean committed"):
        clean_project_commit(tmp_path)


def test_parallel_gpu_schedule_is_preregistered() -> None:
    assert parallel_schedule("pilot", ("1", "2")) == (
        (("B1", "1"), ("G3", "2")),
        (("G4", "1"),),
    )
    assert parallel_schedule("pilot", ("0", "1", "2")) == (
        (("B1", "0"), ("G3", "1"), ("G4", "2")),
    )
    assert parallel_schedule("formal", ("0", "1", "2")) == (
        (("B1", "0"), ("G1", "1"), ("G2", "2")),
        (("G3", "0"), ("B0", "1")),
    )
    assert parallel_schedule("formal", ("0", "1", "2", "3"))[0] == (
        ("B1", "0"),
        ("G1", "1"),
        ("G2", "2"),
        ("G3", "3"),
    )
    assert _parallel_waves(("B0", "B1", "G1", "G2", "G3"), ("0", "1")) == (
        (("B0", "0"), ("B1", "1")),
        (("G1", "0"), ("G2", "1")),
        (("G3", "0"),),
    )
    with pytest.raises(Phase5PanelError, match="two or three"):
        parallel_schedule("pilot", ("0",))
    with pytest.raises(Phase5PanelError, match="three or four"):
        parallel_schedule("formal", ("0", "1"))


def test_three_gpu_namespaces_preserve_every_scientific_config_field() -> None:
    pairs = (
        (
            "configs/thought5/phase5_smoke_v3.yaml",
            "configs/thought5/phase5_smoke_v4.yaml",
        ),
        (
            "configs/thought5/phase5_pilot_v2.yaml",
            "configs/thought5/phase5_pilot_v3.yaml",
        ),
        (
            "configs/thought5/phase5_smoke_v4.yaml",
            "configs/thought5/phase5_smoke_v5.yaml",
        ),
        (
            "configs/thought5/phase5_pilot_v3.yaml",
            "configs/thought5/phase5_pilot_v4.yaml",
        ),
    )
    for old_path, new_path in pairs:
        old = deepcopy(dict(load_thought5_config(old_path).raw))
        new = deepcopy(dict(load_thought5_config(new_path).raw))
        for payload in (old, new):
            payload["experiment"] = dict(payload["experiment"])
            payload["experiment"].pop("name")
            payload["experiment"].pop("output_dir")
        assert old == new


def test_fresh_worker_imports_libero_without_parent_sys_path_side_effects(
    tmp_path, monkeypatch
) -> None:
    from fastwam_ood_eval.envs.libero_adapter import configure_libero_package

    config_path = str((tmp_path / "worker_libero").resolve())
    monkeypatch.setenv("LIBERO_CONFIG_PATH", config_path)
    configured = configure_libero_package(
        Path("third_party/LIBERO-plus"), tmp_path / "worker_libero"
    )
    config_path = str(configured["config_dir"])
    environment = _worker_environment(
        physical_gpu="0",
        project_commit="a" * 40,
        libero_config_path=config_path,
        base_environment={"PYTHONPATH": "/tmp/sentinel"},
    )
    entries = environment["PYTHONPATH"].split(os.pathsep)
    assert str((Path.cwd() / "third_party/LIBERO-plus").resolve()) in entries
    assert "/tmp/sentinel" in entries
    assert environment["LIBERO_CONFIG_PATH"] == config_path

    result = _validate_worker_import_preflight(
        physical_gpu="0",
        project_commit="a" * 40,
        libero_config_path=config_path,
    )
    assert result["returncode"] == 0


def test_execution_schedule_checksum_is_fail_closed(tmp_path) -> None:
    cfg = load_thought5_config("configs/thought5/phase5_pilot_v3.yaml")
    cfg = replace(cfg, experiment=replace(cfg.experiment, output_dir=tmp_path))
    schedule = {
        "schema_version": "thought5.phase5.execution_schedule.v1",
        "status": "frozen",
        "execution_only": True,
        "stage": "pilot",
        "config_fingerprint": cfg.fingerprint,
        "physical_gpu_ids": ["0", "1", "2"],
    }
    schedule["schedule_sha256"] = object_sha256(schedule)
    path = write_json_once(tmp_path / "execution_schedule.json", schedule)
    assert _validated_execution_schedule(cfg)["schedule_sha256"] == schedule[
        "schedule_sha256"
    ]

    schedule["physical_gpu_ids"] = ["2", "1", "0"]
    path.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(Phase5PanelError, match="schedule is invalid"):
        _validated_execution_schedule(cfg)


def test_rollout_semantics_are_frozen_in_config() -> None:
    evaluation = formal_config().evaluation
    assert (
        evaluation.rollout_max_steps,
        evaluation.rollout_wait_steps,
        evaluation.rollout_control_horizon,
        evaluation.rollout_image_size,
    ) == (400, 30, 10, (256, 256))


def test_training_only_pilot_cannot_unlock_formal(tmp_path) -> None:
    cfg = load_thought5_config("configs/thought5/phase5_pilot_v4.yaml")
    cfg = replace(
        cfg,
        experiment=replace(cfg.experiment, output_dir=tmp_path),
    )
    tracks = {
        name: {
            "selected_step": 100,
            "development_rows": [
                {"step": 100, "selection_objective": score}
            ],
        }
        for name, score in {"B1": 1.0, "G3": 0.5, "G4": 1.1}.items()
    }
    with pytest.raises(Phase5PanelError, match="collector has not committed"):
        _pilot_direction_and_freeze(cfg, tracks)
    assert not (tmp_path / "formal_protocol_frozen.json").exists()


def test_artifact_manifest_recomputes_every_file(tmp_path) -> None:
    write_json_once(tmp_path / "a.json", {"status": "NOT RUN"})
    manifest = build_artifact_manifest(tmp_path)
    assert manifest["status"] == "complete"
    validate_artifact_manifest(tmp_path, manifest)
    (tmp_path / "a.json").write_text("{}\n")
    with pytest.raises(Thought5ArtifactError, match="mismatch"):
        validate_artifact_manifest(tmp_path, manifest)


def evidence(**changes):
    values = dict(
        h1_camera_gap_reduction_fraction=0.3,
        h1_paired_ci_upper_below_zero=True,
        h1_task_ci_upper_below_zero=True,
        h1_clean_non_degraded=True,
        h1_lighting_specific=True,
        h2_a1_better_a0=True,
        h2_a1_better_shuffle=True,
        h2_utility_gain_grouped_ci_lower_above_zero=True,
        h2_utility_gain_task_ci_lower_above_zero=True,
        h2_a0_not_abnormally_worse=True,
        h3_camera_gain_grouped_ci_lower_above_zero=True,
        h3_camera_gain_task_ci_lower_above_zero=True,
        h3_clean_noninferior=True,
        h3_camera_specific=True,
        matched_control_explains_gain=False,
        shuffled_control_matches_gain=False,
    )
    values.update(changes)
    return MechanismEvidence(**values)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "full_mechanism_support"),
        ({"h2_a1_better_a0": False, "h3_camera_gain_task_ci_lower_above_zero": False}, "representation_only_support"),
        ({"h3_camera_gain_task_ci_lower_above_zero": False}, "utility_without_closed_loop_support"),
        ({"h2_a1_better_a0": False}, "closed_loop_without_future_mediation"),
        ({"h1_camera_gap_reduction_fraction": 0.1}, "mechanism_not_supported"),
        ({"h1_task_ci_upper_below_zero": False}, "mechanism_not_supported"),
        ({"h2_utility_gain_task_ci_lower_above_zero": False}, "closed_loop_without_future_mediation"),
        ({"h3_camera_gain_grouped_ci_lower_above_zero": False}, "utility_without_closed_loop_support"),
        ({"shuffled_control_matches_gain": True}, "mechanism_not_supported"),
    ],
)
def test_only_preregistered_mechanism_classifications(changes, expected) -> None:
    assert classify_mechanism(evidence(**changes))["classification"] == expected
