from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from fastwam_ood_eval.thought6.checkpoint_resolver import resolve_adapter_checkpoint
from fastwam_ood_eval.thought6.config import load_thought6_config
from fastwam_ood_eval.thought6.rollout_policy import freeze_for_phase6
from fastwam_ood_eval.thought6.schemas import (
    Thought6ArtifactError,
    build_artifact_manifest,
    seal_full_object,
    validate_artifact_manifest,
    validate_full_object_seal,
    write_once_json,
)
from fastwam_ood_eval.thought6.task_selection import select_phase6_tasks


def _cfg():
    return load_thought6_config("configs/thought6/phase6_audit.yaml")


def test_33_authoritative_adapter_sha() -> None:
    row = resolve_adapter_checkpoint(_cfg())
    assert row["adapter_file_sha256"] == "0ebff4705039c4ca0a1e77330a9480f0ed4b6bc0b21235b447153417b64730b0"


def test_34_authoritative_adapter_parameter_count() -> None:
    assert resolve_adapter_checkpoint(_cfg())["trainable_parameter_count"] == 1_371_137


def test_35_freeze_disables_all_gradients() -> None:
    model, adapter = nn.Linear(2, 2), nn.Linear(2, 2)
    freeze_for_phase6(model, adapter)
    assert not any(p.requires_grad for p in list(model.parameters()) + list(adapter.parameters()))


def test_36_completed_artifact_is_immutable(tmp_path) -> None:
    path = tmp_path / "x.json"
    write_once_json(path, {"status": "complete", "x": 1})
    with pytest.raises(Thought6ArtifactError):
        write_once_json(path, {"status": "complete", "x": 2})


def test_37_execution_integrity_seal_recomputes() -> None:
    value = seal_full_object({"schema_version": "x", "status": "complete"})
    assert validate_full_object_seal(value)


def test_38_execution_integrity_detects_tamper() -> None:
    value = seal_full_object({"schema_version": "x", "status": "complete"})
    value["status"] = "tampered"
    assert not validate_full_object_seal(value)


def test_39_artifact_manifest_validates(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    value = build_artifact_manifest(tmp_path, names=["a.txt"], status="complete")
    validate_artifact_manifest(tmp_path, value)


def test_40_artifact_manifest_detects_tamper(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    value = build_artifact_manifest(tmp_path, names=["a.txt"], status="complete")
    (tmp_path / "a.txt").write_text("b", encoding="utf-8")
    with pytest.raises(Thought6ArtifactError):
        validate_artifact_manifest(tmp_path, value)


def test_41_task_selection_is_outcome_blind() -> None:
    value = select_phase6_tasks(_cfg())
    assert value["selection_is_outcome_blind"] and value["outcome_fields_read"] is False


def test_42_phase6_tasks_exclude_historical_goal_zero() -> None:
    value = select_phase6_tasks(_cfg())
    selected = {row["canonical_id"] for row in value["selected_tasks"]}
    assert "libero_goal/0" not in selected and len(selected) == 8


def test_43_goal_episode_provenance_is_available() -> None:
    value = select_phase6_tasks(_cfg())
    goal = [row for row in value["selected_tasks"] if row["suite"] == "libero_goal"]
    assert all(len(row["phase6b_episode_ids"]) == 4 for row in goal)


def test_44_missing_suites_fail_closed() -> None:
    value = select_phase6_tasks(_cfg())
    assert value["status"] == "blocked_missing_suite_datasets"


def test_45_no_optimizer_is_constructed_in_phase6_source() -> None:
    source = open("src/fastwam_ood_eval/thought6/rollout_policy.py", encoding="utf-8").read()
    assert "torch.optim" not in source and "AdamW(" not in source


def test_46_tensor_hash_supports_bfloat16() -> None:
    from fastwam_ood_eval.thought6.schemas import tensor_sha256

    assert len(tensor_sha256(torch.ones(2, dtype=torch.bfloat16))) == 64


def test_47_frozen_backbone_hash_is_unchanged_without_optimizer() -> None:
    from fastwam_ood_eval.thought5.checkpointing import frozen_parameter_sha256

    model, adapter = nn.Linear(2, 2), nn.Linear(2, 2)
    freeze_for_phase6(model, adapter)
    before = frozen_parameter_sha256(model.named_parameters())
    with torch.inference_mode():
        model(torch.ones(1, 2))
    after = frozen_parameter_sha256(model.named_parameters())
    assert before == after
