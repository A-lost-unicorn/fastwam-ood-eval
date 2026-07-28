from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    PHASE_E2_LR_GRID,
    _matched_recipe_payload,
)
from fastwam_ood_eval.thought3.phase_e4_diversified_flow import (
    PHASE_E3_V2_FROZEN_ARTIFACTS,
    PHASE_E4_CONFIG_FINGERPRINT,
    PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS,
    PHASE_E4_SCHEMA,
    _assert_phase_e4_scope,
    _run_phase_e4,
    _track_checks,
    derive_e4_track_config,
    verify_frozen_phase_e3_v2,
)
from fastwam_ood_eval.thought3.real_training import (
    DIVERSIFIED_HELDOUT_FLOW_STEPS,
    DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET,
    _flow_objective_identity,
    aggregate_multiflow_probe_rows,
    diversified_flow_schedule_sha256,
    diversified_training_flow_slot,
    multiflow_subset_outcome,
    run_diversified_flow_training,
)


CONFIG = "configs/thought3/phase_e4_diversified_flow_diagnostic.yaml"


def test_phase_e4_scope_and_frozen_parent_evidence() -> None:
    cfg = load_thought3_config(CONFIG)
    _assert_phase_e4_scope(cfg)
    assert cfg.fingerprint == PHASE_E4_CONFIG_FINGERPRINT
    assert PHASE_E4_SCHEMA == "thought3.phase_e4.diversified_flow.v1"
    assert PHASE_E3_V2_FROZEN_ARTIFACTS == {
        "gate_e3_result.json": (
            "517c1e0cfc198f0bc44ab03d0d59349f20131d5c00efd958dd10f67aee1defe3"
        ),
        "run_status.json": (
            "f1bfa70b18df2a9494a88dea52501659cfd10f7f368bf4531d7da12582dc70c3"
        ),
        "pre_validation_result.json": (
            "68b7af97b5e17473ddb76472fe22c95abf5e1ec06e54ed7baeff324a2918ec14"
        ),
        "data_preparation.json": (
            "0b505d9764cbf97e45fdebb9d95c68cbb4e3cd88bed2e0d73cebe95b1ce14ae6"
        ),
        "logs/phase_e3.log": (
            "861c4bc58ac2bd3d3729d30e72aba3886908d996e01eb3e8f14858007191becc"
        ),
    }
    evidence = verify_frozen_phase_e3_v2()
    assert evidence["gate_e3_passed"] is False
    assert len(evidence["sample_ids"]) == 8


def test_phase_e4_tracks_change_only_variant_k_and_frozen_lr() -> None:
    cfg = load_thought3_config(CONFIG)
    assert PHASE_E2_LR_GRID == (
        ("lr_1e_04", 1e-4),
        ("lr_3e_04", 3e-4),
        ("lr_1e_03", 1e-3),
    )
    for lr_slug, learning_rate in PHASE_E2_LR_GRID:
        a0 = derive_e4_track_config(
            cfg,
            variant="A0",
            lr_slug=lr_slug,
            learning_rate=learning_rate,
        )
        a1 = derive_e4_track_config(
            cfg,
            variant="A1",
            lr_slug=lr_slug,
            learning_rate=learning_rate,
        )
        assert _matched_recipe_payload(a0) == _matched_recipe_payload(a1)
        assert (a0.variant, a0.sampler.active_k) == ("A0", 0)
        assert (a1.variant, a1.sampler.active_k) == ("A1", 1)


def test_diversified_flow_slots_are_unique_and_probe_disjoint() -> None:
    slots = [
        diversified_training_flow_slot(step)
        for step in range(1, 201)
    ]
    assert DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET == 10_000
    assert DIVERSIFIED_HELDOUT_FLOW_STEPS == (1, 2, 3, 4, 5)
    assert slots == list(range(10_001, 10_201))
    assert len(set(slots)) == 200
    assert not set(slots) & {0, *DIVERSIFIED_HELDOUT_FLOW_STEPS}
    with pytest.raises(RuntimeError, match="1..200"):
        diversified_training_flow_slot(0)
    with pytest.raises(RuntimeError, match="1..200"):
        diversified_training_flow_slot(201)


def test_diversified_flow_identity_and_schedule_hash_are_stable() -> None:
    identity = _flow_objective_identity(
        base_sample_id="sample-0",
        train_seed=3407,
        flow_step=10_001,
    )
    assert identity == {
        "action_noise_seed": 6150587608909682894,
        "action_timestep_seed": 3885227025470525887,
        "flow_objective_sha256": (
            "cc554860d4ea2ea665825850bd5de5be70b22866c1ee33c4ef35d47f3aa01432"
        ),
        "flow_step": 10_001,
    }
    rows = []
    for step in range(1, 201):
        flow_slot = diversified_training_flow_slot(step)
        objective = _flow_objective_identity(
            base_sample_id=f"sample-{(step - 1) % 8}",
            train_seed=3407,
            flow_step=flow_slot,
        )
        rows.append(
            {
                **objective,
                "action_weight": 1.0,
                "base_sample_id": f"sample-{(step - 1) % 8}",
                "global_step": step,
                "timestep": 500.0,
                "training_flow_slot": flow_slot,
            }
        )
    assert (
        diversified_flow_schedule_sha256(rows)
        == diversified_flow_schedule_sha256(rows)
    )
    corrupted = [dict(row) for row in rows]
    corrupted[1]["training_flow_slot"] += 1
    with pytest.raises(RuntimeError, match="not contiguous"):
        diversified_flow_schedule_sha256(corrupted)


def test_frozen_schedule_zero_weight_steps_are_known_before_run() -> None:
    result = json.loads(
        Path(
            "outputs/thought3/phase_e3_multiflow_v2/"
            "gate_e3_result.json"
        ).read_text()
    )
    sample_ids = result["initial_probes"]["A0"]["sample_ids"]
    zero_steps = []
    for global_step in range(1, 201):
        base_sample_id = sample_ids[(global_step - 1) % 8]
        identity = _flow_objective_identity(
            base_sample_id=base_sample_id,
            train_seed=3407,
            flow_step=diversified_training_flow_slot(global_step),
        )
        generator = torch.Generator(device="cpu").manual_seed(
            identity["action_timestep_seed"]
        )
        u = torch.rand(
            (1,),
            generator=generator,
            dtype=torch.float32,
        )
        sigma = 5.0 * u / (1.0 + 4.0 * u)
        timestep = (sigma * 1000.0).to(dtype=torch.bfloat16)
        if float(timestep.float()) == 1000.0:
            zero_steps.append(global_step)
    assert tuple(zero_steps) == PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS
    assert PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS == (49, 142)


def test_phase_e4_refuses_without_confirmation_before_model_load(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E4", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate E.4 loaded Fast-WAM before confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e4_diversified_flow."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E4"):
        _run_phase_e4(cfg, resume=False)


def test_phase_e4_track_checks_cover_schedule_probe_and_zero_weights(
    tmp_path,
) -> None:
    base_cfg = load_thought3_config(CONFIG)
    output = tmp_path / "thought3" / "track"
    cfg = replace(
        derive_e4_track_config(
            base_cfg,
            variant="A1",
            lr_slug="lr_1e_04",
            learning_rate=1e-4,
        ),
        experiment=replace(
            base_cfg.experiment,
            name="fake_phase_e4_a1",
            output_dir=output,
        ),
    )
    output.mkdir(parents=True)
    sample_ids = [f"sample-{index}" for index in range(8)]
    metrics = []
    for step in range(1, 201):
        base_sample_id = sample_ids[(step - 1) % 8]
        flow_slot = diversified_training_flow_slot(step)
        identity = _flow_objective_identity(
            base_sample_id=base_sample_id,
            train_seed=3407,
            flow_step=flow_slot,
        )
        zero_weight = step in PHASE_E4_EXPECTED_ZERO_WEIGHT_STEPS
        groups = {
            name: {
                "finite": True,
                "l2": (
                    1.0
                    if name == "gate"
                    or (
                        step >= 2
                        and name
                        in {
                            "attention",
                            "future_projector",
                            "non_gate",
                        }
                    )
                    else 0.0
                ),
                "nonzero_element_count": (
                    0
                    if step == 1 and name == "non_gate"
                    else 1
                ),
            }
            for name in (
                "all",
                "attention",
                "future_projector",
                "gate",
                "non_gate",
            )
        }
        metrics.append(
            {
                **identity,
                "action_weight": 0.0 if zero_weight else 1.0,
                "base_sample_id": base_sample_id,
                "global_step": step,
                "gradient_groups": groups,
                "loss": 0.0 if zero_weight else 1.0,
                "nan_or_inf": False,
                "timestep": 1000.0 if zero_weight else 500.0,
                "training_flow_slot": flow_slot,
                "zero_weight_objective": zero_weight,
            }
        )

    def probe(loss: float, ratio: float) -> dict:
        rows = []
        for sample_index, base_sample_id in enumerate(sample_ids):
            for flow_step in DIVERSIFIED_HELDOUT_FLOW_STEPS:
                endpoint = sample_index == 7 and flow_step == 5
                rows.append(
                    {
                        "action_hidden_norm": 10.0,
                        "action_loss": 0.0 if endpoint else loss,
                        "action_weight": 0.0 if endpoint else 1.0,
                        "attention_residual_norm": 2.0,
                        "base_sample_id": base_sample_id,
                        "flow_step": flow_step,
                        "gated_delta_nonzero_fraction": (
                            0.0 if ratio == 0 else 1.0
                        ),
                        "gated_delta_norm": ratio * 10.0,
                        "gated_delta_to_action_hidden_ratio": ratio,
                        "latency_ms": 5.0,
                        "peak_memory_mib": 100.0,
                        "timestep": (
                            1000.0 if endpoint else 100.0 + flow_step
                        ),
                    }
                )
        return aggregate_multiflow_probe_rows(
            rows,
            sample_ids=sample_ids,
            flow_steps=DIVERSIFIED_HELDOUT_FLOW_STEPS,
            variant="A1",
        )

    initial = probe(1.0, 0.0)
    final = probe(0.8, 0.1)
    initial_row = {**initial, "global_step": 0, "learning_rate": 1e-4}
    final_row = {**final, "global_step": 200, "learning_rate": 1e-4}
    metrics_path = output / "train_metrics.jsonl"
    probes_path = output / "heldout_multiflow_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics)
    )
    probes_path.write_text(
        json.dumps(initial_row, sort_keys=True)
        + "\n"
        + json.dumps(final_row, sort_keys=True)
        + "\n"
    )
    (output / "training_manifest.json").write_text("{}\n")
    (output / "training_state.json").write_text("{}\n")
    result = {
        "checkpoint_roundtrip": {
            "global_step": 200,
            "state_equal": True,
        },
        "completed_steps": 200,
        "final_probe": final_row,
        "first_attention_nonzero_gradient_step": 2,
        "first_non_gate_nonzero_gradient_step": 2,
        "first_projector_nonzero_gradient_step": 2,
        "initial_probe": initial_row,
        "max_peak_memory_mib": 100.0,
        "metrics": str(metrics_path),
        "optimizer_parameter_scope": "adapter_only",
        "outcome": multiflow_subset_outcome(initial, final),
        "probe_metrics": str(probes_path),
        "sample_count": 8,
        "sample_ids": sample_ids,
        "status": "complete",
        "train_flow_schedule_sha256": (
            diversified_flow_schedule_sha256(metrics)
        ),
        "uses_development_outcomes": False,
        "uses_ground_truth_future_input": False,
        "uses_ood_or_success_outcomes": False,
        "zero_weight_step_count": 2,
    }
    checks, artifacts = _track_checks(cfg, result)
    assert all(checks.values())
    assert len(artifacts["objective_schedule_signature"]) == 200


def test_diversified_training_api_has_no_outcome_input() -> None:
    parameters = set(
        inspect.signature(run_diversified_flow_training).parameters
    )
    assert parameters == {
        "cfg",
        "device",
        "frozen_parameter_sha256",
        "model",
        "prepared",
        "progress",
        "resume",
    }
    assert not parameters & {
        "development",
        "future_rgb",
        "ood",
        "success",
    }
