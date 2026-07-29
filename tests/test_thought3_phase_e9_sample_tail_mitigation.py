from __future__ import annotations

import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.objective_aggregation_training import (
    OBJECTIVE_AGGREGATION_UPDATES,
    OBJECTIVES_PER_UPDATE,
    _backward_mean_objective,
    objective_aggregation_flow_slot,
    sample_loss_weights_sha256,
)
from fastwam_ood_eval.thought3.phase_e9_sample_tail_mitigation import (
    PHASE_E9_BOOTSTRAP_REPLICATES,
    PHASE_E9_CALIBRATION_PAYLOAD_SHA256,
    PHASE_E9_CONFIG_FINGERPRINT,
    PHASE_E9_EXPECTED_ZERO_WEIGHT_SLOTS,
    PHASE_E9_FAMILYWISE_COMPARISONS,
    PHASE_E9_HELDOUT_FLOW_STEPS,
    PHASE_E9_HELDOUT_IDENTITY_SHA256,
    PHASE_E9_HELDOUT_ZERO_WEIGHT_POSITIONS,
    PHASE_E9_IDENTITY_SCHEDULE_SHA256,
    PHASE_E9_INITIAL_SAMPLE_LOSSES,
    PHASE_E9_NORMALIZED_PROTOCOL,
    PHASE_E9_RAW_PROTOCOL,
    PHASE_E9_RESERVED_COHORT_SHA256,
    PHASE_E9_ROOT,
    PHASE_E9_SAMPLE_LOSS_WEIGHTS,
    PHASE_E9_SAMPLE_LOSS_WEIGHTS_SHA256,
    PHASE_E9_SCHEMA,
    PHASE_E9B_IDENTITY_SCHEDULE_SHA256,
    PHASE_E9B_ZERO_WEIGHT_POSITIONS,
    _assert_frozen_design,
    _assert_phase_e9_scope,
    _run_phase_e9,
    classify_sample_tail_mitigation,
    derive_e9_track_config,
    paired_tail_bootstrap,
    probe_identity_schedule_sha256,
    run_phase_e9_sample_tail_mitigation,
    verify_frozen_phase_e8,
    verify_reserved_replication_cohort,
)
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    _matched_recipe_payload,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
)


CONFIG = "configs/thought3/phase_e9_sample_tail_mitigation.yaml"


@pytest.fixture(scope="module")
def frozen_parent() -> dict:
    return verify_frozen_phase_e8()


def _zero_positions(
    sample_ids: list[str],
    flow_steps: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    observed = []
    for sample_index, base_sample_id in enumerate(sample_ids):
        for flow_step in flow_steps:
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=3407,
                flow_step=flow_step,
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
                observed.append((sample_index, flow_step))
    return tuple(observed)


def test_phase_e9_scope_and_result_conditioned_parent_are_frozen(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    _assert_phase_e9_scope(cfg)
    assert cfg.fingerprint == PHASE_E9_CONFIG_FINGERPRINT
    assert cfg.experiment.output_dir == PHASE_E9_ROOT
    assert PHASE_E9_SCHEMA == "thought3.phase_e9.sample_tail_mitigation.v1"
    assert frozen_parent["classification"] == "mixed_or_inconclusive"
    assert frozen_parent["known_before_e9"] == {
        "all_e8_results_read": True,
        "mitigation_selected_after_e8": True,
        "not_independent_confirmatory": True,
    }


def test_fixed_inverse_initial_loss_weights_are_exact(
    frozen_parent,
) -> None:
    sample_ids = frozen_parent["sample_ids"]
    inverse = [
        1.0 / PHASE_E9_INITIAL_SAMPLE_LOSSES[sample_id]
        for sample_id in sample_ids
    ]
    inverse_mean = sum(inverse) / len(inverse)
    for sample_id, value in zip(sample_ids, inverse):
        assert math.isclose(
            PHASE_E9_SAMPLE_LOSS_WEIGHTS[sample_id],
            value / inverse_mean,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    assert sum(PHASE_E9_SAMPLE_LOSS_WEIGHTS.values()) == 8.0
    assert (
        sample_loss_weights_sha256(
            sample_ids,
            PHASE_E9_SAMPLE_LOSS_WEIGHTS,
        )
        == PHASE_E9_SAMPLE_LOSS_WEIGHTS_SHA256
    )
    assert len(PHASE_E9_CALIBRATION_PAYLOAD_SHA256) == 64


def test_weighted_backward_is_the_only_optimizer_objective_change() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    weights = list(PHASE_E9_SAMPLE_LOSS_WEIGHTS.values())
    for coefficient, weight in enumerate(weights, start=1):
        _backward_mean_objective(
            parameter * coefficient,
            accumulation_factor=8,
            sample_weight=weight,
        )
    expected = sum(
        coefficient * weight
        for coefficient, weight in enumerate(weights, start=1)
    ) / 8
    assert parameter.grad is not None
    assert math.isclose(float(parameter.grad), expected, rel_tol=1e-6)
    assert PHASE_E9_RAW_PROTOCOL.gradient_reduction == "arithmetic_mean"
    assert PHASE_E9_NORMALIZED_PROTOCOL.gradient_reduction == (
        "fixed_sample_normalized_mean"
    )


def test_four_tracks_are_matched_except_variant_k_and_external_weights() -> None:
    cfg = load_thought3_config(CONFIG)
    tracks = {
        f"{recipe}/{variant}": derive_e9_track_config(
            cfg,
            recipe=recipe,
            variant=variant,
        )
        for recipe in ("raw", "normalized")
        for variant in ("A0", "A1")
    }
    for recipe in ("raw", "normalized"):
        assert (
            _matched_recipe_payload(tracks[f"{recipe}/A0"])
            == _matched_recipe_payload(tracks[f"{recipe}/A1"])
        )
    for variant in ("A0", "A1"):
        assert (
            _matched_recipe_payload(tracks[f"raw/{variant}"])
            == _matched_recipe_payload(tracks[f"normalized/{variant}"])
        )
    assert tracks["raw/A0"].sampler.active_k == 0
    assert tracks["normalized/A1"].sampler.active_k == 1


def test_new_train_and_heldout_flow_namespaces_are_frozen(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    design = _assert_frozen_design(
        frozen_parent["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    assert design["train_identity_schedule_sha256"] == (
        PHASE_E9_IDENTITY_SCHEDULE_SHA256
    )
    assert design["heldout_identity_schedule_sha256"] == (
        PHASE_E9_HELDOUT_IDENTITY_SHA256
    )
    slots = [
        objective_aggregation_flow_slot(
            update,
            micro,
            flow_slot_offset=40_000,
        )
        for update in range(1, OBJECTIVE_AGGREGATION_UPDATES + 1)
        for micro in range(1, OBJECTIVES_PER_UPDATE + 1)
    ]
    assert slots == list(range(40_001, 41_601))
    assert PHASE_E9_HELDOUT_FLOW_STEPS == tuple(range(75, 107))
    assert not set(slots) & set(PHASE_E9_HELDOUT_FLOW_STEPS)
    assert _zero_positions(
        frozen_parent["sample_ids"],
        PHASE_E9_HELDOUT_FLOW_STEPS,
    ) == PHASE_E9_HELDOUT_ZERO_WEIGHT_POSITIONS


def test_training_zero_weight_schedule_is_frozen(frozen_parent) -> None:
    observed = []
    for update in range(1, 201):
        for micro, base_sample_id in enumerate(
            frozen_parent["sample_ids"],
            start=1,
        ):
            slot = objective_aggregation_flow_slot(
                update,
                micro,
                flow_slot_offset=40_000,
            )
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=3407,
                flow_step=slot,
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
                observed.append((update, micro, slot))
    assert tuple(observed) == PHASE_E9_EXPECTED_ZERO_WEIGHT_SLOTS


def test_positions_17_to_28_are_identity_only_reserved(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    reserved = verify_reserved_replication_cohort(
        cfg,
        used_sample_ids=frozen_parent["sample_ids"],
    )
    assert reserved["cohort_sha256"] == PHASE_E9_RESERVED_COHORT_SHA256
    assert reserved["identity_schedule_sha256"] == (
        PHASE_E9B_IDENTITY_SCHEDULE_SHA256
    )
    assert reserved["decoded_or_trained_by_e9a"] is False
    assert len(reserved["sample_ids"]) == 12
    assert not set(reserved["sample_ids"]) & set(
        frozen_parent["sample_ids"]
    )
    assert _zero_positions(
        reserved["sample_ids"],
        tuple(range(107, 139)),
    ) == PHASE_E9B_ZERO_WEIGHT_POSITIONS


def _probe(
    sample_ids: list[str],
    *,
    final_factor: float,
    harmed_indices: tuple[int, ...] = (),
) -> dict:
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        for flow_step in PHASE_E9_HELDOUT_FLOW_STEPS:
            factor = (
                1.2 if sample_index in harmed_indices else final_factor
            )
            rows.append(
                {
                    "action_loss": factor,
                    "base_sample_id": sample_id,
                    "flow_step": flow_step,
                }
            )
    return {
        "flow_steps": list(PHASE_E9_HELDOUT_FLOW_STEPS),
        "per_objective": rows,
        "sample_ids": sample_ids,
    }


def test_tail_bootstrap_and_classification_are_frozen(
    frozen_parent,
) -> None:
    sample_ids = frozen_parent["sample_ids"]
    initial = _probe(sample_ids, final_factor=1.0)
    raw_final = _probe(
        sample_ids,
        final_factor=0.9,
        harmed_indices=(0,),
    )
    normalized_final = _probe(sample_ids, final_factor=0.9)
    raw_tail = paired_tail_bootstrap(
        initial,
        raw_final,
        track_key="raw/A0",
    )
    normalized_tail = paired_tail_bootstrap(
        initial,
        normalized_final,
        track_key="normalized/A0",
    )
    assert raw_tail["bootstrap_replicates"] == (
        PHASE_E9_BOOTSTRAP_REPLICATES
    )
    assert raw_tail["familywise_comparisons"] == (
        PHASE_E9_FAMILYWISE_COMPARISONS
    )
    assert raw_tail["confirmed_worsened_sample_count"] == 1
    assert normalized_tail["confirmed_worsened_sample_count"] == 0

    performance = {
        f"{recipe}/{variant}": {"frozen": True}
        for recipe in ("raw", "normalized")
        for variant in ("A0", "A1")
    }
    tail = {
        "raw/A0": raw_tail,
        "raw/A1": {**raw_tail, "confirmed_worsened_sample_count": 0},
        "normalized/A0": normalized_tail,
        "normalized/A1": normalized_tail,
    }
    supported = classify_sample_tail_mitigation(
        performance,
        {"frozen": True},
        tail,
    )
    assert supported["classification"] == (
        "tail_mitigation_candidate_supported"
    )
    assert supported["independent_replication_candidate"] is True
    tail["raw/A0"] = {
        **raw_tail,
        "confirmed_worsened_sample_count": 0,
    }
    stable = classify_sample_tail_mitigation(
        performance,
        {"frozen": True},
        tail,
    )
    assert stable["classification"] == (
        "stable_normalized_candidate_without_tail_contrast"
    )
    failed = classify_sample_tail_mitigation(
        performance,
        {"frozen": False},
        tail,
    )
    assert failed["classification"] == (
        "sample_tail_mitigation_not_supported"
    )
    assert failed["independent_replication_candidate"] is False


def test_e9_refuses_before_model_load_or_run_state_write(
    monkeypatch,
) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E9A", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("E.9a loaded model without confirmation")

    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase_e9_sample_tail_mitigation."
        "_load_upstream_model",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E9A"):
        _run_phase_e9(
            cfg,
            resume=False,
            execution_repository={},
        )
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E9A"):
        run_phase_e9_sample_tail_mitigation(cfg)


def test_e9_source_and_runner_keep_locked_stages_out() -> None:
    source = inspect.getsource(_run_phase_e9)
    assert '"A2"' not in source
    assert '"A4"' not in source
    assert "run_rollout" not in source
    runner = Path(
        "scripts/run_thought3_phase_e9_sample_tail_mitigation.sh"
    ).read_text(encoding="utf-8")
    assert "single-GPU only" in runner
    assert "CONFIRM_THOUGHT3_PHASE_E9A" in runner
    assert "THOUGHT3_GPU_ID" in runner


def test_scope_rejects_learning_rate_or_step_selection_drift() -> None:
    cfg = load_thought3_config(CONFIG)
    with pytest.raises(RuntimeError, match="changes more"):
        _assert_phase_e9_scope(
            replace(
                cfg,
                training=replace(cfg.training, learning_rate=1e-4),
            )
        )
