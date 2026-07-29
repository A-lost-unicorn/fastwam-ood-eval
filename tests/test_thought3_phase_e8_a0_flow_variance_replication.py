from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_e8_a0_flow_variance_replication import (
    PHASE_E8_BOOTSTRAP_REPLICATES,
    PHASE_E8_CHECKPOINT_STEPS,
    PHASE_E8_CONFIG_FINGERPRINT,
    PHASE_E8_E7_TARGET_SAMPLE_IDS,
    PHASE_E8_FLOW_BLOCK_A,
    PHASE_E8_FLOW_BLOCK_B,
    PHASE_E8_FLOW_STEPS,
    PHASE_E8_IDENTITY_SCHEDULE_SHA256,
    PHASE_E8_PROBE_OBJECTIVES,
    PHASE_E8_ROOT,
    PHASE_E8_SCHEMA,
    PHASE_E8_ZERO_WEIGHT_POSITIONS,
    _aggregate_probe_rows,
    _assert_frozen_probe_design,
    _assert_phase_e8_scope,
    _probe_checks,
    _run_phase_e8,
    classify_a0_flow_variance,
    five_flow_resampling_sensitivity,
    paired_flow_bootstrap,
    probe_identity_schedule_sha256,
    run_phase_e8_a0_flow_variance_replication,
    verify_frozen_phase_e7,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
)


CONFIG = "configs/thought3/phase_e8_a0_flow_variance_replication.yaml"


@pytest.fixture(scope="module")
def frozen_parent() -> dict:
    return verify_frozen_phase_e7()


def test_phase_e8_scope_and_parent_are_frozen(frozen_parent) -> None:
    cfg = load_thought3_config(CONFIG)
    _assert_phase_e8_scope(cfg)
    assert cfg.fingerprint == PHASE_E8_CONFIG_FINGERPRINT
    assert cfg.experiment.output_dir == PHASE_E8_ROOT
    assert PHASE_E8_SCHEMA == (
        "thought3.phase_e8.a0_flow_variance_replication.v1"
    )
    assert PHASE_E8_CHECKPOINT_STEPS == (100, 200)
    assert frozen_parent["e7_classification"] == (
        "not_supported_no_material_late_degradation"
    )
    assert tuple(frozen_parent["e7_target_sample_ids"]) == (
        PHASE_E8_E7_TARGET_SAMPLE_IDS
    )


def test_phase_e8_new_flow_namespace_and_identity_are_frozen(
    frozen_parent,
) -> None:
    cfg = load_thought3_config(CONFIG)
    identity = _assert_frozen_probe_design(
        frozen_parent["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    assert identity == PHASE_E8_IDENTITY_SCHEDULE_SHA256
    assert PHASE_E8_FLOW_BLOCK_A == tuple(range(11, 43))
    assert PHASE_E8_FLOW_BLOCK_B == tuple(range(43, 75))
    assert PHASE_E8_FLOW_STEPS == tuple(range(11, 75))
    assert len(PHASE_E8_FLOW_STEPS) == 64
    assert not set(PHASE_E8_FLOW_STEPS) & set(range(0, 11))
    assert PHASE_E8_PROBE_OBJECTIVES == 1_536
    assert (
        probe_identity_schedule_sha256(
            frozen_parent["sample_ids"],
            train_seed=cfg.training.train_seed,
            flow_steps=PHASE_E8_FLOW_STEPS,
        )
        == PHASE_E8_IDENTITY_SCHEDULE_SHA256
    )

    observed = []
    for sample_index, base_sample_id in enumerate(
        frozen_parent["sample_ids"],
        start=1,
    ):
        for flow_step in PHASE_E8_FLOW_STEPS:
            identity_row = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=cfg.training.train_seed,
                flow_step=flow_step,
            )
            generator = torch.Generator(device="cpu").manual_seed(
                identity_row["action_timestep_seed"]
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
    assert tuple(observed) == PHASE_E8_ZERO_WEIGHT_POSITIONS


def _probe(
    sample_ids: tuple[str, ...],
    *,
    final: bool,
    worsened_indices: tuple[int, ...] = (),
) -> dict:
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        for flow_step in PHASE_E8_FLOW_STEPS:
            loss = 1.0
            if final:
                loss = (
                    1.20
                    if sample_index in worsened_indices
                    else 0.90
                )
            row = {
                "action_hidden_norm": 10.0,
                "action_loss": loss,
                "action_weight": 1.0,
                "attention_residual_norm": 2.0,
                "base_sample_id": sample_id,
                "flow_step": flow_step,
                "gated_delta_nonzero_fraction": 0.0 if not final else 1.0,
                "gated_delta_norm": 0.0 if not final else 2.0,
                "gated_delta_to_action_hidden_ratio": (
                    0.0 if not final else 0.2
                ),
                "latency_ms": 5.0,
                "peak_memory_mib": 100.0,
                "timestep": 100.0 + flow_step,
            }
            row.update(
                _flow_objective_identity(
                    base_sample_id=sample_id,
                    train_seed=3407,
                    flow_step=flow_step,
                )
            )
            rows.append(row)
    result = _aggregate_probe_rows(
        rows,
        sample_ids=sample_ids,
        flow_steps=PHASE_E8_FLOW_STEPS,
    )
    result.update(
        {
            "gate_raw": 0.0 if not final else 0.1,
            "identity_schedule_sha256": (
                probe_identity_schedule_sha256(
                    sample_ids,
                    train_seed=3407,
                    flow_steps=PHASE_E8_FLOW_STEPS,
                )
            ),
            "max_objective_peak_memory_mib": 100.0,
            "mean_objective_latency_ms": 5.0,
            "train_seed": 3407,
        }
    )
    return result


def test_large_probe_grid_and_paired_bootstrap_confirm_persistent_harm(
    frozen_parent,
) -> None:
    sample_ids = tuple(frozen_parent["sample_ids"])
    initial = _probe(sample_ids, final=False)
    final = _probe(
        sample_ids,
        final=True,
        worsened_indices=(1, 2),
    )
    # Synthetic rows do not reproduce the real frozen zero-weight locations,
    # but every RNG identity and the complete 512-objective grid are exact.
    checks = _probe_checks(
        final,
        sample_ids=sample_ids,
        train_seed=3407,
    )
    assert checks["complete_probe_grid"]
    assert checks["probe_rng_identity_exact"]
    stats = paired_flow_bootstrap(
        initial,
        final,
        checkpoint_step=200,
    )
    assert stats["bootstrap_replicates"] == PHASE_E8_BOOTSTRAP_REPLICATES
    assert stats["confirmed_worsened_sample_count"] == 2
    assert set(stats["confirmed_target_sample_ids"]) == set(
        PHASE_E8_E7_TARGET_SAMPLE_IDS[:2]
    )


def _panel_checks(passed: bool) -> dict[str, dict]:
    return {
        name: {"performance_checks": {"frozen": passed}}
        for name in ("full", "block_a", "block_b")
    }


def _classification_step(
    *,
    confirmed: tuple[str, ...],
    confirmed_targets: tuple[str, ...],
    panels_pass: bool,
) -> dict:
    return {
        "paired_bootstrap": {
            "confirmed_target_sample_ids": list(confirmed_targets),
            "confirmed_worsened_sample_ids": list(confirmed),
        },
        "panels": _panel_checks(panels_pass),
    }


def test_tail_risk_variance_and_mixed_classifications_are_mutually_exclusive() -> None:
    target_a, target_b, _ = PHASE_E8_E7_TARGET_SAMPLE_IDS
    tail = classify_a0_flow_variance(
        {
            100: _classification_step(
                confirmed=(),
                confirmed_targets=(),
                panels_pass=True,
            ),
            200: _classification_step(
                confirmed=(target_a, target_b),
                confirmed_targets=(target_a, target_b),
                panels_pass=False,
            ),
        }
    )
    assert tail["classification"] == (
        "persistent_target_tail_risk_supported"
    )
    assert tail["binary_answer"] == "tail_risk"
    assert tail["onset_subclassification"] == (
        "late_emergent_after_step100"
    )

    variance = classify_a0_flow_variance(
        {
            100: _classification_step(
                confirmed=(),
                confirmed_targets=(),
                panels_pass=True,
            ),
            200: _classification_step(
                confirmed=(),
                confirmed_targets=(),
                panels_pass=True,
            ),
        }
    )
    assert variance["classification"] == (
        "five_flow_panel_variance_supported"
    )
    assert variance["binary_answer"] == "five_flow_variance"

    mixed = classify_a0_flow_variance(
        {
            100: _classification_step(
                confirmed=(),
                confirmed_targets=(),
                panels_pass=True,
            ),
            200: _classification_step(
                confirmed=(target_a,),
                confirmed_targets=(target_a,),
                panels_pass=False,
            ),
        }
    )
    assert mixed["classification"] == "mixed_or_inconclusive"
    assert mixed["binary_answer"] == "inconclusive"


def test_five_flow_resampling_is_deterministic() -> None:
    sample_ids = tuple(
        (*PHASE_E8_E7_TARGET_SAMPLE_IDS, "s3", "s4", "s5", "s6", "s7")
    )
    initial = _probe(sample_ids, final=False)
    final = _probe(sample_ids, final=True)
    first = five_flow_resampling_sensitivity(
        initial,
        final,
        checkpoint_step=200,
    )
    second = five_flow_resampling_sensitivity(
        initial,
        final,
        checkpoint_step=200,
    )
    assert first == second
    assert first["five_flow_gate_pass_rate"] == 1.0
    assert first["five_flow_gate_fail_rate"] == 0.0


def test_phase_e8_refuses_before_any_output_write(monkeypatch) -> None:
    cfg = load_thought3_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_E8", raising=False)
    tracked = [
        PHASE_E8_ROOT / "run_status.json",
        PHASE_E8_ROOT / "gate_e8_result.json",
    ]
    before = {
        path: path.read_bytes() if path.is_file() else None
        for path in tracked
    }
    with pytest.raises(RuntimeError, match="CONFIRM_THOUGHT3_PHASE_E8"):
        run_phase_e8_a0_flow_variance_replication(cfg)
    after = {
        path: path.read_bytes() if path.is_file() else None
        for path in tracked
    }
    assert after == before


def test_phase_e8_recipe_drift_fails_closed() -> None:
    cfg = load_thought3_config(CONFIG)
    with pytest.raises(RuntimeError, match="changes more"):
        _assert_phase_e8_scope(
            replace(
                cfg,
                training=replace(cfg.training, learning_rate=1e-4),
            )
        )


def test_phase_e8_orchestrator_has_no_training_or_outcome_api() -> None:
    source = inspect.getsource(_run_phase_e8)
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "save_adapter_checkpoint" not in source
    signature = inspect.signature(
        run_phase_e8_a0_flow_variance_replication
    )
    assert set(signature.parameters) == {"cfg", "resume"}
