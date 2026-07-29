"""Full-cohort objective aggregation for real Fast-WAM training.

This module intentionally lives beside, rather than inside, ``real_training``
so the executed Gate E.4 implementation remains frozen.  Gate E.5 changes one
optimizer input: every optimizer update is the mean of one unique action-flow
objective for each of eight source-filtered training samples.  Later protocols
may reuse the trainer only through an explicit frozen schedule identity.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from fastwam_ood_eval.thought3.checkpointing import (
    adapter_state_sha256,
    find_latest_checkpoint,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.injection import (
    ActionEncoderFutureInjector,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from fastwam_ood_eval.thought3.real_training import (
    DIVERSIFIED_HELDOUT_FLOW_STEPS,
    PreparedRealTrainingData,
    ProgressCallback,
    RealTrainingError,
    _checkpoint_expected,
    _checkpoint_manifest,
    _checkpoint_roundtrip,
    _flow_objective_identity,
    _flow_timestep_and_weight_scalars,
    _loss_for_real_sample,
    _ordered_samples,
    adapter_gradient_groups,
    build_real_adapter,
    evaluate_multiflow_subset_probe,
    multiflow_subset_outcome,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_thought3_output_path,
)
from fastwam_ood_eval.thought3.schemas import canonical_json


OBJECTIVE_AGGREGATION_UPDATES = 200
OBJECTIVES_PER_UPDATE = 8
OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET = 20_000
OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS = (
    DIVERSIFIED_HELDOUT_FLOW_STEPS
)
OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS = (
    (20, 3, 20_155),
    (22, 5, 20_173),
    (26, 4, 20_204),
    (41, 5, 20_325),
    (48, 8, 20_384),
    (51, 1, 20_401),
    (55, 2, 20_434),
    (61, 6, 20_486),
    (81, 5, 20_645),
    (84, 3, 20_667),
    (92, 3, 20_731),
    (94, 1, 20_745),
    (94, 5, 20_749),
    (108, 5, 20_861),
    (113, 8, 20_904),
    (118, 3, 20_939),
    (123, 2, 20_978),
    (130, 3, 21_035),
    (134, 8, 21_072),
    (151, 2, 21_202),
    (158, 5, 21_261),
    (173, 6, 21_382),
    (177, 3, 21_411),
    (190, 5, 21_517),
)


@dataclass(frozen=True)
class ObjectiveAggregationProtocol:
    """Frozen schedule/provenance identity for one full-cohort Gate."""

    gate_label: str
    checkpoint_marker_key: str
    flow_slot_offset: int
    expected_zero_weight_slots: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if (
            not self.gate_label
            or not self.checkpoint_marker_key
            or self.flow_slot_offset < 1
            or any(
                len(row) != 3
                or not 1 <= row[0] <= OBJECTIVE_AGGREGATION_UPDATES
                or not 1 <= row[1] <= OBJECTIVES_PER_UPDATE
                or row[2]
                != (
                    self.flow_slot_offset
                    + (row[0] - 1) * OBJECTIVES_PER_UPDATE
                    + row[1]
                )
                for row in self.expected_zero_weight_slots
            )
        ):
            raise ValueError("invalid objective-aggregation protocol")


PHASE_E5_OBJECTIVE_AGGREGATION_PROTOCOL = ObjectiveAggregationProtocol(
    gate_label="Gate E.5",
    checkpoint_marker_key="gate_e5_objective_aggregation",
    flow_slot_offset=OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
    expected_zero_weight_slots=(
        OBJECTIVE_AGGREGATION_EXPECTED_ZERO_WEIGHT_SLOTS
    ),
)


class ObjectiveAggregationTrainingError(RealTrainingError):
    """Raised when a frozen objective-aggregation contract is violated."""


def objective_aggregation_flow_slot(
    optimizer_update: int,
    micro_index: int,
    *,
    flow_slot_offset: int = OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
) -> int:
    """Map one update/micro pair to a unique slot outside all probe slots."""

    if (
        isinstance(optimizer_update, bool)
        or isinstance(micro_index, bool)
        or not isinstance(optimizer_update, int)
        or not isinstance(micro_index, int)
        or isinstance(flow_slot_offset, bool)
        or not isinstance(flow_slot_offset, int)
        or flow_slot_offset < 1
        or not 1 <= optimizer_update <= OBJECTIVE_AGGREGATION_UPDATES
        or not 1 <= micro_index <= OBJECTIVES_PER_UPDATE
    ):
        raise ObjectiveAggregationTrainingError(
            "Gate E.5 requires optimizer_update=1..200 and micro_index=1..8"
        )
    return (
        flow_slot_offset
        + (optimizer_update - 1) * OBJECTIVES_PER_UPDATE
        + micro_index
    )


def _validate_schedule_positions(
    rows: Sequence[Mapping[str, Any]],
    *,
    flow_slot_offset: int = OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
) -> None:
    if (
        not rows
        or len(rows)
        > OBJECTIVE_AGGREGATION_UPDATES * OBJECTIVES_PER_UPDATE
        or len(rows) % OBJECTIVES_PER_UPDATE
    ):
        raise ObjectiveAggregationTrainingError(
            "Gate E.5 objective schedule must be a nonempty full-update prefix"
        )
    for expected_objective, row in enumerate(rows, start=1):
        expected_update = (
            (expected_objective - 1) // OBJECTIVES_PER_UPDATE + 1
        )
        expected_micro = (
            (expected_objective - 1) % OBJECTIVES_PER_UPDATE + 1
        )
        expected_slot = objective_aggregation_flow_slot(
            expected_update,
            expected_micro,
            flow_slot_offset=flow_slot_offset,
        )
        if (
            int(row["objective_index"]) != expected_objective
            or int(row["optimizer_update"]) != expected_update
            or int(row["micro_index"]) != expected_micro
            or int(row["cohort_sample_index"]) != expected_micro - 1
            or int(row["sample_cursor"]) != expected_objective
            or int(row["training_flow_slot"]) != expected_slot
            or int(row["flow_step"]) != expected_slot
        ):
            raise ObjectiveAggregationTrainingError(
                "Gate E.5 objective schedule is not contiguous/full-cohort"
            )


def objective_aggregation_schedule_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    flow_slot_offset: int = OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
) -> str:
    """Hash the complete observed sample/noise/timestep/weight schedule."""

    _validate_schedule_positions(
        rows,
        flow_slot_offset=flow_slot_offset,
    )
    payload = []
    for row in rows:
        payload.append(
            "\0".join(
                (
                    str(int(row["objective_index"])),
                    str(int(row["optimizer_update"])),
                    str(int(row["micro_index"])),
                    str(int(row["cohort_sample_index"])),
                    str(row["base_sample_id"]),
                    str(int(row["training_flow_slot"])),
                    str(int(row["flow_step"])),
                    str(int(row["action_noise_seed"])),
                    str(int(row["action_timestep_seed"])),
                    str(row["flow_objective_sha256"]),
                    repr(float(row["timestep"])),
                    repr(float(row["action_weight"])),
                )
            )
        )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def objective_aggregation_identity_schedule_sha256(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
    update_count: int = OBJECTIVE_AGGREGATION_UPDATES,
    flow_slot_offset: int = OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
) -> str:
    """Hash the schedule knowable before model loading or result inspection."""

    if (
        len(sample_ids) != OBJECTIVES_PER_UPDATE
        or len(set(sample_ids)) != OBJECTIVES_PER_UPDATE
        or update_count < 1
        or update_count > OBJECTIVE_AGGREGATION_UPDATES
    ):
        raise ObjectiveAggregationTrainingError(
            "Gate E.5 identity schedule requires 8 unique samples and 1..200 updates"
        )
    payload = []
    for optimizer_update in range(1, update_count + 1):
        for micro_index, base_sample_id in enumerate(sample_ids, start=1):
            objective_index = (
                (optimizer_update - 1) * OBJECTIVES_PER_UPDATE
                + micro_index
            )
            flow_slot = objective_aggregation_flow_slot(
                optimizer_update,
                micro_index,
                flow_slot_offset=flow_slot_offset,
            )
            identity = _flow_objective_identity(
                base_sample_id=str(base_sample_id),
                train_seed=train_seed,
                flow_step=flow_slot,
            )
            payload.append(
                "\0".join(
                    (
                        str(objective_index),
                        str(optimizer_update),
                        str(micro_index),
                        str(micro_index - 1),
                        str(base_sample_id),
                        str(flow_slot),
                        str(int(identity["action_noise_seed"])),
                        str(int(identity["action_timestep_seed"])),
                        str(identity["flow_objective_sha256"]),
                    )
                )
            )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def objective_aggregation_metric_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash complete ordered metric rows, not only schedule identity fields."""

    return hashlib.sha256(
        "\n".join(
            canonical_json(dict(row)) for row in rows
        ).encode("utf-8")
    ).hexdigest()


def _backward_mean_objective(
    loss: Tensor,
    *,
    accumulation_factor: int,
) -> None:
    """Backpropagate one contribution to an arithmetic-mean objective."""

    if (
        loss.ndim != 0
        or isinstance(accumulation_factor, bool)
        or not isinstance(accumulation_factor, int)
        or accumulation_factor <= 0
    ):
        raise ObjectiveAggregationTrainingError(
            "mean-objective backward requires a scalar loss and positive integer factor"
        )
    (loss / accumulation_factor).backward()


def _objective_rows_for_resume(
    path: Path,
    *,
    start_update: int,
    flow_slot_offset: int = OBJECTIVE_AGGREGATION_FLOW_SLOT_OFFSET,
) -> list[dict[str, Any]]:
    if not path.is_file():
        if start_update:
            raise ObjectiveAggregationTrainingError(
                "Gate E.5 checkpoint exists without objective metrics"
            )
        return []
    rows = load_jsonl(path)
    _validate_schedule_positions(
        rows,
        flow_slot_offset=flow_slot_offset,
    )
    required = start_update * OBJECTIVES_PER_UPDATE
    if len(rows) < required:
        raise ObjectiveAggregationTrainingError(
            "Gate E.5 checkpoint exceeds committed objective metrics"
        )
    return rows[:required]


def _update_rows_for_resume(
    path: Path,
    *,
    start_update: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        if start_update:
            raise ObjectiveAggregationTrainingError(
                "Gate E.5 checkpoint exists without update metrics"
            )
        return []
    rows = load_jsonl(path)
    updates = [int(row["optimizer_update"]) for row in rows]
    if updates != list(range(1, len(rows) + 1)):
        raise ObjectiveAggregationTrainingError(
            "Gate E.5 optimizer-update metrics are not contiguous"
        )
    if len(rows) < start_update:
        raise ObjectiveAggregationTrainingError(
            "Gate E.5 checkpoint exceeds committed update metrics"
        )
    return rows[:start_update]


def _first_gradient_updates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int | None, int | None]:
    first_non_gate: int | None = None
    first_projector: int | None = None
    first_attention: int | None = None
    for row in rows:
        update = int(row["optimizer_update"])
        groups = row["gradient_groups"]
        if (
            first_non_gate is None
            and int(groups["non_gate"]["nonzero_element_count"]) > 0
        ):
            first_non_gate = update
        if (
            first_projector is None
            and int(
                groups["future_projector"]["nonzero_element_count"]
            )
            > 0
        ):
            first_projector = update
        if (
            first_attention is None
            and int(groups["attention"]["nonzero_element_count"]) > 0
        ):
            first_attention = update
    return first_non_gate, first_projector, first_attention


def _validate_resume_metric_provenance(
    *,
    global_step: int,
    sample_cursor: int,
    extra: Mapping[str, Any],
    objective_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    identity_schedule_sha256: str,
    protocol: ObjectiveAggregationProtocol = (
        PHASE_E5_OBJECTIVE_AGGREGATION_PROTOCOL
    ),
) -> None:
    """Bind a resumed checkpoint to both committed metric prefixes."""

    expected_objectives = global_step * OBJECTIVES_PER_UPDATE
    if (
        global_step < 1
        or global_step > OBJECTIVE_AGGREGATION_UPDATES
        or sample_cursor != expected_objectives
        or len(objective_rows) != expected_objectives
        or len(update_rows) != global_step
        or extra.get(protocol.checkpoint_marker_key) is not True
        or extra.get("gradient_reduction") != "arithmetic_mean"
        or int(extra.get("objectives_per_update", -1))
        != OBJECTIVES_PER_UPDATE
        or int(extra.get("objective_count", -1))
        != expected_objectives
        or int(extra.get("training_flow_slot_offset", -1))
        != protocol.flow_slot_offset
        or list(extra.get("heldout_flow_steps", []))
        != list(OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS)
        or extra.get("identity_schedule_sha256")
        != identity_schedule_sha256
        or extra.get("train_flow_schedule_sha256")
        != objective_aggregation_schedule_sha256(
            objective_rows,
            flow_slot_offset=protocol.flow_slot_offset,
        )
        or extra.get("objective_metrics_prefix_sha256")
        != objective_aggregation_metric_rows_sha256(objective_rows)
        or extra.get("update_metrics_prefix_sha256")
        != objective_aggregation_metric_rows_sha256(update_rows)
    ):
        raise ObjectiveAggregationTrainingError(
            f"{protocol.gate_label} checkpoint/metric-prefix "
            "provenance mismatch"
        )


def run_full_cohort_objective_aggregation(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
    device: str,
    progress: ProgressCallback | None = None,
    protocol: ObjectiveAggregationProtocol = (
        PHASE_E5_OBJECTIVE_AGGREGATION_PROTOCOL
    ),
) -> dict[str, Any]:
    """Train one frozen track with eight mean-aggregated objectives/update."""

    if (
        cfg.runtime.backend != "fastwam"
        or cfg.variant not in {"A0", "A1"}
        or cfg.training.max_steps != OBJECTIVE_AGGREGATION_UPDATES
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps
        != OBJECTIVES_PER_UPDATE
        or cfg.training.checkpoint_interval != 50
        or device != "cuda:0"
        or cfg.runtime.device != device
    ):
        raise ObjectiveAggregationTrainingError(
            f"{protocol.gate_label} requires real A0/A1, cuda:0, "
            "200 optimizer updates, "
            "microbatch 1, mean accumulation 8, checkpoint 50"
        )
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    status_path = output / "run_status.json"
    objective_metrics_path = output / "train_objective_metrics.jsonl"
    update_metrics_path = output / "train_update_metrics.jsonl"
    probe_path = output / "heldout_multiflow_metrics.jsonl"
    state_path = output / "training_state.json"
    manifest_path = output / "training_manifest.json"
    checkpoints_root = output / "checkpoints"
    if manifest_path.is_file() and resume:
        existing = load_json(manifest_path)
        if (
            existing.get("status") == "complete"
            and int(existing.get("completed_steps", -1))
            == OBJECTIVE_AGGREGATION_UPDATES
        ):
            return existing
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(
            f"{protocol.gate_label} track output exists: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "gradient_accumulation_steps": OBJECTIVES_PER_UPDATE,
            "learning_rate": cfg.training.learning_rate,
            "loss_reduction": "arithmetic_mean",
            "started_at_unix_s": time.time(),
            "status": "running",
            "variant": cfg.variant,
        },
    )

    samples = _ordered_samples(
        (sample for sample in prepared.samples if sample.split == "train"),
        seed=cfg.training.train_seed,
    )
    if (
        len(samples) != OBJECTIVES_PER_UPDATE
        or len(prepared.samples) != OBJECTIVES_PER_UPDATE
        or any(sample.split != "train" for sample in prepared.samples)
    ):
        raise ObjectiveAggregationTrainingError(
            f"{protocol.gate_label} requires exactly 8 "
            "source-filtered train samples"
        )
    sample_ids = [sample.base_sample_id for sample in samples]
    identity_schedule_sha256 = (
        objective_aggregation_identity_schedule_sha256(
            sample_ids,
            train_seed=cfg.training.train_seed,
            flow_slot_offset=protocol.flow_slot_offset,
        )
    )
    adapter = build_real_adapter(cfg, device=device)
    initial_adapter_sha256 = adapter_state_sha256(adapter.state_dict())
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {
        id(parameter) for parameter in adapter.parameters()
    }:
        raise ObjectiveAggregationTrainingError(
            f"{protocol.gate_label} optimizer contains "
            "non-Adapter parameters"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ObjectiveAggregationTrainingError(
            f"{protocol.gate_label} frozen Fast-WAM parameter "
            "became trainable"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    start_update = 0
    sample_cursor = 0
    loaded_manifest = None
    latest = find_latest_checkpoint(checkpoints_root) if resume else None
    if latest is not None:
        loaded_manifest = load_adapter_checkpoint(
            latest,
            adapter=adapter,
            optimizer=optimizer,
            expected=_checkpoint_expected(
                cfg,
                prepared,
                frozen_parameter_sha256=frozen_parameter_sha256,
            ),
        )
        start_update = loaded_manifest.global_step
        sample_cursor = loaded_manifest.sample_cursor
        if sample_cursor != start_update * OBJECTIVES_PER_UPDATE:
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} checkpoint cursor is not update*8"
            )
    elif (
        resume
        and checkpoints_root.exists()
        and any(checkpoints_root.iterdir())
    ):
        raise ObjectiveAggregationTrainingError(
            f"{protocol.gate_label} resume requested without "
            "a valid checkpoint"
        )

    existing_objectives = _objective_rows_for_resume(
        objective_metrics_path,
        start_update=start_update,
        flow_slot_offset=protocol.flow_slot_offset,
    )
    existing_updates = _update_rows_for_resume(
        update_metrics_path,
        start_update=start_update,
    )
    if loaded_manifest is not None:
        _validate_resume_metric_provenance(
            global_step=loaded_manifest.global_step,
            sample_cursor=loaded_manifest.sample_cursor,
            extra=loaded_manifest.extra,
            objective_rows=existing_objectives,
            update_rows=existing_updates,
            identity_schedule_sha256=identity_schedule_sha256,
            protocol=protocol,
        )
    if state_path.is_file():
        state = load_json(state_path)
        if (
            state["config_fingerprint"] != cfg.fingerprint
            or state["frozen_parameter_sha256"]
            != frozen_parameter_sha256
            or state["initial_adapter_sha256"]
            != initial_adapter_sha256
            or list(state["sample_ids"]) != sample_ids
            or int(state["training_flow_slot_offset"])
            != protocol.flow_slot_offset
            or int(state["objectives_per_update"])
            != OBJECTIVES_PER_UPDATE
            or state["loss_reduction"] != "arithmetic_mean"
            or state["identity_schedule_sha256"]
            != identity_schedule_sha256
            or list(state["heldout_flow_steps"])
            != list(OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS)
        ):
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} training-state provenance mismatch"
            )
        initial_probe = dict(state["initial_probe"])
    else:
        if start_update:
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} checkpoint exists without "
                "initial state"
            )
        initial_probe = evaluate_multiflow_subset_probe(
            cfg,
            model,
            adapter,
            injector,
            samples,
            flow_steps=OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS,
            device=device,
        )
        if (
            float(
                initial_probe[
                    "max_gated_delta_to_action_hidden_ratio"
                ]
            )
            != 0
            or any(
                float(row["gated_delta_nonzero_fraction"]) != 0
                for row in initial_probe["per_objective"]
            )
        ):
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} zero-gate initialization is "
                "not exact identity"
            )
        state = {
            "cache_fingerprint": prepared.cache_fingerprint,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "frozen_parameter_sha256": frozen_parameter_sha256,
            "heldout_flow_steps": list(
                OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS
            ),
            "identity_schedule_sha256": identity_schedule_sha256,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_probe": initial_probe,
            "loss_reduction": "arithmetic_mean",
            "objectives_per_update": OBJECTIVES_PER_UPDATE,
            "sample_ids": sample_ids,
            "split_fingerprint": prepared.split_fingerprint,
            "training_flow_slot_offset": (
                protocol.flow_slot_offset
            ),
            "uses_ground_truth_future_input": False,
            "variant": cfg.variant,
        }
        atomic_write_json(state_path, state)

    if probe_path.is_file():
        probe_rows = load_jsonl(probe_path)
        if (
            not probe_rows
            or int(probe_rows[0]["global_step"]) != 0
            or list(probe_rows[0]["sample_ids"]) != sample_ids
            or float(probe_rows[0]["mean_action_loss"])
            != float(initial_probe["mean_action_loss"])
            or [int(row["global_step"]) for row in probe_rows]
            not in ([0], [0, OBJECTIVE_AGGREGATION_UPDATES])
        ):
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} held-out probe history is invalid"
            )
        if (
            start_update < OBJECTIVE_AGGREGATION_UPDATES
            and len(probe_rows) != 1
        ):
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} final probe precedes "
                "final checkpoint"
            )
    else:
        if start_update:
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} checkpoint exists without "
                "initial probe"
            )
        probe_rows = [
            {
                **initial_probe,
                "global_step": 0,
                "learning_rate": cfg.training.learning_rate,
            }
        ]
        atomic_write_jsonl(probe_path, probe_rows)

    (
        first_non_gate_update,
        first_projector_update,
        first_attention_update,
    ) = _first_gradient_updates(existing_updates)
    new_objectives: list[dict[str, Any]] = []
    new_updates: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for update_offset in range(
            start_update,
            OBJECTIVE_AGGREGATION_UPDATES,
        ):
            optimizer_update = update_offset + 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            update_started = time.perf_counter()
            gate_before = float(adapter.gate.detach().float().cpu())
            raw_losses: list[float] = []
            action_weights: list[float] = []
            gate_contributions: list[float] = []
            update_objective_rows: list[dict[str, Any]] = []

            for micro_index, sample in enumerate(samples, start=1):
                objective_index = (
                    (optimizer_update - 1) * OBJECTIVES_PER_UPDATE
                    + micro_index
                )
                flow_slot = objective_aggregation_flow_slot(
                    optimizer_update,
                    micro_index,
                    flow_slot_offset=protocol.flow_slot_offset,
                )
                identity = _flow_objective_identity(
                    base_sample_id=sample.base_sample_id,
                    train_seed=cfg.training.train_seed,
                    flow_step=flow_slot,
                )
                timestep, action_weight = (
                    _flow_timestep_and_weight_scalars(
                        model,
                        sample,
                        train_seed=cfg.training.train_seed,
                        step=flow_slot,
                        device=device,
                    )
                )
                if optimizer_update <= 2 and action_weight <= 0:
                    raise ObjectiveAggregationTrainingError(
                        f"{protocol.gate_label} first two full cohorts "
                        "require positive weights"
                    )
                cumulative_gate_before = (
                    0.0
                    if adapter.gate.grad is None
                    else float(adapter.gate.grad.detach().float().cpu())
                )
                loss = _loss_for_real_sample(
                    cfg,
                    model,
                    adapter,
                    injector,
                    sample,
                    step=flow_slot,
                    device=device,
                )
                if not bool(torch.isfinite(loss).item()):
                    raise ObjectiveAggregationTrainingError(
                        f"{protocol.gate_label} action loss is NaN/Inf"
                    )
                raw_loss = float(loss.detach().float().cpu())
                diagnostics = adapter.last_diagnostics
                if (
                    diagnostics is None
                    or diagnostics.action_hidden_norm <= 0
                ):
                    raise ObjectiveAggregationTrainingError(
                        f"{protocol.gate_label} Adapter diagnostics "
                        "are missing"
                    )
                _backward_mean_objective(
                    loss,
                    accumulation_factor=OBJECTIVES_PER_UPDATE,
                )
                if adapter.gate.grad is None:
                    raise ObjectiveAggregationTrainingError(
                        f"{protocol.gate_label} gate gradient is missing"
                    )
                cumulative_gate_after = float(
                    adapter.gate.grad.detach().float().cpu()
                )
                gate_contribution = (
                    cumulative_gate_after - cumulative_gate_before
                )
                if not math.isfinite(gate_contribution):
                    raise ObjectiveAggregationTrainingError(
                        f"{protocol.gate_label} gate contribution "
                        "is non-finite"
                    )
                sample_cursor += 1
                row = {
                    **identity,
                    "action_hidden_norm": diagnostics.action_hidden_norm,
                    "action_loss": raw_loss,
                    "action_weight": action_weight,
                    "attention_residual_norm": (
                        diagnostics.attention_residual_norm
                    ),
                    "base_sample_id": sample.base_sample_id,
                    "cohort_sample_index": micro_index - 1,
                    "future_token_norm": diagnostics.future_token_norm,
                    "gate_gradient_contribution_mean_scaled": (
                        gate_contribution
                    ),
                    "gate_gradient_contribution_sign": (
                        1
                        if gate_contribution > 0
                        else -1
                        if gate_contribution < 0
                        else 0
                    ),
                    "gate_gradient_contribution_unscaled": (
                        gate_contribution * OBJECTIVES_PER_UPDATE
                    ),
                    "gate_gradient_cumulative": cumulative_gate_after,
                    "gate_raw_before_update": gate_before,
                    "gated_delta_nonzero_fraction": (
                        diagnostics.gated_delta_nonzero_fraction
                    ),
                    "gated_delta_norm": diagnostics.gated_delta_norm,
                    "gated_delta_to_action_hidden_ratio": (
                        diagnostics.gated_delta_norm
                        / diagnostics.action_hidden_norm
                    ),
                    "gradient_reduction": "arithmetic_mean",
                    "learning_rate": cfg.training.learning_rate,
                    "mean_scaled_backward_loss": (
                        raw_loss / OBJECTIVES_PER_UPDATE
                    ),
                    "micro_index": micro_index,
                    "nan_or_inf": False,
                    "objective_index": objective_index,
                    "optimizer_update": optimizer_update,
                    "sample_cursor": sample_cursor,
                    "timestep": timestep,
                    "training_flow_slot": flow_slot,
                    "variant": cfg.variant,
                    "zero_weight_objective": action_weight == 0,
                }
                if action_weight == 0 and raw_loss != 0:
                    raise ObjectiveAggregationTrainingError(
                        f"{protocol.gate_label} zero-weight objective "
                        "has nonzero loss"
                    )
                raw_losses.append(raw_loss)
                action_weights.append(action_weight)
                gate_contributions.append(gate_contribution)
                update_objective_rows.append(row)
                del loss

            groups = adapter_gradient_groups(adapter)
            if not all(bool(value["finite"]) for value in groups.values()):
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} Adapter gradient is non-finite"
                )
            if optimizer_update == 1 and (
                float(groups["gate"]["l2"]) <= 0
                or int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                != 0
            ):
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} first-update zero-gate "
                    "contract failed"
                )
            if optimizer_update == 2 and (
                int(
                    groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                <= 0
                or int(
                    groups["attention"]["nonzero_element_count"]
                )
                <= 0
            ):
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} second-update non-gate "
                    "gradient contract failed"
                )
            if (
                first_non_gate_update is None
                and int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                > 0
            ):
                first_non_gate_update = optimizer_update
            if (
                first_projector_update is None
                and int(
                    groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                > 0
            ):
                first_projector_update = optimizer_update
            if (
                first_attention_update is None
                and int(
                    groups["attention"]["nonzero_element_count"]
                )
                > 0
            ):
                first_attention_update = optimizer_update
            backbone_grads = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if backbone_grads:
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} frozen Fast-WAM "
                    "received gradients: "
                    f"{backbone_grads[:5]}"
                )
            gate_gradient = float(
                adapter.gate.grad.detach().float().cpu()
            )
            if not math.isclose(
                sum(gate_contributions),
                gate_gradient,
                rel_tol=1e-5,
                abs_tol=1e-8,
            ):
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} per-objective gate "
                    "contributions do not sum "
                    "to the accumulated mean gradient"
                )
            absolute_contribution_sum = sum(
                abs(value) for value in gate_contributions
            )
            cancellation_ratio = (
                abs(gate_gradient) / absolute_contribution_sum
                if absolute_contribution_sum
                else 0.0
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            gate_after = float(adapter.gate.detach().float().cpu())
            peak_memory_mib = (
                int(torch.cuda.max_memory_allocated(device)) / 2**20
            )
            if (
                peak_memory_mib
                >= cfg.runtime.max_gpu_memory_gb * 1024
            ):
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} exceeded the frozen "
                    "GPU-memory ceiling"
                )
            update_time_ms = (
                time.perf_counter() - update_started
            ) * 1000.0
            for row in update_objective_rows:
                row["gate_raw_after_update"] = gate_after
                row["optimizer_update_peak_memory_mib"] = (
                    peak_memory_mib
                )
                row["optimizer_update_time_ms"] = update_time_ms
            update_row = {
                "action_weight_mean": statistics.fmean(action_weights),
                "action_weight_sum": sum(action_weights),
                "gate_gradient": gate_gradient,
                "gate_gradient_absolute_contribution_sum": (
                    absolute_contribution_sum
                ),
                "gate_gradient_cancellation_ratio": cancellation_ratio,
                "gate_gradient_sign": (
                    1
                    if gate_gradient > 0
                    else -1
                    if gate_gradient < 0
                    else 0
                ),
                "gate_raw_after_update": gate_after,
                "gate_raw_before_update": gate_before,
                "gradient_groups": groups,
                "gradient_reduction": "arithmetic_mean",
                "learning_rate": cfg.training.learning_rate,
                "mean_action_loss": statistics.fmean(raw_losses),
                "nan_or_inf": False,
                "objective_count": OBJECTIVES_PER_UPDATE,
                "objective_index_end": sample_cursor,
                "objective_index_start": (
                    sample_cursor - OBJECTIVES_PER_UPDATE + 1
                ),
                "optimizer_update": optimizer_update,
                "peak_memory_mib": peak_memory_mib,
                "sample_cursor": sample_cursor,
                "summed_action_loss": sum(raw_losses),
                "update_time_ms": update_time_ms,
                "variant": cfg.variant,
                "zero_weight_objective_count": sum(
                    weight == 0 for weight in action_weights
                ),
            }
            new_objectives.extend(update_objective_rows)
            new_updates.append(update_row)

            should_checkpoint = (
                optimizer_update % cfg.training.checkpoint_interval == 0
                or optimizer_update == OBJECTIVE_AGGREGATION_UPDATES
            )
            if should_checkpoint:
                committed_objectives = [
                    *existing_objectives,
                    *new_objectives,
                ]
                committed_updates = [
                    *existing_updates,
                    *new_updates,
                ]
                atomic_write_jsonl(
                    objective_metrics_path,
                    committed_objectives,
                )
                atomic_write_jsonl(
                    update_metrics_path,
                    committed_updates,
                )
                checkpoint = (
                    checkpoints_root
                    / f"step_{optimizer_update:08d}"
                )
                save_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    optimizer=optimizer,
                    manifest=_checkpoint_manifest(
                        cfg,
                        adapter,
                        split_fingerprint=prepared.split_fingerprint,
                        cache_fingerprint=prepared.cache_fingerprint,
                        frozen_parameter_sha256=(
                            frozen_parameter_sha256
                        ),
                        global_step=optimizer_update,
                        sample_cursor=sample_cursor,
                        train_sample_count=len(samples),
                        extra={
                            protocol.checkpoint_marker_key: True,
                            "gradient_reduction": "arithmetic_mean",
                            "heldout_flow_steps": list(
                                OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS
                            ),
                            "identity_schedule_sha256": (
                                identity_schedule_sha256
                            ),
                            "objective_count": len(
                                committed_objectives
                            ),
                            "objective_metrics_prefix_sha256": (
                                objective_aggregation_metric_rows_sha256(
                                    committed_objectives
                                )
                            ),
                            "objectives_per_update": (
                                OBJECTIVES_PER_UPDATE
                            ),
                            "subset_sample_count": len(samples),
                            "train_flow_schedule_sha256": (
                                objective_aggregation_schedule_sha256(
                                    committed_objectives,
                                    flow_slot_offset=(
                                        protocol.flow_slot_offset
                                    ),
                                )
                            ),
                            "training_flow_slot_offset": (
                                protocol.flow_slot_offset
                            ),
                            "update_metrics_prefix_sha256": (
                                objective_aggregation_metric_rows_sha256(
                                    committed_updates
                                )
                            ),
                        },
                    ),
                )
                if progress is not None:
                    progress(
                        "objective_aggregation_checkpoint",
                        {
                            "learning_rate": (
                                cfg.training.learning_rate
                            ),
                            "objective_count": len(
                                committed_objectives
                            ),
                            "optimizer_update": optimizer_update,
                            "variant": cfg.variant,
                            "zero_weight_objectives": sum(
                                bool(row["zero_weight_objective"])
                                for row in committed_objectives
                            ),
                        },
                    )
            torch.cuda.empty_cache()

        all_objectives = [*existing_objectives, *new_objectives]
        all_updates = [*existing_updates, *new_updates]
        atomic_write_jsonl(objective_metrics_path, all_objectives)
        atomic_write_jsonl(update_metrics_path, all_updates)
        if (
            len(all_objectives)
            != OBJECTIVE_AGGREGATION_UPDATES
            * OBJECTIVES_PER_UPDATE
            or len(all_updates) != OBJECTIVE_AGGREGATION_UPDATES
        ):
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} did not commit 1600 "
                "objectives and 200 updates"
            )
        final_probe_rows = [
            row
            for row in probe_rows
            if int(row["global_step"])
            == OBJECTIVE_AGGREGATION_UPDATES
        ]
        if final_probe_rows:
            if len(final_probe_rows) != 1:
                raise ObjectiveAggregationTrainingError(
                    f"{protocol.gate_label} final probe is duplicated"
                )
            final_probe = final_probe_rows[0]
        else:
            final_probe = evaluate_multiflow_subset_probe(
                cfg,
                model,
                adapter,
                injector,
                samples,
                flow_steps=OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS,
                device=device,
            )
            final_probe = {
                **final_probe,
                "global_step": OBJECTIVE_AGGREGATION_UPDATES,
                "learning_rate": cfg.training.learning_rate,
            }
            probe_rows.append(final_probe)
            atomic_write_jsonl(probe_path, probe_rows)
        final_outcome = multiflow_subset_outcome(
            initial_probe,
            final_probe,
        )
        latest_checkpoint = find_latest_checkpoint(checkpoints_root)
        if latest_checkpoint is None:
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} wrote no checkpoint"
            )
        roundtrip = _checkpoint_roundtrip(
            cfg,
            adapter,
            optimizer,
            latest_checkpoint,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            device=device,
        )
        schedule_sha256 = objective_aggregation_schedule_sha256(
            all_objectives,
            flow_slot_offset=protocol.flow_slot_offset,
        )
        zero_slots = tuple(
            (
                int(row["optimizer_update"]),
                int(row["micro_index"]),
                int(row["training_flow_slot"]),
            )
            for row in all_objectives
            if float(row["action_weight"]) == 0
        )
        if (
            zero_slots
            != protocol.expected_zero_weight_slots
        ):
            raise ObjectiveAggregationTrainingError(
                f"{protocol.gate_label} zero-weight objective "
                "schedule changed"
            )
        result = {
            "adapter_fingerprint": cfg.adapter_structural_fingerprint,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_roundtrip": roundtrip,
            "completed_objectives": len(all_objectives),
            "completed_steps": OBJECTIVE_AGGREGATION_UPDATES,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "device": device,
            "final_gate_raw": float(
                adapter.gate.detach().float().cpu()
            ),
            "final_probe": final_probe,
            "first_attention_nonzero_gradient_update": (
                first_attention_update
            ),
            "first_non_gate_nonzero_gradient_update": (
                first_non_gate_update
            ),
            "first_projector_nonzero_gradient_update": (
                first_projector_update
            ),
            "gradient_reduction": "arithmetic_mean",
            "heldout_flow_steps": list(
                OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS
            ),
            "identity_schedule_sha256": identity_schedule_sha256,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_probe": initial_probe,
            "learning_rate": cfg.training.learning_rate,
            "max_peak_memory_mib": max(
                float(row["peak_memory_mib"])
                for row in all_updates
            ),
            "mean_optimizer_update_time_ms": statistics.fmean(
                float(row["update_time_ms"]) for row in all_updates
            ),
            "objective_metrics": str(objective_metrics_path),
            "objectives_per_update": OBJECTIVES_PER_UPDATE,
            "optimizer_parameter_scope": "adapter_only",
            "outcome": final_outcome,
            "probe_metrics": str(probe_path),
            "resumed_from_update": start_update,
            "sample_count": len(samples),
            "sample_ids": sample_ids,
            "status": "complete",
            "train_flow_schedule_sha256": schedule_sha256,
            "train_flow_slot_end": objective_aggregation_flow_slot(
                OBJECTIVE_AGGREGATION_UPDATES,
                OBJECTIVES_PER_UPDATE,
                flow_slot_offset=protocol.flow_slot_offset,
            ),
            "train_flow_slot_start": objective_aggregation_flow_slot(
                1,
                1,
                flow_slot_offset=protocol.flow_slot_offset,
            ),
            "trainable_parameter_count": (
                adapter.trainable_parameter_count
            ),
            "training_flow_slot_offset": (
                protocol.flow_slot_offset
            ),
            "update_metrics": str(update_metrics_path),
            "uses_development_outcomes": False,
            "uses_ground_truth_future_input": False,
            "uses_ood_or_success_outcomes": False,
            "variant": cfg.variant,
            "wall_s_this_invocation": time.perf_counter() - started,
            "zero_weight_objective_count": len(zero_slots),
            "zero_weight_slots": [list(value) for value in zero_slots],
        }
        atomic_write_json(manifest_path, result)
        atomic_write_json(
            status_path,
            {
                "completed_objectives": len(all_objectives),
                "completed_steps": OBJECTIVE_AGGREGATION_UPDATES,
                "finished_at_unix_s": time.time(),
                "learning_rate": cfg.training.learning_rate,
                "status": "complete",
                "variant": cfg.variant,
            },
        )
        return result
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_unix_s": time.time(),
                "learning_rate": cfg.training.learning_rate,
                "status": "failed",
                "variant": cfg.variant,
            },
        )
        raise
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()
