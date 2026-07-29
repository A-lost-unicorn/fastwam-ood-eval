"""Gate E.6: post-selection replication on an unused train cohort.

The 3e-4 learning rate is frozen *after* inspecting Gate E.5.  E.6 is
therefore a sequential replication, not an independent confirmatory test.
It runs only the matched A0/A1 pair on train-order positions 9--16 and uses
an entirely new objective-flow namespace.
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    load_thought3_config,
    validate_config,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    sha256_file,
)
from fastwam_ood_eval.thought3.objective_aggregation_training import (
    OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS,
    OBJECTIVE_AGGREGATION_UPDATES,
    OBJECTIVES_PER_UPDATE,
    ObjectiveAggregationProtocol,
    objective_aggregation_flow_slot,
    objective_aggregation_identity_schedule_sha256,
    run_full_cohort_objective_aggregation,
)
from fastwam_ood_eval.thought3.phase_c_smoke import _load_upstream_model
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    _matched_recipe_payload,
)
from fastwam_ood_eval.thought3.phase_e5_objective_aggregation import (
    PHASE_E5_CONFIG_FINGERPRINT,
    PHASE_E5_ROOT,
    _initial_probe_signature,
    _track_checks,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    _training_order_key,
    prepare_real_training_data,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E6_SCHEMA = "thought3.phase_e6.fresh_cohort_replication.v1"
PHASE_E6_EXPERIMENT_NAME = (
    "thought3_phase_e6_fresh_cohort_replication"
)
PHASE_E6_ROOT = Path(
    "outputs/thought3/phase_e6_fresh_cohort_replication_v1"
)
PHASE_E6_CONFIG = Path(
    "configs/thought3/phase_e6_fresh_cohort_replication.yaml"
)
PHASE_E6_CONFIG_FINGERPRINT = (
    "8cb2ab718eed2cc226491038423c92f1c59128246d966a2a9c3700d505f292d9"
)
PHASE_E6_LEARNING_RATE = 3e-4
PHASE_E6_LR_SLUG = "lr_3e_04"
PHASE_E6_TRAIN_ONLY_OFFSET = 8
PHASE_E6_FLOW_SLOT_OFFSET = 31_000
PHASE_E6_FROZEN_COHORT = (
    {
        "base_sample_id": (
            "9610d2aed3a6ddf382c514715ead977c9f9a25b56265b2705a9146ac28f6c0cc"
        ),
        "demonstration_id": "episode_000014",
        "episode_index": 14,
        "split": "train",
    },
    {
        "base_sample_id": (
            "75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b"
        ),
        "demonstration_id": "episode_000010",
        "episode_index": 10,
        "split": "train",
    },
    {
        "base_sample_id": (
            "5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44"
        ),
        "demonstration_id": "episode_000011",
        "episode_index": 11,
        "split": "train",
    },
    {
        "base_sample_id": (
            "8f34793be5e051e0d62c0397b83cc341f17b626bd73660968f48ff1f6339d1b9"
        ),
        "demonstration_id": "episode_000030",
        "episode_index": 30,
        "split": "train",
    },
    {
        "base_sample_id": (
            "8c00174e915504c49a3c69057f9c199af1654a6ecef414070c1657316b1e4418"
        ),
        "demonstration_id": "episode_000019",
        "episode_index": 19,
        "split": "train",
    },
    {
        "base_sample_id": (
            "461a673f2745ab243d99d617f4514a737644d44ba2fc5fdece8b45f347e51564"
        ),
        "demonstration_id": "episode_000038",
        "episode_index": 38,
        "split": "train",
    },
    {
        "base_sample_id": (
            "739baab482230ba4ee1ae9c0cccf5886268db9ee37c895435af6c6891d22c3b0"
        ),
        "demonstration_id": "episode_000000",
        "episode_index": 0,
        "split": "train",
    },
    {
        "base_sample_id": (
            "81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b"
        ),
        "demonstration_id": "episode_000012",
        "episode_index": 12,
        "split": "train",
    },
)
PHASE_E6_FROZEN_COHORT_SHA256 = (
    "6a354151d6d3e93335b66743f16be1908abc8d0fe835ee3811562b2eeb63d7c3"
)
PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256 = (
    "419b09a2ec30ce7bffc99c95aff1a343f77d39e83e77a752fc67bc984508febc"
)
PHASE_E6_EXPECTED_ZERO_WEIGHT_SLOTS = (
    (17, 1, 31129),
    (20, 2, 31154),
    (28, 4, 31220),
    (41, 5, 31325),
    (44, 5, 31349),
    (50, 2, 31394),
    (54, 7, 31431),
    (57, 6, 31454),
    (79, 3, 31627),
    (82, 2, 31650),
    (83, 7, 31663),
    (96, 2, 31762),
    (108, 2, 31858),
    (121, 3, 31963),
    (145, 8, 32160),
    (172, 8, 32376),
    (177, 6, 32414),
    (178, 6, 32422),
    (197, 3, 32571),
)
PHASE_E6_PROTOCOL = ObjectiveAggregationProtocol(
    gate_label="Gate E.6",
    checkpoint_marker_key="gate_e6_fresh_cohort_replication",
    flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET,
    expected_zero_weight_slots=PHASE_E6_EXPECTED_ZERO_WEIGHT_SLOTS,
)
PHASE_E5_FROZEN_ARTIFACTS = {
    "gate_e5_result.json": (
        "c797a98f646855a9b37caa7e251c97e8001d2d4aecb7efbcb5a539f77911f7bd"
    ),
    "run_status.json": (
        "cdc5944d35a03309230206ef817b75b17c1dbdea4b8f1706b98c1e7cec514f37"
    ),
    "pre_validation_result.json": (
        "63061d304a4a3c77c4e95f782d061be478b2e03a7dd88a39de8861f8ccde63ae"
    ),
    "data_preparation.json": (
        "ef95e5972ccabc455e7781afae19582f4f7880eb9e8800f0cd3e0a152f7261b6"
    ),
    "logs/phase_e5.log": (
        "fc334690b893555c09d36a2eb288e562b6e8454531d601570a544b91911d8582"
    ),
}


class PhaseE6GateError(RuntimeError):
    """Raised when the frozen E.6 replication contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    stage: str,
    values: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    import json

    payload = dict(values or {})
    payload.update(extra)
    print(
        json.dumps(
            {
                "phase": "E.6",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e6_scope(cfg: Thought3Config) -> None:
    e5 = load_thought3_config(
        "configs/thought3/phase_e5_objective_aggregation_diagnostic.yaml"
    )
    expected = replace(
        e5,
        variant="A1",
        experiment=replace(
            e5.experiment,
            name=PHASE_E6_EXPERIMENT_NAME,
            output_dir=PHASE_E6_ROOT,
        ),
        training=replace(
            e5.training,
            learning_rate=PHASE_E6_LEARNING_RATE,
        ),
    )
    observed_payload = cfg.to_dict()
    expected_payload = expected.to_dict()
    observed_payload.pop("source_path")
    expected_payload.pop("source_path")
    if observed_payload != expected_payload:
        raise PhaseE6GateError(
            "Gate E.6 changes more than name/output and post-E.5 LR"
        )
    if (
        cfg.experiment.name != PHASE_E6_EXPERIMENT_NAME
        or cfg.experiment.output_dir != PHASE_E6_ROOT
        or cfg.fingerprint != PHASE_E6_CONFIG_FINGERPRINT
    ):
        raise PhaseE6GateError("Gate E.6 frozen config identity changed")


def _require_phase_e6_confirmation() -> None:
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E6") != "YES":
        raise PhaseE6GateError(
            "set CONFIRM_THOUGHT3_PHASE_E6=YES for the real E.6 replication"
        )


def _cohort_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        "\0".join(
            (
                str(index),
                str(row["base_sample_id"]),
                str(row["demonstration_id"]),
                str(row["episode_index"]),
                str(row["split"]),
            )
        )
        for index, row in enumerate(rows, start=1)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_fresh_cohort(
    cfg: Thought3Config,
    *,
    e5_sample_ids: Sequence[str],
) -> dict[str, Any]:
    entries, plan = load_cache_plan(cfg.cache.root)
    k1_entries = [entry for entry in entries if entry.k == 1]
    train = sorted(
        (entry for entry in k1_entries if entry.split == "train"),
        key=lambda entry: _training_order_key(
            entry.identity.base_sample_id,
            seed=cfg.training.train_seed,
        ),
    )
    development_ids = {
        entry.identity.base_sample_id
        for entry in k1_entries
        if entry.split == "development"
    }
    selected = train[
        PHASE_E6_TRAIN_ONLY_OFFSET:
        PHASE_E6_TRAIN_ONLY_OFFSET + OBJECTIVES_PER_UPDATE
    ]
    rows = [
        {
            "base_sample_id": entry.identity.base_sample_id,
            "demonstration_id": entry.identity.demonstration_id,
            "episode_index": entry.identity.episode_index,
            "split": entry.split,
        }
        for entry in selected
    ]
    sample_ids = [str(row["base_sample_id"]) for row in rows]
    identity_sha = objective_aggregation_identity_schedule_sha256(
        sample_ids,
        train_seed=cfg.training.train_seed,
        flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET,
    )
    if (
        tuple(rows) != PHASE_E6_FROZEN_COHORT
        or _cohort_sha256(rows) != PHASE_E6_FROZEN_COHORT_SHA256
        or identity_sha != PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
        or len(set(sample_ids)) != OBJECTIVES_PER_UPDATE
        or set(sample_ids) & set(e5_sample_ids)
        or set(sample_ids) & development_ids
        or len({row["demonstration_id"] for row in rows})
        != OBJECTIVES_PER_UPDATE
        or plan["split_fingerprint"]
        != "ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb"
    ):
        raise PhaseE6GateError("Gate E.6 fresh cohort identity changed")
    return {
        "cohort_sha256": PHASE_E6_FROZEN_COHORT_SHA256,
        "development_overlap_count": 0,
        "e5_overlap_count": 0,
        "identity_schedule_sha256": identity_sha,
        "sample_ids": sample_ids,
        "samples": rows,
        "selection": "train order positions 9-16 (1-based)",
        "train_only_offset": PHASE_E6_TRAIN_ONLY_OFFSET,
    }


def verify_frozen_phase_e5() -> dict[str, Any]:
    artifact_sha256: dict[str, str] = {}
    for name, expected in PHASE_E5_FROZEN_ARTIFACTS.items():
        path = PHASE_E5_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise PhaseE6GateError(
                f"frozen Gate E.5 artifact changed/missing: {path}"
            )
        artifact_sha256[str(path)] = expected
    result = load_json(PHASE_E5_ROOT / "gate_e5_result.json")
    status = load_json(PHASE_E5_ROOT / "run_status.json")
    a0 = result["tracks"][PHASE_E6_LR_SLUG]["A0"]
    a1 = result["tracks"][PHASE_E6_LR_SLUG]["A1"]
    if (
        result.get("schema_version")
        != "thought3.phase_e5.objective_aggregation.v1"
        or result.get("config_fingerprint") != PHASE_E5_CONFIG_FINGERPRINT
        or result.get("status") != "failed"
        or result.get("gate_e5_passed") is not False
        or result.get("selected_learning_rate") is not None
        or result.get("selected_lr_slug") is not None
        or status.get("status") != "failed"
        or status.get("gate_e5_passed") is not False
        or not all(a0["execution_checks"].values())
        or not all(a1["execution_checks"].values())
        or not all(a1["performance_checks"].values())
        or a0["performance_checks"]
        .get("mean_loss_reduction_at_least_10_percent") is not False
        or any(
            not value
            for key, value in a0["performance_checks"].items()
            if key != "mean_loss_reduction_at_least_10_percent"
        )
        or a0["result"]["outcome"]["loss_reduction_fraction"]
        != 0.02638221303219166
        or a1["result"]["outcome"]["loss_reduction_fraction"]
        != 0.1966841709077128
    ):
        raise PhaseE6GateError(
            "Gate E.5 is not the frozen valid-negative parent result"
        )
    sample_ids = list(a0["result"]["sample_ids"])
    if sample_ids != list(a1["result"]["sample_ids"]):
        raise PhaseE6GateError("Gate E.5 3e-4 matched sample IDs changed")
    return {
        "artifact_sha256": artifact_sha256,
        "gate_e5_passed": False,
        "learning_rate_source": (
            "chosen after inspecting Gate E.5; A1@3e-4 had the largest "
            "held-out mean loss reduction among the tested E.5 A1 tracks"
        ),
        "post_selection": True,
        "root": str(PHASE_E5_ROOT),
        "sample_ids": sample_ids,
        "selected_by_e5_gate": False,
        "tested_learning_rate": PHASE_E6_LEARNING_RATE,
    }


def derive_e6_track_config(
    cfg: Thought3Config,
    *,
    variant: str,
) -> Thought3Config:
    if variant not in {"A0", "A1"}:
        raise PhaseE6GateError(f"unsupported Gate E.6 variant: {variant}")
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=f"thought3_phase_e6_{variant.lower()}_{PHASE_E6_LR_SLUG}",
            output_dir=cfg.experiment.output_dir / "tracks" / variant.lower(),
        ),
        sampler=replace(cfg.sampler, active_k=0 if variant == "A0" else 1),
        training=replace(cfg.training, learning_rate=PHASE_E6_LEARNING_RATE),
    )
    validate_config(derived)
    return derived


def replication_performance_checks(
    variant: str,
    result: Mapping[str, Any],
) -> dict[str, bool]:
    outcome = result["outcome"]
    common = {
        "at_least_6_of_8_samples_non_worsened": (
            int(outcome["non_worsened_sample_count"]) >= 6
        ),
        "max_delta_hidden_at_most_1_0": (
            float(outcome["max_objective_gated_delta_to_action_hidden_ratio"])
            <= 1.0
        ),
        "median_delta_hidden_at_most_0_5": (
            float(outcome["median_gated_delta_to_action_hidden_ratio"]) <= 0.5
        ),
        "no_sample_loss_above_2x_initial": (
            int(outcome["catastrophic_sample_count"]) == 0
        ),
    }
    if variant == "A1":
        common["a1_mean_loss_reduction_at_least_10_percent"] = (
            float(outcome["loss_reduction_fraction"]) >= 0.1
        )
    elif variant == "A0":
        common["a0_mean_loss_does_not_worsen"] = (
            float(outcome["loss_reduction_fraction"]) >= 0.0
        )
    else:
        raise PhaseE6GateError(f"unsupported performance variant: {variant}")
    return common


def paired_superiority_checks(
    a0: Mapping[str, Any],
    a1: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    a0_outcome = a0["outcome"]
    a1_outcome = a1["outcome"]
    a0_rows = {
        str(row["base_sample_id"]): float(row["final_action_loss"])
        for row in a0_outcome["per_sample"]
    }
    a1_rows = {
        str(row["base_sample_id"]): float(row["final_action_loss"])
        for row in a1_outcome["per_sample"]
    }
    a0_mean = float(a0_outcome["final_mean_action_loss"])
    a1_mean = float(a1_outcome["final_mean_action_loss"])
    relative = (a0_mean - a1_mean) / a0_mean if a0_mean > 0 else float("-inf")
    count = (
        sum(a1_rows[sample_id] <= a0_rows[sample_id] for sample_id in a0_rows)
        if a0_rows.keys() == a1_rows.keys()
        else 0
    )
    values = {
        "a0_final_mean_action_loss": a0_mean,
        "a1_final_mean_action_loss": a1_mean,
        "a1_non_higher_sample_count": count,
        "a1_relative_mean_improvement_over_a0": relative,
    }
    checks = {
        "a1_final_mean_at_least_10_percent_below_a0": relative >= 0.1,
        "a1_not_higher_than_a0_on_at_least_6_of_8_samples": count >= 6,
        "paired_sample_ids_exact": a0_rows.keys() == a1_rows.keys(),
    }
    return checks, values


def _run_phase_e6(cfg: Thought3Config, *, resume: bool) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import parameter_state_sha256

    _assert_phase_e6_scope(cfg)
    _require_phase_e6_confirmation()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE6GateError("Gate E.6 requires exactly one CUDA-visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE6GateError(
            "Gate E.6 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.cuda.set_device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase_e5 = verify_frozen_phase_e5()
    cohort = verify_frozen_fresh_cohort(
        cfg,
        e5_sample_ids=phase_e5["sample_ids"],
    )
    phase_d = _verify_phase_d_gate(cfg)
    tracks = {
        variant: derive_e6_track_config(cfg, variant=variant)
        for variant in ("A0", "A1")
    }
    if _matched_recipe_payload(tracks["A0"]) != _matched_recipe_payload(
        tracks["A1"]
    ):
        raise PhaseE6GateError("Gate E.6 A0/A1 recipes are not matched")
    _progress(
        "frozen_inputs_verified",
        cohort_sha256=PHASE_E6_FROZEN_COHORT_SHA256,
        identity_schedule_sha256=PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256,
        learning_rate=PHASE_E6_LEARNING_RATE,
        post_selection=True,
    )
    _progress("model_load_started", device="cuda:0")
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize("cuda:0")
    model_report["load_peak_mib"] = (
        int(torch.cuda.max_memory_allocated("cuda:0")) / 2**20
    )
    _progress("model_loaded", load_peak_mib=model_report["load_peak_mib"])
    prepared = prepare_real_training_data(
        cfg,
        model=model,
        upstream_cfg=upstream_cfg,
        device="cuda:0",
        progress=_progress,
        train_only_limit=8,
        train_only_offset=PHASE_E6_TRAIN_ONLY_OFFSET,
    )
    data_report = dict(prepared.report)
    atomic_write_json(output / "data_preparation.json", data_report)
    source = data_report["current_source"]
    prepared_ids = [sample.base_sample_id for sample in prepared.samples]
    if (
        source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or int(source["action_target_rows_read"]) != 256
        or int(source["current_camera_frames_decoded"]) != 16
        or int(source["state_rows_read"]) != 8
        or data_report["future_rgb_used_as_input"] is not False
        or data_report["split_counts"] != {"train": 8, "development": 0}
        or data_report["available_split_counts"]
        != {"train": 28, "development": 4}
        or data_report["selection_mode"] != "ordered_train_window"
        or int(data_report["train_only_offset"]) != PHASE_E6_TRAIN_ONLY_OFFSET
        or prepared_ids != cohort["sample_ids"]
    ):
        raise PhaseE6GateError("Gate E.6 data-access/cohort audit failed")

    frozen_before = parameter_state_sha256(iter(model.named_parameters()))
    results: dict[str, Mapping[str, Any]] = {}
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    try:
        for variant in ("A0", "A1"):
            _progress(
                "track_started",
                learning_rate=PHASE_E6_LEARNING_RATE,
                variant=variant,
            )
            track_result = run_full_cohort_objective_aggregation(
                tracks[variant],
                model=model,
                prepared=prepared,
                frozen_parameter_sha256=frozen_before,
                resume=resume,
                device="cuda:0",
                progress=_progress,
                protocol=PHASE_E6_PROTOCOL,
            )
            results[variant] = track_result
            _progress(
                "track_complete",
                loss_reduction_fraction=track_result["outcome"][
                    "loss_reduction_fraction"
                ],
                non_worsened=track_result["outcome"][
                    "non_worsened_sample_count"
                ],
                variant=variant,
            )
    except BaseException as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    frozen_after = parameter_state_sha256(iter(model.named_parameters()))
    prevalidation = {
        "captured_at": _utc_now(),
        "data_preparation": data_report,
        "execution_error": (
            None
            if execution_error is None
            else f"{type(execution_error).__name__}: {execution_error}"
        ),
        "execution_traceback": execution_traceback,
        "fresh_cohort": cohort,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e5_frozen": phase_e5,
        "schema_version": PHASE_E6_SCHEMA,
        "tracks": {key: dict(value) for key, value in results.items()},
    }
    atomic_write_json(output / "pre_validation_result.json", prevalidation)
    _progress("frozen_hash_after", sha256=frozen_after)
    if execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE6GateError(
            "Gate E.6 track execution failed after frozen hash capture"
        ) from execution_error

    execution_checks: dict[str, dict[str, bool]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    performance: dict[str, dict[str, bool]] = {}
    for variant in ("A0", "A1"):
        execution_checks[variant], artifacts[variant] = _track_checks(
            tracks[variant],
            results[variant],
            protocol=PHASE_E6_PROTOCOL,
            frozen_identity_schedule_sha256=(
                PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
            ),
            forbidden_training_slot_ranges=(
                (10_001, 10_200),
                (20_001, 21_600),
            ),
        )
        performance[variant] = replication_performance_checks(
            variant,
            results[variant],
        )
    a0 = results["A0"]
    a1 = results["A1"]
    paired_checks = {
        "same_sample_ids": a0["sample_ids"] == a1["sample_ids"],
        "same_initial_adapter": (
            a0["initial_adapter_sha256"] == a1["initial_adapter_sha256"]
        ),
        "same_initial_multiflow_probe": (
            _initial_probe_signature(a0) == _initial_probe_signature(a1)
        ),
        "same_identity_schedule": (
            artifacts["A0"]["identity_schedule_sha256"]
            == artifacts["A1"]["identity_schedule_sha256"]
        ),
        "same_observed_objective_schedule": (
            artifacts["A0"]["observed_schedule_sha256"]
            == artifacts["A1"]["observed_schedule_sha256"]
        ),
        "same_parameter_count": (
            a0["trainable_parameter_count"]
            == a1["trainable_parameter_count"]
        ),
        "same_training_budget": (
            a0["completed_steps"]
            == a1["completed_steps"]
            == OBJECTIVE_AGGREGATION_UPDATES
            and a0["completed_objectives"]
            == a1["completed_objectives"]
            == OBJECTIVE_AGGREGATION_UPDATES * OBJECTIVES_PER_UPDATE
        ),
    }
    superiority_checks, superiority_values = paired_superiority_checks(a0, a1)
    cross_checks = {
        "fresh_cohort_exact": list(a0["sample_ids"]) == cohort["sample_ids"],
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
        "phase_e5_artifacts_unchanged": all(
            sha256_file(PHASE_E5_ROOT / name) == expected
            for name, expected in PHASE_E5_FROZEN_ARTIFACTS.items()
        ),
        "slots_disjoint_from_all_prior_training_namespaces": (
            objective_aggregation_flow_slot(
                1, 1, flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET
            )
            == 31_001
            and objective_aggregation_flow_slot(
                200, 8, flow_slot_offset=PHASE_E6_FLOW_SLOT_OFFSET
            )
            == 32_600
        ),
    }
    gate_passed = (
        all(all(checks.values()) for checks in execution_checks.values())
        and all(all(checks.values()) for checks in performance.values())
        and all(paired_checks.values())
        and all(superiority_checks.values())
        and all(cross_checks.values())
    )
    result = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "cross_checks": cross_checks,
        "data_preparation": data_report,
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "flash_sdp": False,
            "math_sdp": True,
            "mem_efficient_sdp": False,
            "tf32": False,
        },
        "fresh_cohort": cohort,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e6_passed": gate_passed,
        "model_load": model_report,
        "paired_checks": paired_checks,
        "paired_superiority_checks": superiority_checks,
        "paired_superiority_values": superiority_values,
        "phase_d_frozen": phase_d,
        "phase_e5_frozen": phase_e5,
        "post_selection_disclosure": {
            "independent_confirmatory_test": False,
            "learning_rate": PHASE_E6_LEARNING_RATE,
            "learning_rate_chosen_after_e5": True,
            "learning_rate_selected_by_e5_frozen_gate": False,
            "reason": phase_e5["learning_rate_source"],
            "thresholds_chosen_after_e5": True,
        },
        "preregistered_gate": {
            "a0_min_loss_reduction_fraction": 0.0,
            "a1_min_loss_reduction_fraction": 0.1,
            "a1_vs_a0_min_relative_mean_improvement": 0.1,
            "a1_vs_a0_min_non_higher_samples": 6,
            "gradient_reduction": "arithmetic_mean",
            "heldout_flow_steps": list(
                OBJECTIVE_AGGREGATION_HELDOUT_FLOW_STEPS
            ),
            "identity_schedule_sha256": (
                PHASE_E6_FROZEN_IDENTITY_SCHEDULE_SHA256
            ),
            "learning_rate": PHASE_E6_LEARNING_RATE,
            "max_catastrophic_samples": 0,
            "max_median_delta_hidden_ratio": 0.5,
            "max_sample_delta_hidden_ratio": 1.0,
            "min_non_worsened_samples_per_variant": 6,
            "objectives_per_update": OBJECTIVES_PER_UPDATE,
            "optimizer_updates_per_track": OBJECTIVE_AGGREGATION_UPDATES,
            "sample_count": OBJECTIVES_PER_UPDATE,
            "training_flow_slot_end": 32_600,
            "training_flow_slot_start": 31_001,
            "training_flow_slot_offset": PHASE_E6_FLOW_SLOT_OFFSET,
            "variants": ["A0", "A1"],
        },
        "schema_version": PHASE_E6_SCHEMA,
        "scope": {
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "heldout_probe_objectives": 160,
            "learning_rate_count": 1,
            "matched_optimizer_update_budget": True,
            "ood_outcomes_read": False,
            "optimizer_updates": 400,
            "rollout_started": False,
            "sample_count": 8,
            "single_gpu": True,
            "success_outcomes_read": False,
            "task_count": 1,
            "track_count": 2,
            "training_objectives": 3200,
            "uses_ground_truth_future": False,
        },
        "status": "passed" if gate_passed else "failed",
        "tested_learning_rate": PHASE_E6_LEARNING_RATE,
        "tracks": {
            variant: {
                "artifacts": artifacts[variant],
                "execution_checks": execution_checks[variant],
                "performance_checks": performance[variant],
                "result": dict(results[variant]),
            }
            for variant in ("A0", "A1")
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e6_fresh_cohort_replication(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run E.6 while preserving every completed positive/negative Run ID."""

    _assert_phase_e6_scope(cfg)
    _require_phase_e6_confirmation()
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e6_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if resume and existing.get("gate_e6_passed") is True:
            return existing
        if resume:
            raise PhaseE6GateError(
                "existing Gate E.6 result failed; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.6 result exists; pass --resume: {result_path}"
        )
    if status_path.is_file() and not resume:
        raise PhaseE6GateError(
            "existing partial Gate E.6 requires --resume or a new Run ID"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E6_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e6(cfg, resume=resume)
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["gate_e6_passed"] is not True:
            raise PhaseE6GateError(
                "Gate E.6 hard checks failed; inspect gate_e6_result.json"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e6_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E6_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e6_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E6_SCHEMA,
            "status": "passed",
        },
    )
    return result
