"""Gate E.3: held-out multi-flow evaluation of frozen Gate E.2 tracks.

This diagnostic never trains.  It loads the six completed Gate E.2 step-200
Adapter-only checkpoints and evaluates five deterministic action-flow draws
that were not used by the Gate E.2 optimizer.  The goal is to separate
per-sample stability from a single fixed noise/timestep draw without reading
development, OOD, success, or rollout outcomes.
"""

from __future__ import annotations

import gc
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fastwam_ood_eval.thought3.checkpointing import (
    adapter_state_sha256,
    load_adapter_checkpoint,
)
from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    load_thought3_config,
)
from fastwam_ood_eval.thought3.injection import (
    ActionEncoderFutureInjector,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    sha256_file,
)
from fastwam_ood_eval.thought3.phase_c_smoke import (
    _load_upstream_model,
)
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    PHASE_E2_LR_GRID,
    _assert_phase_e2_scope,
    _derive_tracks,
    performance_checks,
    select_smallest_eligible_lr,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    _checkpoint_expected,
    build_real_adapter,
    evaluate_multiflow_subset_probe,
    multiflow_subset_outcome,
    prepare_real_training_data,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_thought3_output_path,
)


PHASE_E3_SCHEMA = "thought3.phase_e3.multiflow.v2"
PHASE_E3_FLOW_STEPS = (1, 2, 3, 4, 5)
PHASE_E2_CONFIG = Path(
    "configs/thought3/phase_e2_eight_sample_diagnostic.yaml"
)
PHASE_E2_ROOT = Path(
    "outputs/thought3/phase_e2_eight_sample_v1"
)
PHASE_E2_CONFIG_FINGERPRINT = (
    "f1a4cb39a2c6866331543a55c00f6d592be7a609c788802827f1848e244d6fd3"
)
PHASE_E2_FROZEN_ARTIFACTS = {
    "gate_e2_result.json": (
        "40f66bc50acd8e175ecb61ec150a04ef9ed5c55bf1fa9090802cc529104214bb"
    ),
    "run_status.json": (
        "570774031d338ee27754f460c46deaf2a12f77d39e1b68cd3b08cb6af1a91e58"
    ),
    "pre_validation_result.json": (
        "7aa98cfb95fbc73ab409ef47545e8a912ae221586fe57f2afa841676c6a9a7bb"
    ),
    "data_preparation.json": (
        "fb92b8c7f01129689c5a4ddd7ab96aaa184687dcec15b07b9f180d049dc01b4e"
    ),
}
PHASE_E2_SAMPLE_PAYLOAD_SHA256 = (
    "1bb4cfb6f4fc357f6227d7c369ad5fc00ed621b530270cd16cec9e1eba56973e"
)


class PhaseE3GateError(RuntimeError):
    """Raised when Gate E.3 violates its frozen diagnostic protocol."""


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
                "phase": "E.3",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e3_scope(cfg: Thought3Config) -> None:
    """Reject any expansion beyond the preregistered held-out probe."""

    _assert_phase_e2_scope(cfg)
    if (
        cfg.experiment.name
        != "thought3_phase_e3_multiflow_diagnostic_v2"
    ):
        raise PhaseE3GateError("Gate E.3 experiment name changed")
    if cfg.experiment.output_dir != Path(
        "outputs/thought3/phase_e3_multiflow_v2"
    ):
        raise PhaseE3GateError("Gate E.3 output directory changed")


def verify_frozen_phase_e2() -> dict[str, Any]:
    """Validate the exact failed Gate E.2 evidence before model loading."""

    artifact_sha256: dict[str, str] = {}
    for name, expected_sha in PHASE_E2_FROZEN_ARTIFACTS.items():
        path = PHASE_E2_ROOT / name
        if not path.is_file():
            raise PhaseE3GateError(
                f"frozen Gate E.2 artifact is missing: {path}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise PhaseE3GateError(
                f"frozen Gate E.2 artifact changed: {path}"
            )
        artifact_sha256[str(path)] = actual_sha

    result = load_json(PHASE_E2_ROOT / "gate_e2_result.json")
    status = load_json(PHASE_E2_ROOT / "run_status.json")
    if (
        result.get("schema_version")
        != "thought3.phase_e2.eight_sample.v1"
        or result.get("config_fingerprint")
        != PHASE_E2_CONFIG_FINGERPRINT
        or result.get("gate_e2_passed") is not False
        or result.get("status") != "failed"
        or result.get("selected_lr_slug") is not None
        or status.get("gate_e2_passed") is not False
        or status.get("status") != "failed"
    ):
        raise PhaseE3GateError(
            "Gate E.2 root result is not the frozen failed diagnostic"
        )
    if not all(bool(value) for value in result["cross_track_checks"].values()):
        raise PhaseE3GateError("Gate E.2 cross-track checks were not all true")
    if any(bool(value) for value in result["eligibility"].values()):
        raise PhaseE3GateError(
            "Gate E.2 unexpectedly contains an eligible learning rate"
        )

    track_evidence: dict[str, dict[str, Any]] = {}
    for lr_slug, _ in PHASE_E2_LR_GRID:
        track_evidence[lr_slug] = {}
        if not all(
            bool(value)
            for value in result["paired_checks"][lr_slug].values()
        ):
            raise PhaseE3GateError(
                f"Gate E.2 pairing checks failed at {lr_slug}"
            )
        for variant, directory_name in (("A0", "a0"), ("A1", "a1")):
            track = result["tracks"][lr_slug][variant]
            track_result = track["result"]
            if (
                track_result.get("status") != "complete"
                or int(track_result.get("completed_steps", -1)) != 200
                or not all(
                    bool(value)
                    for value in track["execution_checks"].values()
                )
            ):
                raise PhaseE3GateError(
                    f"Gate E.2 track is incomplete: {lr_slug}/{variant}"
                )
            checkpoint = (
                PHASE_E2_ROOT
                / "tracks"
                / lr_slug
                / directory_name
                / "checkpoints"
                / "step_00000200"
            )
            if (
                Path(str(track_result["checkpoint"])).resolve()
                != checkpoint.resolve()
            ):
                raise PhaseE3GateError(
                    f"Gate E.2 checkpoint path changed: {lr_slug}/{variant}"
                )
            track_evidence[lr_slug][variant] = {
                "checkpoint": str(checkpoint),
                "checkpoint_adapter_sha256": sha256_file(
                    checkpoint / "adapter.safetensors"
                ),
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint / "manifest.json"
                ),
                "expected_adapter_state_sha256": track_result[
                    "checkpoint_roundtrip"
                ]["adapter_state_sha256"],
                "training_manifest_sha256": sha256_file(
                    PHASE_E2_ROOT
                    / "tracks"
                    / lr_slug
                    / directory_name
                    / "training_manifest.json"
                ),
            }
    return {
        "artifact_sha256": artifact_sha256,
        "config_fingerprint": PHASE_E2_CONFIG_FINGERPRINT,
        "eligibility": dict(result["eligibility"]),
        "gate_e2_passed": False,
        "root": str(PHASE_E2_ROOT),
        "tracks": track_evidence,
    }


def _initial_action_signature(
    probe: Mapping[str, Any],
) -> tuple[tuple[str, int, float, float, float, float], ...]:
    return tuple(
        (
            str(row["base_sample_id"]),
            int(row["flow_step"]),
            float(row["timestep"]),
            float(row["action_weight"]),
            float(row["action_loss"]),
            float(row["gated_delta_norm"]),
        )
        for row in probe["per_objective"]
    )


def _probe_checks(probe: Mapping[str, Any]) -> dict[str, bool]:
    rows = list(probe["per_objective"])
    return {
        "complete_heldout_grid": (
            list(probe["flow_steps"]) == list(PHASE_E3_FLOW_STEPS)
            and int(probe["flow_objective_count"]) == 40
            and int(probe["sample_count"]) == 8
            and len(rows) == 40
            and {
                (
                    str(row["base_sample_id"]),
                    int(row["flow_step"]),
                )
                for row in rows
            }
            == {
                (str(base_sample_id), flow_step)
                for base_sample_id in probe["sample_ids"]
                for flow_step in PHASE_E3_FLOW_STEPS
            }
        ),
        "finite_probe": all(
            all(
                math.isfinite(float(row[field]))
                for field in (
                    "action_hidden_norm",
                    "action_loss",
                    "action_weight",
                    "attention_residual_norm",
                    "gated_delta_nonzero_fraction",
                    "gated_delta_norm",
                    "gated_delta_to_action_hidden_ratio",
                    "latency_ms",
                    "peak_memory_mib",
                    "timestep",
                )
            )
            for row in rows
        ),
        "memory_below_43_gib": (
            float(probe["max_objective_peak_memory_mib"]) < 43 * 1024
        ),
        "no_ground_truth_future": (
            probe.get("uses_ground_truth_future_input") is False
        ),
        "zero_weight_count_consistent": (
            int(probe["zero_weight_objective_count"])
            == sum(
                float(row["action_weight"]) == 0
                for row in rows
            )
        ),
        "zero_weight_loss_exact": all(
            float(row["action_weight"]) != 0
            or float(row["action_loss"]) == 0
            for row in rows
        ),
    }


def _run_phase_e3(cfg: Thought3Config) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e3_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E3_V2") != "YES":
        raise PhaseE3GateError(
            "set CONFIRM_THOUGHT3_PHASE_E3_V2=YES for real held-out probes"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE3GateError(
            "Gate E.3 requires exactly one CUDA-visible GPU"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE3GateError(
            "Gate E.3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
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
    e2_evidence = verify_frozen_phase_e2()
    phase_d = _verify_phase_d_gate(cfg)
    original_e2_cfg = load_thought3_config(PHASE_E2_CONFIG)
    _assert_phase_e2_scope(original_e2_cfg)
    if original_e2_cfg.fingerprint != PHASE_E2_CONFIG_FINGERPRINT:
        raise PhaseE3GateError("Gate E.2 config fingerprint changed")
    tracks = _derive_tracks(original_e2_cfg)
    _progress(
        "frozen_inputs_verified",
        gate_e2_sha256=PHASE_E2_FROZEN_ARTIFACTS[
            "gate_e2_result.json"
        ],
    )

    _progress("model_load_started", device="cuda:0")
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize("cuda:0")
    model_report["load_peak_mib"] = (
        int(torch.cuda.max_memory_allocated("cuda:0")) / 2**20
    )
    _progress(
        "model_loaded",
        load_peak_mib=model_report["load_peak_mib"],
    )
    prepared = prepare_real_training_data(
        cfg,
        model=model,
        upstream_cfg=upstream_cfg,
        device="cuda:0",
        progress=_progress,
        train_only_limit=8,
    )
    data_report = dict(prepared.report)
    atomic_write_json(output / "data_preparation.json", data_report)
    source = data_report["current_source"]
    if (
        source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or int(source["action_target_rows_read"]) != 256
        or int(source["current_camera_frames_decoded"]) != 16
        or int(source["state_rows_read"]) != 8
        or data_report["future_rgb_used_as_input"] is not False
        or data_report["split_counts"]
        != {"train": 8, "development": 0}
        or data_report["available_split_counts"]
        != {"train": 28, "development": 4}
        or data_report["selection_mode"] != "ordered_train_only"
        or data_report["sample_payload_sha256"]
        != PHASE_E2_SAMPLE_PAYLOAD_SHA256
    ):
        raise PhaseE3GateError("Gate E.3 data-access audit failed")

    e2_sample_ids = list(
        load_json(
            PHASE_E2_ROOT
            / "tracks"
            / "lr_1e_04"
            / "a0"
            / "training_manifest.json"
        )["sample_ids"]
    )
    samples_by_id = {
        sample.base_sample_id: sample for sample in prepared.samples
    }
    if (
        len(e2_sample_ids) != 8
        or len(set(e2_sample_ids)) != 8
        or set(samples_by_id) != set(e2_sample_ids)
    ):
        raise PhaseE3GateError("Gate E.3 sample identities changed")
    samples = tuple(
        samples_by_id[base_sample_id]
        for base_sample_id in e2_sample_ids
    )
    _progress("probe_data_ready", samples=len(samples))

    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    initial_probes: dict[str, Mapping[str, Any]] = {}
    initial_adapter_sha256: dict[str, str] = {}
    final_results: dict[str, dict[str, dict[str, Any]]] = {}
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    try:
        for variant in ("A0", "A1"):
            track_cfg = tracks["lr_1e_04"][variant]
            adapter = build_real_adapter(track_cfg, device="cuda:0")
            initial_adapter_sha256[variant] = adapter_state_sha256(
                adapter.state_dict()
            )
            injector = ActionEncoderFutureInjector(
                model.action_expert.action_encoder,
                adapter,
            )
            try:
                probe = evaluate_multiflow_subset_probe(
                    track_cfg,
                    model,
                    adapter,
                    injector,
                    samples,
                    flow_steps=PHASE_E3_FLOW_STEPS,
                    device="cuda:0",
                )
            finally:
                injector.close()
                del adapter
                torch.cuda.empty_cache()
            initial_probes[variant] = probe
            _progress(
                "initial_probe_complete",
                mean_action_loss=probe["mean_action_loss"],
                variant=variant,
            )

        for lr_slug, learning_rate in PHASE_E2_LR_GRID:
            final_results[lr_slug] = {}
            for variant, directory_name in (("A0", "a0"), ("A1", "a1")):
                track_cfg = tracks[lr_slug][variant]
                checkpoint = (
                    PHASE_E2_ROOT
                    / "tracks"
                    / lr_slug
                    / directory_name
                    / "checkpoints"
                    / "step_00000200"
                )
                adapter = build_real_adapter(
                    track_cfg,
                    device="cuda:0",
                )
                if (
                    adapter_state_sha256(adapter.state_dict())
                    != initial_adapter_sha256[variant]
                ):
                    raise PhaseE3GateError(
                        "Gate E.3 initial Adapter state drifted"
                    )
                manifest = load_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    expected=_checkpoint_expected(
                        track_cfg,
                        prepared,
                        frozen_parameter_sha256=frozen_before,
                    ),
                )
                if (
                    int(manifest.global_step) != 200
                    or manifest.extra.get("gate_e2_eight_sample")
                    is not True
                    or manifest.extra.get("adapter_state_sha256")
                    != e2_evidence["tracks"][lr_slug][variant][
                        "expected_adapter_state_sha256"
                    ]
                ):
                    raise PhaseE3GateError(
                        "Gate E.2 checkpoint semantic provenance changed"
                    )
                injector = ActionEncoderFutureInjector(
                    model.action_expert.action_encoder,
                    adapter,
                )
                try:
                    final_probe = evaluate_multiflow_subset_probe(
                        track_cfg,
                        model,
                        adapter,
                        injector,
                        samples,
                        flow_steps=PHASE_E3_FLOW_STEPS,
                        device="cuda:0",
                    )
                finally:
                    injector.close()
                    del adapter
                    torch.cuda.empty_cache()
                outcome = multiflow_subset_outcome(
                    initial_probes[variant],
                    final_probe,
                )
                final_results[lr_slug][variant] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_adapter_state_sha256": (
                        manifest.extra["adapter_state_sha256"]
                    ),
                    "final_probe": final_probe,
                    "initial_probe": initial_probes[variant],
                    "learning_rate": learning_rate,
                    "outcome": outcome,
                    "performance_checks": performance_checks(
                        {"outcome": outcome}
                    ),
                    "probe_checks": _probe_checks(final_probe),
                    "variant": variant,
                }
                _progress(
                    "checkpoint_probe_complete",
                    learning_rate=learning_rate,
                    loss_reduction_fraction=outcome[
                        "loss_reduction_fraction"
                    ],
                    lr_slug=lr_slug,
                    non_worsened=outcome[
                        "non_worsened_sample_count"
                    ],
                    variant=variant,
                )
    except BaseException as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    frozen_after = parameter_state_sha256(
        iter(model.named_parameters())
    )
    prevalidation = {
        "captured_at": _utc_now(),
        "data_preparation": data_report,
        "execution_error": (
            None
            if execution_error is None
            else f"{type(execution_error).__name__}: {execution_error}"
        ),
        "execution_traceback": execution_traceback,
        "final_results": final_results,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "initial_adapter_sha256": initial_adapter_sha256,
        "initial_probes": initial_probes,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "phase_e2_frozen": e2_evidence,
        "schema_version": PHASE_E3_SCHEMA,
    }
    atomic_write_json(
        output / "pre_validation_result.json",
        prevalidation,
    )
    _progress("frozen_hash_after", sha256=frozen_after)
    if execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE3GateError(
            "Gate E.3 probe execution failed after frozen hash capture"
        ) from execution_error

    initial_checks = {
        "a0_a1_initial_action_exact": (
            _initial_action_signature(initial_probes["A0"])
            == _initial_action_signature(initial_probes["A1"])
        ),
        "initial_adapter_sha_equal": (
            initial_adapter_sha256["A0"]
            == initial_adapter_sha256["A1"]
        ),
        "initial_a0_probe_valid": all(
            _probe_checks(initial_probes["A0"]).values()
        ),
        "initial_a1_probe_valid": all(
            _probe_checks(initial_probes["A1"]).values()
        ),
        "initial_zero_gate_exact": all(
            float(row["gated_delta_norm"]) == 0
            for variant in ("A0", "A1")
            for row in initial_probes[variant]["per_objective"]
        ),
    }
    eligibility: dict[str, bool] = {}
    paired_checks: dict[str, dict[str, bool]] = {}
    for lr_slug, _ in PHASE_E2_LR_GRID:
        a0 = final_results[lr_slug]["A0"]
        a1 = final_results[lr_slug]["A1"]
        paired_checks[lr_slug] = {
            "same_flow_steps": (
                a0["final_probe"]["flow_steps"]
                == a1["final_probe"]["flow_steps"]
                == list(PHASE_E3_FLOW_STEPS)
            ),
            "same_sample_ids": (
                a0["final_probe"]["sample_ids"]
                == a1["final_probe"]["sample_ids"]
            ),
            "same_probe_budget": (
                a0["final_probe"]["flow_objective_count"]
                == a1["final_probe"]["flow_objective_count"]
                == 40
            ),
        }
        eligibility[lr_slug] = (
            all(a0["probe_checks"].values())
            and all(a1["probe_checks"].values())
            and all(a0["performance_checks"].values())
            and all(a1["performance_checks"].values())
            and all(paired_checks[lr_slug].values())
        )
    selected_slug = select_smallest_eligible_lr(eligibility)
    learning_rates = dict(PHASE_E2_LR_GRID)
    all_final_probe_checks_passed = all(
        all(final_results[lr_slug][variant]["probe_checks"].values())
        for lr_slug, _ in PHASE_E2_LR_GRID
        for variant in ("A0", "A1")
    )
    cross_checks = {
        "all_initial_checks_passed": all(initial_checks.values()),
        "all_final_probe_checks_passed": (
            all_final_probe_checks_passed
        ),
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
        "frozen_fastwam_has_no_grad": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "frozen_fastwam_not_trainable": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "no_optimizer_created": True,
        "no_backward_called": True,
        "phase_e2_artifacts_unchanged": all(
            sha256_file(PHASE_E2_ROOT / name) == expected_sha
            for name, expected_sha in PHASE_E2_FROZEN_ARTIFACTS.items()
        ),
    }
    gate_passed = (
        all(cross_checks.values())
        and all(
            all(checks.values())
            for checks in paired_checks.values()
        )
        and selected_slug is not None
    )
    result = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "cross_checks": cross_checks,
        "data_preparation": data_report,
        "eligibility": eligibility,
        "flow_steps": list(PHASE_E3_FLOW_STEPS),
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e3_passed": gate_passed,
        "initial_adapter_sha256": initial_adapter_sha256,
        "initial_checks": initial_checks,
        "initial_probes": initial_probes,
        "model_load": model_report,
        "paired_checks": paired_checks,
        "phase_d_frozen": phase_d,
        "phase_e2_frozen": e2_evidence,
        "preregistered_gate": {
            "catastrophic_loss_ratio": 2.0,
            "flow_steps": list(PHASE_E3_FLOW_STEPS),
            "learning_rates": [
                value for _, value in PHASE_E2_LR_GRID
            ],
            "max_catastrophic_samples": 0,
            "max_median_delta_hidden_ratio": 0.5,
            "max_sample_delta_hidden_ratio": 1.0,
            "min_loss_reduction_fraction": 0.1,
            "min_non_worsened_samples": 6,
            "sample_count": 8,
            "selection_rule": (
                "smallest learning rate eligible for both A0 and A1"
            ),
            "zero_initial_objective_policy": (
                "exclude only from non-gating per-objective loss-ratio "
                "telemetry; retain in sample-equal official weighted loss"
            ),
        },
        "schema_version": PHASE_E3_SCHEMA,
        "scope": {
            "backward_calls": 0,
            "checkpoint_count": 6,
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "heldout_flow_objectives": 320,
            "ood_outcomes_read": False,
            "optimizer_steps": 0,
            "rollout_started": False,
            "success_outcomes_read": False,
            "uses_ground_truth_future": False,
        },
        "selected_learning_rate": (
            learning_rates[selected_slug]
            if selected_slug is not None
            else None
        ),
        "selected_lr_slug": selected_slug,
        "status": "passed" if gate_passed else "failed",
        "tracks": final_results,
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e3_multiflow(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate E.3 and preserve both passing and failed outcomes."""

    _assert_phase_e3_scope(cfg)
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e3_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        if resume:
            existing = load_json(result_path)
            if existing.get("gate_e3_passed") is True:
                return existing
            raise PhaseE3GateError(
                "existing Gate E.3 result failed; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.3 result exists; pass --resume: {result_path}"
        )
    if status_path.is_file() or (
        output / "pre_validation_result.json"
    ).is_file():
        raise PhaseE3GateError(
            "existing incomplete/failed Gate E.3 evidence must be "
            "preserved under its Run ID"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E3_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e3(cfg)
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["gate_e3_passed"] is not True:
            raise PhaseE3GateError(
                "Gate E.3 hard checks failed; inspect gate_e3_result.json"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e3_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E3_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e3_passed": True,
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E3_SCHEMA,
            "status": "passed",
        },
    )
    return result
