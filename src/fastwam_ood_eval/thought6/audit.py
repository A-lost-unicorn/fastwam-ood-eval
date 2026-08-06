"""Read-only upstream audit and immutable Phase 6 preregistration materializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought6 import FUSION_MODES, SIGMA_THRESHOLD
from fastwam_ood_eval.thought6.checkpoint_resolver import resolve_adapter_checkpoint
from fastwam_ood_eval.thought6.config import Thought6Config
from fastwam_ood_eval.thought6.paired_noise import (
    build_noise_pairing_manifest,
    offline_noise_identity,
    rollout_noise_identity,
)
from fastwam_ood_eval.thought6.reporting import not_run_result, render_not_run_report
from fastwam_ood_eval.thought6.schemas import (
    build_artifact_manifest,
    file_sha256,
    object_sha256,
    seal_full_object,
    write_once_json,
    write_once_text,
    write_report_transition,
    write_stage_json,
)
from fastwam_ood_eval.thought6.sigma_gate import build_inference_sigma_schedule, schedule_manifest
from fastwam_ood_eval.thought6.task_selection import select_phase6_tasks


PLANNED_FILES = (
    "src/fastwam_ood_eval/thought6/__init__.py",
    "src/fastwam_ood_eval/thought6/schemas.py",
    "src/fastwam_ood_eval/thought6/config.py",
    "src/fastwam_ood_eval/thought6/audit.py",
    "src/fastwam_ood_eval/thought6/checkpoint_resolver.py",
    "src/fastwam_ood_eval/thought6/sigma_gate.py",
    "src/fastwam_ood_eval/thought6/future_modes.py",
    "src/fastwam_ood_eval/thought6/offline_utility.py",
    "src/fastwam_ood_eval/thought6/rollout_policy.py",
    "src/fastwam_ood_eval/thought6/paired_noise.py",
    "src/fastwam_ood_eval/thought6/task_selection.py",
    "src/fastwam_ood_eval/thought6/statistics.py",
    "src/fastwam_ood_eval/thought6/gate_decision.py",
    "src/fastwam_ood_eval/thought6/reporting.py",
    "src/fastwam_ood_eval/thought6/cli.py",
    "configs/thought6/phase6_audit.yaml",
    "configs/thought6/phase6a_smoke.yaml",
    "configs/thought6/phase6b_offline_utility.yaml",
    "configs/thought6/phase6c_rollout_stage1.yaml",
    "configs/thought6/phase6c_rollout_stage2.yaml",
    "scripts/run_thought6_audit.sh",
    "scripts/run_thought6_phase6a_smoke.sh",
    "scripts/run_thought6_phase6b_utility.sh",
    "scripts/run_thought6_phase6c_stage1.sh",
    "scripts/run_thought6_phase6c_stage2.sh",
)


def _source_audit() -> dict[str, Any]:
    sources = {
        "thought3_injection": Path("src/fastwam_ood_eval/thought3/injection.py"),
        "thought3_counterfactual": Path("src/fastwam_ood_eval/thought3/phase1_k1_online_counterfactual.py"),
        "thought3_phase2_result": Path("outputs/thought3/phase2_full_28_4_a0_a1_v1/phase2_training_result.json"),
        "thought5_pilot": Path("outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/run_status.json"),
        "thought5_failure_analysis": Path("outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4_readonly_failure_v1/analysis_result.json"),
        "scheduler": Path("third_party/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py"),
    }
    result: dict[str, Any] = {}
    for name, path in sources.items():
        result[name] = {
            "path": str(path),
            "available": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None,
        }
    return result


def _noise_manifest(cfg: Thought6Config, task_manifest: dict[str, Any]) -> dict[str, Any]:
    offline = []
    rollout = []
    missing: list[str] = []
    for task in task_manifest["selected_tasks"]:
        episodes = list(task.get("phase6b_episode_ids", []))
        if len(episodes) < cfg.utility_episodes_per_task:
            missing.append(task["canonical_id"])
        for episode in episodes:
            for flow_slot in range(cfg.utility_flow_slots):
                offline.append(
                    offline_noise_identity(
                        suite=task["suite"],
                        task_id=int(task["task_id"]),
                        episode_id=episode,
                        flow_slot=flow_slot,
                        seed=cfg.seed,
                    )
                )
        for state_index in range(cfg.stage2_total_states_per_task):
            rollout.append(
                rollout_noise_identity(
                    stage=1 if state_index < cfg.stage1_states_per_task else 2,
                    suite=task["suite"],
                    task_id=int(task["task_id"]),
                    initial_state_index=state_index,
                    seed=cfg.seed,
                )
            )
    payload = build_noise_pairing_manifest(offline, rollout)
    payload.update(
        {
            "status": "ready" if not missing else "blocked_missing_episode_provenance",
            "missing_task_episode_provenance": missing,
            "offline_flow_slots_per_episode": cfg.utility_flow_slots,
            "stage1_initial_states_per_task": cfg.stage1_states_per_task,
            "stage2_new_initial_state_range": [
                cfg.stage1_states_per_task,
                cfg.stage2_total_states_per_task - 1,
            ],
        }
    )
    payload["manifest_sha256"] = object_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    return payload


def collect_audit(cfg: Thought6Config) -> dict[str, Any]:
    adapter = resolve_adapter_checkpoint(cfg)
    tasks = select_phase6_tasks(cfg)
    schedule = schedule_manifest(build_inference_sigma_schedule())
    noise = _noise_manifest(cfg, tasks)
    missing_suites = [
        suite
        for suite, inventory in tasks["dataset_inventories"].items()
        if not inventory["available"]
    ]
    status = (
        "complete_ready_for_phase6a_and_phase6b"
        if not missing_suites
        else "complete_phase6a_ready_phase6b_blocked_missing_datasets"
    )
    return {
        "schema_version": "thought6.audit.v1",
        "status": status,
        "scientific_result": False,
        "config_fingerprint": cfg.fingerprint,
        "adapter": adapter,
        "task_selection": tasks,
        "noise_pairing": noise,
        "inference_schedule": schedule,
        "source_audit": _source_audit(),
        "missing_suite_datasets": missing_suites,
        "phase6a_ready": True,
        "phase6b_ready": not missing_suites,
        "phase6c_locked": True,
        "planned_files": list(PLANNED_FILES),
    }


def _frozen_protocol(cfg: Thought6Config, audit: dict[str, Any]) -> dict[str, Any]:
    return seal_full_object(
        {
            "schema_version": "thought6.frozen_protocol.v1",
            "status": "frozen",
            "config_fingerprint": cfg.fingerprint,
            "hypothesis": "future control utility is flow-noise dependent",
            "sigma_threshold": SIGMA_THRESHOLD,
            "sigma_threshold_configurable": False,
            "variants": list(FUSION_MODES),
            "primary_method": "Fsigma",
            "backbone": "original_frozen_FastWAM",
            "adapter": "Thought3_A1_K1_fixed_step_200_frozen",
            "training": False,
            "optimizer_allowed": False,
            "action_denoise_steps": 20,
            "online_gate_source": "actual_scheduler_sigma",
            "offline_gate_source": "sampled_training_style_effective_BF16_sigma",
            "phase6b_gates": [
                "clean_noninferiority",
                "camera_positive_utility",
                "correct_content_specificity",
                "timing_benefit",
                "no_artificial_degradation",
            ],
            "bootstrap": {"replicates": 10000, "seed": 6607},
            "stage2_rule": "positive_camera_direction_CI_inconclusive_plus_clean_noninferiority_plus_better_than_F0",
            "historical_facts": {
                "clean_success_percent": 97.25,
                "ood_success_percent": 47.70,
                "camera_ood_success_percent": 15.13,
                "thought3_A1_minus_A0_heldout_loss_percent": 3.624,
                "thought5_G3_stopped": True,
                "thought5_sigma_utility": {
                    "[0,0.25)": -0.049520,
                    "[0.25,0.50)": -0.040457,
                    "[0.50,0.75)": 0.002821,
                    "[0.75,1.00]": 0.001216,
                },
            },
            "audit_sha256": object_sha256(audit),
        }
    )


def _audit_markdown(audit: dict[str, Any]) -> str:
    adapter = audit["adapter"]
    tasks = audit["task_selection"]
    schedule = audit["inference_schedule"]["steps"]
    selected = ", ".join(row["canonical_id"] for row in tasks["selected_tasks"])
    excluded = json.dumps(tasks["historical_exclusions"]["excluded_tasks"], ensure_ascii=False)
    sigma_values = ", ".join(f"{row['effective_sigma']:.3f}" for row in schedule)
    missing = ", ".join(audit["missing_suite_datasets"]) or "none"
    return f"""# Phase 6 Audit

Status: `{audit['status']}`. This audit is not a scientific Phase 6 result.

1. Unique Adapter: `{adapter['checkpoint_path']}`; file SHA-256 `{adapter['adapter_file_sha256']}`; fixed A1/K=1/step 200, no loss/recency selection.
2. Injection point: output hook immediately after `model.action_expert.action_encoder`, before Action DiT blocks.
3. Offline sigma: CPU FP32 uniform -> `phi(u,5)` -> `t=sigma*1000` -> BF16 timestep -> effective `t_bf16/1000`.
4. Scheduler: `phi(u,s)=s*u/(1+(s-1)u)` in `third_party/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py`.
5. Online 20-step effective sigma: `{sigma_values}`; the 0.5 gate activates 17/20 steps.
6. formal-null: Thought3 `ActionEncoderFutureInjector.activate_null(expected_calls=20)` is a parameter-free identity.
7. B0 parity: same initial action noise/cache/scheduler plus identity hook; Phase 6A compares the full output bitwise to Thought3 formal-null.
8. Cache reuse: correct and shuffle K=1 latents are generated once per paired observation and reused across objective arms; null constructs no zero future.
9. Seeds: SHA-256 namespaces independently freeze action timestep/noise, future noise, environment, episode and initial state identities.
10. Historical use: Thought3/4/5 used `libero_goal/0`; episode exclusions are retained in the task manifest.
11. Unused tasks: eight canonical tasks exist: `{selected}`.
12. Selection: exclude historical task/episodes, sort canonical task ID ascending, take two; outcome, success, utility and difficulty are unread. Exclusions: `{excluded}`.
13. Three-card schedule: rank 0/1/2 receive selected task rows by stable task ordinal modulo 3; aggregation occurs only after all immutable shards complete.
14. Smoke estimate: one 24 GiB card, approximately 23.0–23.5 GiB peak and 20–35 minutes for two states and all technical arms; this is an engineering estimate, not measured Phase 6 latency.
15. Planned files: {', '.join(audit['planned_files'])}.

## Fail-closed readiness

Missing demonstrations: `{missing}`. Phase 6A may use the available unused `libero_goal` task. Phase 6B and all Phase 6C rollout launchers remain locked until all four suite roots and episode provenance are frozen in a fresh immutable protocol namespace.
"""


def run_audit(cfg: Thought6Config) -> dict[str, Any]:
    output = cfg.output_dir
    audit = collect_audit(cfg)
    output.mkdir(parents=True, exist_ok=True)
    write_once_json(output / "frozen_protocol.json", _frozen_protocol(cfg, audit))
    write_once_json(output / "adapter_checkpoint_manifest.json", audit["adapter"])
    write_once_json(output / "task_selection_manifest.json", audit["task_selection"])
    write_once_json(output / "noise_pairing_manifest.json", audit["noise_pairing"])
    write_once_text(output / "audit_report.md", _audit_markdown(audit))
    scaffolds = {
        "phase6a_smoke_results.json": not_run_result(
            "thought6.phase6a.smoke_result.v1",
            reason="real single-GPU smoke requires explicit confirmation",
            prerequisites=["CONFIRM_THOUGHT6_PHASE6A=YES", "idle_24GiB_GPU"],
        ),
        "phase6b_utility_results.json": not_run_result(
            "thought6.phase6b.utility_result.v1",
            reason="Phase 6A and four-suite demonstration readiness are required",
            prerequisites=["phase6a_passed", "four_suite_datasets_ready"],
        ),
        "phase6b_gate_decision.json": not_run_result(
            "thought6.phase6b_gate_decision.v1",
            reason="Phase 6B utility has not run",
            prerequisites=["phase6b_utility_complete"],
        ),
        "phase6c_stage1_results.json": not_run_result(
            "thought6.phase6c.stage1_result.v1",
            reason="Phase 6C is locked until all five Phase 6B gates pass",
            prerequisites=["phase6c_unlocked=true"],
        ),
        "phase6c_stage2_results.json": not_run_result(
            "thought6.phase6c.stage2_result.v1",
            reason="Stage 2 may only follow the preregistered inconclusive-positive rule",
            prerequisites=["stage2_unlocked=true", "separate_explicit_confirmation"],
        ),
        "mechanism_classification.json": not_run_result(
            "thought6.mechanism_classification.v1",
            reason="No Phase 6 scientific experiment has completed",
            prerequisites=["phase6b_complete", "rollout_or_stop_rule_resolved"],
        ),
    }
    for name, value in scaffolds.items():
        write_stage_json(output / name, value)
    blockers = [
        "Phase 6A real GPU smoke has not run.",
        *[f"demonstration root missing: {suite}" for suite in audit["missing_suite_datasets"]],
        "Phase 6B, Phase 6C Stage 1 and Stage 2 are NOT RUN.",
    ]
    write_report_transition(
        output / "report.md",
        render_not_run_report(audit_status=audit["status"], blockers=blockers),
    )
    integrity = seal_full_object(
        {
            "schema_version": "thought6.execution_integrity.v1",
            "status": "NOT RUN",
            "audit_status": audit["status"],
            "scientific_results_present": False,
            "config_fingerprint": cfg.fingerprint,
            "adapter_file_sha256": audit["adapter"]["adapter_file_sha256"],
            "backbone_checkpoint_sha256": cfg.backbone_checkpoint_sha256,
            "phase6a": "NOT RUN",
            "phase6b": "NOT RUN",
            "phase6c_stage1": "NOT RUN",
            "phase6c_stage2": "NOT RUN",
            "optimizer_created": False,
        }
    )
    write_stage_json(output / "execution_integrity.json", integrity)
    names = [
        "audit_report.md",
        "frozen_protocol.json",
        "adapter_checkpoint_manifest.json",
        "task_selection_manifest.json",
        "noise_pairing_manifest.json",
        *scaffolds,
        "execution_integrity.json",
        "report.md",
    ]
    manifest = build_artifact_manifest(output, names=names, status="NOT RUN")
    write_stage_json(output / "artifact_manifest.json", manifest)
    return {
        "status": audit["status"],
        "scientific_result": False,
        "phase6a_ready": audit["phase6a_ready"],
        "phase6b_ready": audit["phase6b_ready"],
        "missing_suite_datasets": audit["missing_suite_datasets"],
        "output": str(output),
    }
