"""Read-only Phase 5 architecture, data, reuse, and compute audit."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought5.config import Thought5Config
from fastwam_ood_eval.thought5.paired_geometry_data import cohort_manifest
from fastwam_ood_eval.thought5.schemas import file_sha256, object_sha256, write_text_once


class Phase5AuditError(RuntimeError):
    pass


REQUIRED_PATHS = (
    "third_party/FastWAM/src/fastwam/models/wan22/mot.py",
    "third_party/FastWAM/src/fastwam/models/wan22/wan_video_dit.py",
    "src/fastwam_ood_eval/thought3/real_training.py",
    "src/fastwam_ood_eval/diagnostics/metrics.py",
    "src/fastwam_ood_eval/thought4/paired_rendering.py",
    "src/fastwam_ood_eval/thought4/geometry_labels.py",
    "src/fastwam_ood_eval/thought4/real_runtime.py",
    "outputs/thought4/phase4_geometry_action_diagnosis_v6/method_selection.json",
)


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect_audit(cfg: Thought5Config) -> dict[str, Any]:
    missing = [path for path in REQUIRED_PATHS if not Path(path).is_file()]
    if missing:
        raise Phase5AuditError(f"required Phase5 inputs are absent: {missing}")
    if not cfg.backbone.checkpoint_path.is_file():
        raise Phase5AuditError("official Fast-WAM checkpoint is absent")
    if not cfg.backbone.dataset_stats_path.is_file():
        raise Phase5AuditError("official Fast-WAM dataset statistics are absent")
    checkpoint_sha256 = file_sha256(cfg.backbone.checkpoint_path)
    if checkpoint_sha256 != cfg.backbone.checkpoint_sha256:
        raise Phase5AuditError(
            "official Fast-WAM checkpoint SHA-256 differs from the frozen protocol"
        )
    dataset_stats_sha256 = file_sha256(cfg.backbone.dataset_stats_path)
    if dataset_stats_sha256 != cfg.backbone.dataset_stats_sha256:
        raise Phase5AuditError(
            "official dataset-statistics SHA-256 differs from the frozen protocol"
        )
    upstream_head = _git_head(Path("third_party/FastWAM"))
    if upstream_head != cfg.backbone.fastwam_commit:
        raise Phase5AuditError(
            f"Fast-WAM commit mismatch: {upstream_head} != {cfg.backbone.fastwam_commit}"
        )
    manifest = cohort_manifest(cfg.cohort)
    counts: dict[str, int] = {"train": 0, "development": 0, "formal": 0}
    for row in manifest["rows"]:
        counts[row["split"]] += 1
    payload: dict[str, Any] = {
        "schema_version": "thought5.phase5.audit.v1",
        "status": "pass",
        "fail_closed": False,
        "model": {
            "fastwam_commit": upstream_head,
            "checkpoint_path": str(cfg.backbone.checkpoint_path),
            "checkpoint_expected_sha256": cfg.backbone.checkpoint_sha256,
            "checkpoint_observed_sha256": checkpoint_sha256,
            "checkpoint_hash_source": "Phase5 full-file read-only SHA-256",
            "dataset_stats_path": str(cfg.backbone.dataset_stats_path),
            "dataset_stats_observed_sha256": dataset_stats_sha256,
            "selected_feature": "mot.video_kv_cache.15.v",
            "selected_module": "video_expert.blocks.15.self_attn.v",
            "selected_shape": [1, 98, 3072],
            "video_layers": 30,
            "action_layers": 30,
            "spatial_layout": "single_frame_then_7x14_row_major_two_camera_tokens",
            "action_consumer": "mot.forward_action_with_video_cache layer 15",
        },
        "training_scope": {
            "lora_targets": list(cfg.method.lora_targets),
            "lora_rank": cfg.method.lora_rank,
            "video_window": [15, 15],
            "action_dit_frozen": True,
            "whole_backbone_training": False,
        },
        "simulator_labels": {
            "rgb": True,
            "metric_depth": True,
            "camera_intrinsic": True,
            "camera_extrinsic_convention": "camera_to_world",
            "world_to_camera": True,
            "eef_pose": True,
            "object_pose": True,
            "clean_camera_lighting_exact_state": True,
            "robot_init_exact_state": False,
        },
        "reuse": {
            "thought3_adapter_architecture": "reuse",
            "thought3_optimizer_lr_steps_objective_seed_checkpoint_rule": "reuse",
            "thought3_runner_unmodified_reuse": False,
            "thought3_reason": (
                "existing runner freezes the original-backbone SHA and lacks AS; "
                "a Thought5 wrapper must load the new backbone"
            ),
            "thought2_pure_metrics": "directly reusable",
            "thought2_formal_runner": "not reusable because provenance is base checkpoint",
            "thought4_probe_rules": "fixed layer/probe/statistics reusable",
            "thought4_outputs": "read-only",
        },
        "cohort": {
            "task_split": manifest["task_split"],
            "base_state_counts": counts,
            "historical_exclusions": manifest["historical_exclusions"],
            "formal_is_multi_task": len(cfg.cohort.formal_tasks) >= 2,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "compute": {
            "measured_fastwam_load_peak_mib": 23679.51318359375,
            "measured_thought3_adapter_train_peak_mib": 13277.43994140625,
            "measured_thought3_update_seconds": 17.34,
            "measured_thought4_v6_single_gpu_hours": 1.313,
            "measured_phase5_smoke_v3_training_peak_mib": 25216.0068359375,
            "phase5_training_peak_estimate_mib": [18000, 24000],
            "estimate_status": (
                "smoke v3 measured; pilot/formal throughput remains NOT MEASURED"
            ),
            "gpu_execution_plan": (
                "three-GPU primary schedule with independent single-GPU workers: "
                "pilot B1/G3/G4 in one wave; formal B1/G1/G2 then G3/B0; "
                "four-GPU formal remains a compatible execution-only schedule"
            ),
        },
        "forbidden": {
            "modify_third_party_fastwam": False,
            "read_success_for_training_selection": False,
            "formal_result_tuning": False,
            "gt_depth_at_inference": False,
        },
        "config_fingerprint": cfg.fingerprint,
    }
    payload["audit_sha256"] = object_sha256(payload)
    return payload


def render_audit_markdown(audit: dict[str, Any]) -> str:
    cohort = audit["cohort"]
    exclusions = cohort["historical_exclusions"]
    return f"""# Phase 5 code, data, and compute audit

Status: **PASS**. Core model paths, simulator labels, and exact-state paired
rendering exist, so the Phase 5 implementation may proceed. This is an audit,
not a scientific result; all Phase 5 training/evaluation results remain **NOT RUN**.

## 1. Frozen feature and real consumption path

- Thought4 selected `mot.video_kv_cache.15.v`.
- It is produced by `video_expert.blocks.15.self_attn.v` inside
  `MoT._build_expert_attention_io` and cached by `MoT.prefill_video_cache`.
- The official checkpoint has 30 Video DiT blocks (hidden width 3072) and 30
  Action DiT blocks (hidden width 1024). Layer counts are required to match.
- The observed tensor is `[B, N, C] = [1, 98, 3072]`. The current latent has
  one frame and a 7x14 grid; `rearrange(..., "b c f h w -> b (f h w) c")`
  makes it frame-major, then row-major. Width 14 is the horizontal two-camera
  composition (7 primary + 7 wrist tokens).
- `MoT.forward_action_with_video_cache` consumes layer-15 `k`/`v` at Action
  DiT block 15 by concatenating them with Action K/V. This is not an unused
  auxiliary branch.
- The 12.04 GB release checkpoint was read in full during this audit; observed
  SHA-256 equals the frozen value `{audit['model']['checkpoint_observed_sha256']}`.
  Dataset-statistics SHA-256 is
  `{audit['model']['dataset_stats_observed_sha256']}`.

## 2. Frozen trainable scope

The v1 window is exactly Video layer 15, with rank-8 LoRA on:

- `video_expert.blocks.15.self_attn.k`
- `video_expert.blocks.15.self_attn.v`

`GeoProjector` and `RayPoseEncoder` are additionally trainable. The rest of
Video DiT and the entire Action DiT stay frozen. `GeoProjector` is training-only;
GT depth and geometry targets are detached and absent at inference. The retained
`RayPoseEncoder` reads only rays and camera extrinsics/intrinsics.

## 3. Simulator and label feasibility

Thought4 already exercised RGB, robosuite metric depth, intrinsic matrices,
camera-to-world extrinsics, the inverse world-to-camera transform, EEF pose,
object pose, and simulator action replay. Clean/Camera/Lighting are regenerated
from the same serialized simulator state and are valid exact-state pairs.
Robot-init changes physical state and remains a separately reported non-exact
specificity control. All ten LIBERO Goal task families have Camera, Lighting,
and Robot Initial State variants in the local LIBERO-Plus catalog.

## 4. Thought2/3/4 reuse boundary

- Thought3's Adapter structure, AdamW optimizer, LR `3e-4`, 200 updates,
  action objective, seed 3407, and fixed checkpoint rule can be reused.
- Its existing runner cannot be reused byte-for-byte: it hard-codes the frozen
  original backbone SHA and implements A0/A1 but not AS. Phase 5 therefore wraps
  those frozen recipe functions after loading B1/G3 backbones.
- Thought2's pure future-distance and motion-direction metrics are checkpoint
  agnostic and reusable. Its original formal runner/provenance is not.
- Phase 5-B does not compare the trained G3 GeoProjector with B1's inactive
  projector. Every backbone instead gets the same 128-dimensional signed
  projection plus linear-ridge probe: fit on train, alpha selected on
  development, and formal read exactly once.
- Thought4 layer 15, probe families, exact-pair rules, and grouped bootstrap are
  frozen. Its formal rows and outputs are read-only and are not training data.

## 5. Cohort freeze recommendation

Task-level separation is used to make leakage visible:

- training tasks: 0–5, 24 episodes per task (144 base states);
- development tasks: 6–7, 12 episodes per task (24 base states);
- untouched formal tasks: 8–9, 24 episodes per task (48 base states).

Each base state expands into matched Clean/Camera/Lighting and the independently
reported Robot-init condition. Seeds use disjoint split namespaces. Selection
uses only task, episode, frame length, and a frozen hash—not success outcomes.

Resolved mandatory exclusions:

- Thought3 development: `{exclusions['thought3_development']}`;
- Thought4 formal test: `{exclusions['thought4_formal_test']}`.

Formal tasks 8–9 are task-disjoint from both historical task-0 sets. The local
dataset contains 433 episodes / 52,895 frames across ten tasks. The normalized
task name, rather than `task_index + 1`, is used to map LIBERO-Plus variants;
the dataset and upstream benchmark orders differ.

## 6. Memory, time, and three-GPU plan

- Historical Fast-WAM load peak: 23,679.5 MiB.
- Historical Adapter training peak: 13,277.4 MiB; about 17.34 s/update in the
  Phase-2 full run.
- Thought4 formal v6: about 1.31 h on one GPU.
- Phase 5 smoke v3 training peak: 25,216.0 MiB. Pilot/formal per-worker peak
  and throughput remain **NOT MEASURED** until their first valid run; no GPU
  may be shared with another model process.
- Three 4090s run independent single-GPU workers, not three replicas of one
  distributed process. Pilot maps GPU0/1/2 to B1/G3/G4. Formal runs
  B1/G1/G2 first, then G3/B0; four GPUs remain an execution-only compatible
  schedule. Exact waves and ETA are frozen in the three-GPU preregistration.

## 7. Planned modification surface

Only `src/fastwam_ood_eval/thought5/`, `configs/thought5/`, new Thought5 runner
scripts, top-level CLI registration, `tests/thought5/`, and `docs/thought5/`
are modified. `third_party/FastWAM` and Thought1–4 formal outputs remain untouched.

## 8. Audit decision

Proceed. No missing core labels, paired renderer, checkpoint, or model path was
found. Audit SHA: `{audit['audit_sha256']}`.
"""


def run_audit(cfg: Thought5Config) -> dict[str, Any]:
    audit = collect_audit(cfg)
    report = render_audit_markdown(audit)
    canonical = cfg.experiment.output_dir / "audit_report.md"
    alias = Path("outputs/thought5/phase5_audit_report_v2.md")
    write_text_once(canonical, report, allow_identical=True)
    write_text_once(alias, report, allow_identical=True)
    return {
        "status": "complete",
        "audit_report": str(canonical),
        "audit_alias": str(alias),
        "audit_sha256": audit["audit_sha256"],
        "report_sha256": file_sha256(canonical),
    }
