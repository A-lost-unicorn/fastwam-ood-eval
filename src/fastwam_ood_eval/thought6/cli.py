"""Lazy, fail-closed command dispatch for the isolated Thought6 namespace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought6.audit import collect_audit, run_audit
from fastwam_ood_eval.thought6.config import config_summary, load_thought6_config
from fastwam_ood_eval.thought6.rollout_policy import mock_online_contract
from fastwam_ood_eval.thought6.schemas import Thought6Error
from fastwam_ood_eval.thought6.task_selection import assert_phase6b_data_ready


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Thought6Error(f"required gate artifact is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Thought6Error(f"invalid gate artifact: {path}")
    return value


def _phase6b_preflight(cfg: Any) -> dict[str, Any]:
    smoke = _load_json(cfg.output_dir / "phase6a_smoke_results.json")
    if smoke.get("status") != "complete" or smoke.get("phase6b_unlocked") is not True:
        raise Thought6Error("Phase 6B remains locked until the real Phase 6A smoke passes")
    tasks = _load_json(cfg.output_dir / "task_selection_manifest.json")
    assert_phase6b_data_ready(tasks)
    return {"smoke": smoke, "tasks": tasks}


def _phase6c_preflight(cfg: Any, *, stage: int) -> dict[str, Any]:
    gate = _load_json(cfg.output_dir / "phase6b_gate_decision.json")
    if gate.get("status") != "complete" or gate.get("phase6c_unlocked") is not True:
        raise Thought6Error("Phase 6C remains locked until all five Phase 6B gates pass")
    if stage == 2:
        stage1 = _load_json(cfg.output_dir / "phase6c_stage1_results.json")
        decision = stage1.get("stage2_decision", {})
        if stage1.get("status") != "complete" or decision.get("stage2_unlocked") is not True:
            raise Thought6Error("Phase 6C Stage 2 does not meet the preregistered expansion rule")
    return {"phase6b_gate": gate}


def dispatch(args: Any) -> int:
    if args.set:
        raise Thought6Error("Thought6 forbids CLI overrides; the 0.5 threshold is code-frozen")
    cfg = load_thought6_config(args.config)
    expected = {
        "thought6-audit": "audit",
        "thought6-phase6a-smoke": "phase6a",
        "thought6-phase6b-utility": "phase6b",
        "thought6-phase6c-stage1": "phase6c_stage1",
        "thought6-phase6c-stage2": "phase6c_stage2",
    }.get(args.command)
    if args.command == "thought6-dry-run":
        audit = collect_audit(cfg)
        result = {
            "schema_version": "thought6.cpu_mock_dry_run.v1",
            "status": "complete",
            "scientific_result": False,
            "config": config_summary(cfg),
            "mock_online_contract": mock_online_contract(),
            "adapter_resolved": audit["adapter"]["status"] == "resolved_unique",
            "task_count": len(audit["task_selection"]["selected_tasks"]),
            "phase6b_ready": audit["phase6b_ready"],
            "missing_suite_datasets": audit["missing_suite_datasets"],
            "would_load_model": False,
            "would_create_optimizer": False,
            "would_start_gpu": False,
        }
    elif args.command == "thought6-audit":
        if cfg.stage != "audit":
            raise Thought6Error("thought6-audit requires stage=audit")
        result = collect_audit(cfg) if args.dry_run else run_audit(cfg)
    else:
        if expected != cfg.stage:
            raise Thought6Error(f"command expects stage={expected}, config has {cfg.stage}")
        if args.device is not None and args.device != cfg.device:
            raise Thought6Error("logical device must match the frozen config")
        if args.dry_run:
            result = {
                "status": "validated",
                "scientific_result": False,
                "would_run": expected,
                "config": config_summary(cfg),
                "mock_online_contract": mock_online_contract(),
            }
        elif args.command == "thought6-phase6a-smoke":
            from fastwam_ood_eval.thought6.rollout_policy import run_phase6a_smoke

            result = run_phase6a_smoke(cfg)
        elif args.command == "thought6-phase6b-utility":
            _phase6b_preflight(cfg)
            raise Thought6Error(
                "this immutable v1 audit was frozen with missing spatial/object/libero_10 "
                "episode provenance; create a fresh protocol namespace after installing all datasets"
            )
        elif args.command == "thought6-phase6c-stage1":
            _phase6c_preflight(cfg, stage=1)
            raise Thought6Error(
                "rollout execution is unavailable before a completed v1 Phase 6B result; no rollout started"
            )
        else:
            _phase6c_preflight(cfg, stage=2)
            raise Thought6Error(
                "Stage 2 execution is unavailable before an explicitly unlocked Stage 1; no rollout started"
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0

