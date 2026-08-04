"""Lazy command dispatch for the isolated Thought5 namespace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought5.audit import collect_audit, run_audit
from fastwam_ood_eval.thought5.config import config_summary, load_thought5_config
from fastwam_ood_eval.thought5.pipeline import materialize_dry_run


def dispatch(args: Any) -> int:
    if args.set:
        raise ValueError(
            "Thought5 forbids CLI overrides; create and commit a new versioned config"
        )
    cfg = load_thought5_config(args.config)
    expected_stage = {
        "thought5-audit": "audit",
        "thought5-smoke": "smoke",
        "thought5-pilot": "pilot",
        "thought5-formal": "formal",
    }.get(args.command)
    if args.command == "thought5-dry-run":
        result = materialize_dry_run(cfg) if not args.dry_run else config_summary(cfg)
    elif args.command == "thought5-audit":
        result = collect_audit(cfg) if args.dry_run else run_audit(cfg)
    else:
        if expected_stage != cfg.experiment.stage:
            raise ValueError(
                f"command expects stage={expected_stage}, config has {cfg.experiment.stage}"
            )
        if args.device is not None and args.device != cfg.runtime.device:
            raise ValueError("logical device must match the frozen config")
        if args.dry_run:
            result = {
                "status": "validated",
                "scientific_result": False,
                "would_run": expected_stage,
                "config": config_summary(cfg),
            }
        else:
            from fastwam_ood_eval.thought5.real_runtime import run_real_stage

            result = run_real_stage(cfg, resume=bool(args.resume))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0
