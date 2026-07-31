"""Lazy CLI dispatch for the isolated Thought4 namespace."""

from __future__ import annotations

import json
from typing import Any

from fastwam_ood_eval.thought4.config import load_thought4_config
from fastwam_ood_eval.thought4.phase4 import (
    dry_run_payload,
    run_formal_diagnosis,
    run_real_smoke,
)


def dispatch(args: Any) -> int:
    if args.set:
        raise ValueError(
            "Thought4 forbids CLI config overrides; edit and commit a new config"
        )
    cfg = load_thought4_config(args.config)
    if args.device is not None and args.device != cfg.runtime.device:
        raise ValueError(
            "Thought4 device is frozen in config; runner maps the physical GPU "
            "to logical cuda:0 through CUDA_VISIBLE_DEVICES"
        )
    stage = (
        "smoke"
        if args.command == "thought4-phase4-smoke"
        else "formal"
        if args.command == "thought4-phase4-diagnosis"
        else None
    )
    if stage is None:
        raise ValueError(f"unknown Thought4 command: {args.command}")
    if args.dry_run:
        result = dry_run_payload(cfg, stage=stage)
    elif stage == "smoke":
        result = run_real_smoke(cfg, resume=args.resume)
    else:
        result = run_formal_diagnosis(cfg, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

