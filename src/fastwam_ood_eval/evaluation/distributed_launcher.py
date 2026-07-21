"""torchrun-compatible episode-level worker entry point."""

from __future__ import annotations

import os

from fastwam_ood_eval.config import EvalConfig
from fastwam_ood_eval.evaluation.evaluator import evaluate_worker


def distributed_evaluate(
    cfg: EvalConfig,
    *,
    device: str | None = None,
    dry_run: bool = False,
    rerun: str = "incomplete",
) -> dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    selected_device = device or ("cpu" if cfg.benchmark.backend == "mock" else f"cuda:{local_rank}")
    return evaluate_worker(
        cfg,
        rank=rank,
        world_size=world_size,
        device=selected_device,
        dry_run=dry_run,
        rerun=rerun,
    )

