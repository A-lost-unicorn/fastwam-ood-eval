"""Lazy command dispatch for the isolated Thought3 namespace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought3.cache_planner import create_cache_plan
from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    load_thought3_config,
)
from fastwam_ood_eval.thought3.io_utils import sha256_file


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _load(args: Any) -> Thought3Config:
    overrides = list(args.set)
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    return load_thought3_config(args.config, overrides)


def _dry_run_payload(cfg: Thought3Config, command: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": cfg.runtime.backend,
        "command": command,
        "config_fingerprint": cfg.fingerprint,
        "device": cfg.runtime.device,
        "dry_run": True,
        "output_dir": str(cfg.experiment.output_dir),
        "variant": cfg.variant,
        "would_load_checkpoint": False,
        "would_load_fastwam": False,
        "would_write": False,
    }
    if command in {"thought3-plan-cache", "thought3-build-cache"}:
        entries, split, plan = create_cache_plan(cfg)
        payload.update(
            {
                "cache_entry_count": len(entries),
                "cache_fingerprint": plan["cache_fingerprint"],
                "sample_count": plan["sample_count"],
                "split_fingerprint": split.fingerprint,
            }
        )
    return payload


def _audit(cfg: Thought3Config) -> dict[str, Any]:
    documents = (
        Path("docs/thought3_upstream_audit.md"),
        Path("docs/thought3_design.md"),
        Path("docs/thought3_risk_register.md"),
    )
    return {
        "config_fingerprint": cfg.fingerprint,
        "documents": {
            str(path): {
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for path in documents
        },
        "phase_a_confirmed": True,
        "phase_b_backend": "cpu_mock_only",
        "status": "ready",
    }


def _run_counterfactual(cfg: Thought3Config) -> dict[str, Any]:
    import torch

    from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
    from fastwam_ood_eval.thought3.cache_validator import validate_cache
    from fastwam_ood_eval.thought3.checkpointing import (
        find_latest_checkpoint,
        load_adapter_checkpoint,
    )
    from fastwam_ood_eval.thought3.counterfactuals import (
        ShuffleCandidate,
        build_shuffle_pairs,
        run_action_counterfactuals,
        shuffle_manifest,
    )
    from fastwam_ood_eval.thought3.future_cache import FutureCacheReader
    from fastwam_ood_eval.thought3.io_utils import atomic_write_json
    from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
    from fastwam_ood_eval.thought3.trainer import build_mock_model

    cache_report = validate_cache(cfg.cache.root)
    _, plan = load_cache_plan(cfg.cache.root)
    reader = FutureCacheReader(
        cfg.cache.root,
        expected_cache_fingerprint=cache_report["cache_fingerprint"],
        validate=False,
    )
    candidates = [
        ShuffleCandidate.from_cache_metadata(reader.metadata(base_id, k))
        for base_id, k in reader.keys
    ]
    pairs = build_shuffle_pairs(candidates, seed=cfg.experiment.seed)
    manifest = shuffle_manifest(pairs, seed=cfg.experiment.seed)
    output = ensure_thought3_output_path(
        cfg.experiment.output_dir / "counterfactual"
    )
    output.mkdir(parents=True, exist_ok=True)
    path = atomic_write_json(output / "shuffle_manifest.json", manifest)
    result: dict[str, Any] = {
        "fingerprint": manifest["fingerprint"],
        "manifest": str(path),
        "pair_count": len(pairs),
        "status": "shuffle_manifest_ready",
    }
    checkpoint = find_latest_checkpoint(
        cfg.experiment.output_dir / "checkpoints"
    )
    if checkpoint is None:
        result["action_metrics_status"] = "checkpoint_missing"
        return result
    if cfg.runtime.backend != "mock":
        result["action_metrics_status"] = "phase_c_fastwam_not_implemented"
        return result
    active_k = cfg.sampler.active_k if cfg.sampler.active_k else 1
    eligible = [
        pair
        for pair in pairs
        if pair.k == active_k and pair.split == "development"
    ]
    if not eligible:
        eligible = [pair for pair in pairs if pair.k == active_k]
    if not eligible:
        raise RuntimeError(f"no legal A-shuffle pair for K={active_k}")
    pair = eligible[0]
    correct, _, _ = reader.get(pair.recipient_base_sample_id, active_k)
    shuffled, _, _ = reader.get(pair.donor_base_sample_id, active_k)
    different_k = {
        k: reader.get(pair.recipient_base_sample_id, k)[0].unsqueeze(0).float()
        for k in (1, 2, 4)
        if k != active_k
    }
    model = build_mock_model(cfg)
    load_adapter_checkpoint(
        checkpoint,
        adapter=model.adapter,
        expected={
            "adapter_fingerprint": cfg.adapter_structural_fingerprint,
            "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
            "cache_fingerprint": cache_report["cache_fingerprint"],
            "config_fingerprint": cfg.fingerprint,
            "dataset_stats_sha256": cfg.backbone.dataset_stats_sha256,
            "split_fingerprint": str(plan["split_fingerprint"]),
            "variant": cfg.variant,
            "k": cfg.sampler.active_k,
        },
    )
    model.eval()

    @torch.no_grad()
    def action_function(future: torch.Tensor, action_seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(action_seed)
        action = torch.randn((1, 8, 7), generator=generator)
        for _ in range(cfg.runtime.action_denoise_steps):
            velocity = model(
                action,
                future_latent=future.float(),
            )
            action = action - velocity / cfg.runtime.action_denoise_steps
        return action

    action_metrics = run_action_counterfactuals(
        action_function,
        correct_future=correct.unsqueeze(0).float(),
        shuffled_future=shuffled.unsqueeze(0).float(),
        action_seed=cfg.experiment.seed,
        different_k_futures=different_k,
    )
    action_metrics.update(
        {
            "checkpoint": str(checkpoint),
            "mock_only_no_scientific_claim": True,
            "recipient_base_sample_id": pair.recipient_base_sample_id,
            "donor_base_sample_id": pair.donor_base_sample_id,
            "task_success_change": None,
        }
    )
    action_path = atomic_write_json(
        output / "action_counterfactual.json",
        action_metrics,
    )
    model.close()
    result.update(
        {
            "action_metrics": str(action_path),
            "action_metrics_status": "mock_complete",
            "status": "complete",
        }
    )
    return result


def _run_mock_evaluation(
    cfg: Thought3Config,
    *,
    resume: bool,
) -> dict[str, Any]:
    import math

    import torch

    from fastwam_ood_eval.thought3.evaluator import (
        OnlineFutureActionEvaluator,
    )
    from fastwam_ood_eval.thought3.checkpointing import (
        find_latest_checkpoint,
        load_adapter_checkpoint,
    )
    from fastwam_ood_eval.thought3.future_sampler import (
        make_mock_future_sampler,
    )
    from fastwam_ood_eval.thought3.io_utils import (
        atomic_write_json,
        atomic_write_jsonl,
        load_jsonl,
    )
    from fastwam_ood_eval.thought3.latency import summarize_latency
    from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
    from fastwam_ood_eval.thought3.trainer import (
        MockActionBackbone,
        build_mock_model,
    )

    if cfg.runtime.backend != "mock" or cfg.runtime.device != "cpu":
        raise RuntimeError("Phase B online evaluation is backend=mock, device=cpu only")
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    results_path = output / "mock_online_evaluation.jsonl"
    if results_path.exists():
        if resume:
            rows = load_jsonl(results_path)
            return {
                "episodes": len(rows),
                "results": str(results_path),
                "resumed": True,
                "status": "already_complete",
            }
        raise FileExistsError(
            f"mock evaluation exists; pass --resume: {results_path}"
        )
    output.mkdir(parents=True, exist_ok=True)
    backbone = MockActionBackbone(cfg.adapter.action_hidden_dim)
    conditioned = None
    sampler = None
    null_future = False
    k = cfg.sampler.active_k
    checkpoint: Path | None = None
    if cfg.variant != "B0":
        conditioned = build_mock_model(cfg)
        backbone = conditioned.backbone
        null_future = cfg.variant == "A0"
        sampler = None if null_future else make_mock_future_sampler()
        checkpoint = find_latest_checkpoint(
            cfg.experiment.output_dir / "checkpoints"
        )
        if checkpoint is not None:
            load_adapter_checkpoint(
                checkpoint,
                adapter=conditioned.adapter,
                expected={
                    "adapter_fingerprint": cfg.adapter_structural_fingerprint,
                    "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
                    "config_fingerprint": cfg.fingerprint,
                    "dataset_stats_sha256": cfg.backbone.dataset_stats_sha256,
                    "variant": cfg.variant,
                    "k": cfg.sampler.active_k,
                },
            )
    evaluator = OnlineFutureActionEvaluator(
        backbone=backbone,
        conditioned_model=conditioned,
        sampler=sampler,
        action_denoise_steps=cfg.runtime.action_denoise_steps,
    )
    rows: list[dict[str, Any]] = []
    latencies = []
    for episode_index in range(8):
        signal = -0.7 + episode_index * 0.2
        current = torch.full((1, 48, 1, 14, 28), signal)
        donor = (
            torch.full_like(current, -signal)
            if cfg.variant == "A-shuffle"
            else None
        )
        result = evaluator.predict(
            current,
            initial_noise_seed=cfg.sampler.global_cache_seed + episode_index,
            action_noise_seed=cfg.experiment.seed + episode_index,
            k=k,
            null_future=null_future,
            shuffled_donor_current=donor,
        )
        latencies.append(result.latency)
        action_norm = float(result.action_chunk.float().norm().cpu())
        rows.append(
            {
                "action_norm": action_norm,
                "episode_index": episode_index,
                "k": k,
                "latency": result.latency.to_dict(),
                "mock_success": bool(math.isfinite(action_norm) and action_norm < 20),
                "online_cache_read": False,
                "variant": cfg.variant,
            }
        )
    atomic_write_jsonl(results_path, rows)
    summary = {
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "episodes": len(rows),
        "latency": summarize_latency(latencies),
        "mock_only_no_scientific_claim": True,
        "mock_success_rate": sum(row["mock_success"] for row in rows) / len(rows),
        "online_cache_read": False,
        "adapter_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "results": str(results_path),
        "variant": cfg.variant,
    }
    atomic_write_json(output / "mock_online_evaluation_manifest.json", summary)
    if conditioned is not None:
        conditioned.close()
    return summary


def _aggregate(cfg: Thought3Config) -> dict[str, Any]:
    from fastwam_ood_eval.thought3.io_utils import (
        atomic_write_json,
        load_jsonl,
    )
    from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    rows = load_jsonl(output / "mock_online_evaluation.jsonl")
    if not rows:
        raise RuntimeError("no Thought3 evaluation rows to aggregate")
    totals = [float(row["latency"]["total_policy_ms"]) for row in rows]
    summary = {
        "episodes": len(rows),
        "mock_only_no_scientific_claim": True,
        "mock_success_rate": sum(bool(row["mock_success"]) for row in rows)
        / len(rows),
        "online_cache_read": any(bool(row["online_cache_read"]) for row in rows),
        "total_policy_latency_mean_ms": sum(totals) / len(totals),
        "variant": cfg.variant,
    }
    path = atomic_write_json(output / "mock_aggregate.json", summary)
    return {**summary, "summary": str(path)}


def _report(cfg: Thought3Config) -> dict[str, Any]:
    from fastwam_ood_eval.thought3.io_utils import atomic_write_text, load_json
    from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    summary = load_json(output / "mock_aggregate.json")
    report = output / "mock_report.md"
    text = (
        "# Thought3 Phase B mock report\n\n"
        "> This is a CPU/mock engineering validation. It is not an ID/OOD result "
        "and supports no scientific claim about Fast-WAM.\n\n"
        f"- Variant: `{summary['variant']}`\n"
        f"- Episodes: {summary['episodes']}\n"
        f"- Mock success rate: {summary['mock_success_rate']:.3f}\n"
        f"- Mean total policy latency: "
        f"{summary['total_policy_latency_mean_ms']:.3f} ms\n"
        f"- Online training-cache reads: `{summary['online_cache_read']}`\n"
    )
    atomic_write_text(report, text)
    return {"report": str(report), "status": "written"}


def dispatch(args: Any) -> int:
    cfg = _load(args)
    if args.dry_run:
        _emit(_dry_run_payload(cfg, args.command))
        return 0
    if args.command == "thought3-audit":
        result = _audit(cfg)
    elif args.command == "thought3-smoke-real":
        from fastwam_ood_eval.thought3.phase_c_smoke import run_phase_c_smoke

        result = run_phase_c_smoke(cfg, resume=args.resume)
    elif args.command == "thought3-cache-real-smoke":
        from fastwam_ood_eval.thought3.phase_d_cache_smoke import (
            run_phase_d_cache_smoke,
        )

        result = run_phase_d_cache_smoke(cfg, resume=args.resume)
    elif args.command == "thought3-train-real-smoke":
        from fastwam_ood_eval.thought3.phase_e_training_smoke import (
            run_phase_e_training_smoke,
        )

        result = run_phase_e_training_smoke(cfg, resume=args.resume)
    elif args.command == "thought3-diagnose-real-overfit":
        from fastwam_ood_eval.thought3.phase_e1_overfit import (
            run_phase_e1_overfit,
        )

        result = run_phase_e1_overfit(cfg, resume=args.resume)
    elif args.command == "thought3-diagnose-real-eight-sample":
        from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
            run_phase_e2_eight_sample,
        )

        result = run_phase_e2_eight_sample(cfg, resume=args.resume)
    elif args.command == "thought3-diagnose-heldout-multiflow":
        from fastwam_ood_eval.thought3.phase_e3_multiflow import (
            run_phase_e3_multiflow,
        )

        result = run_phase_e3_multiflow(cfg, resume=args.resume)
    elif args.command == "thought3-diagnose-diversified-flow":
        from fastwam_ood_eval.thought3.phase_e4_diversified_flow import (
            run_phase_e4_diversified_flow,
        )

        result = run_phase_e4_diversified_flow(
            cfg,
            resume=args.resume,
        )
    elif args.command == "thought3-diagnose-objective-aggregation":
        from fastwam_ood_eval.thought3.phase_e5_objective_aggregation import (
            run_phase_e5_objective_aggregation,
        )

        result = run_phase_e5_objective_aggregation(
            cfg,
            resume=args.resume,
        )
    elif args.command == "thought3-replicate-fresh-cohort":
        from fastwam_ood_eval.thought3.phase_e6_fresh_cohort_replication import (
            run_phase_e6_fresh_cohort_replication,
        )

        result = run_phase_e6_fresh_cohort_replication(
            cfg,
            resume=args.resume,
        )
    elif args.command == "thought3-diagnose-checkpoint-trajectory":
        from fastwam_ood_eval.thought3.phase_e7_checkpoint_trajectory import (
            run_phase_e7_checkpoint_trajectory,
        )

        result = run_phase_e7_checkpoint_trajectory(
            cfg,
            resume=args.resume,
        )
    elif args.command == "thought3-replicate-a0-flow-variance":
        from fastwam_ood_eval.thought3.phase_e8_a0_flow_variance_replication import (
            run_phase_e8_a0_flow_variance_replication,
        )

        result = run_phase_e8_a0_flow_variance_replication(
            cfg,
            resume=args.resume,
        )
    elif args.command == "thought3-diagnose-sample-tail-mitigation":
        from fastwam_ood_eval.thought3.phase_e9_sample_tail_mitigation import (
            run_phase_e9_sample_tail_mitigation,
        )

        result = run_phase_e9_sample_tail_mitigation(
            cfg,
            resume=args.resume,
        )
    elif args.command == "thought3-plan-cache":
        from fastwam_ood_eval.thought3.cache_planner import write_cache_plan

        result = write_cache_plan(cfg, resume=args.resume)
    elif args.command == "thought3-build-cache":
        from fastwam_ood_eval.thought3.cache_builder import build_cache

        result = build_cache(
            cfg,
            resume=args.resume,
            rank=args.rank,
            world_size=args.world_size,
            device=cfg.runtime.device,
        )
    elif args.command == "thought3-validate-cache":
        from fastwam_ood_eval.thought3.cache_validator import validate_cache

        result = validate_cache(cfg.cache.root)
    elif args.command == "thought3-train":
        from fastwam_ood_eval.thought3.trainer import run_mock_training

        result = run_mock_training(
            cfg,
            resume=args.resume,
            device=cfg.runtime.device,
            world_size=args.world_size,
        )
    elif args.command == "thought3-counterfactual":
        result = _run_counterfactual(cfg)
    elif args.command == "thought3-evaluate":
        result = _run_mock_evaluation(cfg, resume=args.resume)
    elif args.command == "thought3-aggregate":
        result = _aggregate(cfg)
    elif args.command == "thought3-report":
        result = _report(cfg)
    else:
        raise ValueError(f"unknown Thought3 command: {args.command}")
    _emit(result)
    return 0
