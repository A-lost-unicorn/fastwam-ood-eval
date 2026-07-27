"""CPU mock Adapter training with action flow-matching semantics."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.cache_builder import mock_signal
from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.cache_validator import validate_cache
from fastwam_ood_eval.thought3.checkpointing import (
    find_latest_checkpoint,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.future_cache import FutureCacheReader
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
)
from fastwam_ood_eval.thought3.model_wrapper import AdapterConditionedModel
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import (
    AdapterCheckpointManifest,
    NATIVE_FUTURE_SHAPE,
)
from fastwam_ood_eval.thought3.training_dataset import (
    validate_training_example,
)


class Thought3TrainingError(RuntimeError):
    """Raised when mock training violates an experiment invariant."""


ACTION_HORIZON = 8
ACTION_DIMENSION = 7


class MockActionBackbone(nn.Module):
    """Frozen deterministic action encoder/head with the audited hook location."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < ACTION_DIMENSION:
            raise ValueError(
                f"mock action hidden_dim must be >= {ACTION_DIMENSION}"
            )
        self.action_encoder = nn.Linear(
            ACTION_DIMENSION,
            hidden_dim,
            bias=False,
        )
        self.action_head = nn.Linear(
            hidden_dim,
            ACTION_DIMENSION,
            bias=False,
        )
        with torch.no_grad():
            self.action_encoder.weight.zero_()
            self.action_head.weight.zero_()
            self.action_encoder.weight[:ACTION_DIMENSION, :].copy_(
                torch.eye(ACTION_DIMENSION)
            )
            self.action_head.weight[:, :ACTION_DIMENSION].copy_(
                torch.eye(ACTION_DIMENSION)
            )

    def forward(self, noisy_action: Tensor) -> Tensor:
        hidden = self.action_encoder(noisy_action)
        return self.action_head(hidden)


@dataclass(frozen=True)
class MockTrainingSample:
    base_sample_id: str
    split: str
    future_latent: Tensor
    future_mask: Tensor
    target_action: Tensor


def _target_action(base_sample_id: str) -> Tensor:
    signal = mock_signal(base_sample_id)
    time = torch.linspace(-0.1, 0.1, ACTION_HORIZON).view(-1, 1)
    dimensions = torch.linspace(-0.06, 0.06, ACTION_DIMENSION).view(1, -1)
    return torch.full((ACTION_HORIZON, ACTION_DIMENSION), signal) + time + dimensions


def load_mock_training_samples(
    cfg: Thought3Config,
) -> tuple[list[MockTrainingSample], str, str]:
    cache_report = validate_cache(cfg.cache.root)
    _, plan_manifest = load_cache_plan(cfg.cache.root)
    reader = FutureCacheReader(
        cfg.cache.root,
        expected_cache_fingerprint=cache_report["cache_fingerprint"],
        validate=False,
    )
    if cfg.variant in {"B0", "A-shuffle"}:
        raise Thought3TrainingError(
            f"variant={cfg.variant} is not an independently trained Adapter model"
        )
    read_k = cfg.sampler.active_k if cfg.sampler.active_k else 1
    base_ids = sorted(
        base_id for base_id, k in reader.keys if k == read_k
    )
    samples: list[MockTrainingSample] = []
    for base_id in base_ids:
        metadata = reader.metadata(base_id, read_k)
        record = metadata["record"]
        if cfg.variant == "A0":
            future = torch.zeros(NATIVE_FUTURE_SHAPE, dtype=torch.float32)
            mask = torch.ones(
                NATIVE_FUTURE_SHAPE[1:],
                dtype=torch.bool,
            )
        else:
            future, mask, _ = reader.get(base_id, read_k)
            future = future.float()
        sample = MockTrainingSample(
            base_sample_id=base_id,
            split=str(record["split"]),
            future_latent=future,
            future_mask=mask,
            target_action=_target_action(base_id),
        )
        # Exercise the same allowlist a real LIBERO sample must pass.
        validate_training_example(
            {
                "action_is_pad": torch.zeros(
                    ACTION_HORIZON, dtype=torch.bool
                ),
                "context": torch.zeros(1, 1),
                "context_mask": torch.ones(1, dtype=torch.bool),
                "current_proprio": torch.zeros(8),
                "current_rgb": torch.zeros(3, 2, 2),
                "future_latent": sample.future_latent,
                "future_mask": sample.future_mask,
                "metadata": {
                    "base_sample_id": base_id,
                    "source_kind": record["source_kind"],
                },
                "sample_id": record["cache_sample_id"],
                "target_action": sample.target_action,
            }
        )
        samples.append(sample)
    return (
        samples,
        str(plan_manifest["split_fingerprint"]),
        str(cache_report["cache_fingerprint"]),
    )


def build_mock_model(cfg: Thought3Config) -> AdapterConditionedModel:
    # Initialization is intentionally independent of K/variant.
    torch.manual_seed(cfg.experiment.seed)
    adapter = FutureToActionAdapter(
        FutureAdapterSpec(
            input_channels=cfg.adapter.input_channels,
            action_hidden_dim=cfg.adapter.action_hidden_dim,
            future_dim=cfg.adapter.future_dim,
            attention_dim=cfg.adapter.attention_dim,
            num_heads=cfg.adapter.num_heads,
            max_projected_grid=cfg.adapter.max_projected_grid,
            zero_init_gate=cfg.adapter.zero_init_gate,
        )
    )
    return AdapterConditionedModel(
        MockActionBackbone(cfg.adapter.action_hidden_dim),
        adapter,
    )


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _flow_batch(
    samples: Sequence[MockTrainingSample],
    *,
    train_seed: int,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    noisy_actions: list[Tensor] = []
    target_velocities: list[Tensor] = []
    futures: list[Tensor] = []
    masks: list[Tensor] = []
    for sample in samples:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                "thought3-action-flow-v1",
                train_seed,
                step,
                sample.base_sample_id,
            )
        )
        noise = torch.randn(
            sample.target_action.shape,
            generator=generator,
        )
        sigma = 0.15 + 0.70 * torch.rand((), generator=generator).item()
        noisy = sigma * noise + (1.0 - sigma) * sample.target_action
        # Same velocity convention as the audited noise→data flow schedule.
        target_velocity = noise - sample.target_action
        noisy_actions.append(noisy)
        target_velocities.append(target_velocity)
        futures.append(sample.future_latent.float())
        masks.append(sample.future_mask)
    return (
        torch.stack(noisy_actions),
        torch.stack(target_velocities),
        torch.stack(futures),
        torch.stack(masks),
    )


def _ordered_samples(
    samples: Iterable[MockTrainingSample],
    seed: int,
) -> list[MockTrainingSample]:
    return sorted(
        samples,
        key=lambda sample: hashlib.sha256(
            f"thought3-train-order-v1\0{seed}\0"
            f"{sample.base_sample_id}".encode("utf-8")
        ).hexdigest(),
    )


def _select_batch(
    samples: Sequence[MockTrainingSample],
    *,
    cursor: int,
    batch_size: int,
) -> tuple[list[MockTrainingSample], int]:
    if not samples:
        raise Thought3TrainingError("training split is empty")
    batch = [
        samples[(cursor + offset) % len(samples)]
        for offset in range(batch_size)
    ]
    return batch, cursor + batch_size


def _loss_for_samples(
    model: AdapterConditionedModel,
    samples: Sequence[MockTrainingSample],
    *,
    seed: int,
    step: int,
) -> Tensor:
    noisy, target_velocity, future, mask = _flow_batch(
        samples,
        train_seed=seed,
        step=step,
    )
    prediction = model(
        noisy,
        future_latent=future,
        future_mask=mask,
    )
    return (prediction.float() - target_velocity).square().mean()


@torch.no_grad()
def evaluate_mock_action_loss(
    model: AdapterConditionedModel,
    samples: Sequence[MockTrainingSample],
    *,
    seed: int,
) -> float:
    model.eval()
    values = [
        float(
            _loss_for_samples(
                model,
                [sample],
                seed=seed,
                step=90_000 + index,
            ).cpu()
        )
        for index, sample in enumerate(samples)
    ]
    model.train()
    return sum(values) / len(values)


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            raise Thought3TrainingError("Adapter gradient contains NaN/Inf")
        squared += float(gradient.square().sum().cpu())
    return math.sqrt(squared)


def _checkpoint_manifest(
    cfg: Thought3Config,
    model: AdapterConditionedModel,
    *,
    split_fingerprint: str,
    cache_fingerprint: str,
    global_step: int,
    sample_cursor: int,
    world_size: int,
) -> AdapterCheckpointManifest:
    return AdapterCheckpointManifest(
        backbone_checkpoint_sha256=cfg.backbone.checkpoint_sha256,
        dataset_stats_sha256=cfg.backbone.dataset_stats_sha256,
        fastwam_commit=cfg.backbone.fastwam_commit,
        adapter_fingerprint=cfg.adapter_structural_fingerprint,
        config_fingerprint=cfg.fingerprint,
        split_fingerprint=split_fingerprint,
        cache_fingerprint=cache_fingerprint,
        variant=cfg.variant,
        k=cfg.sampler.active_k,
        train_seed=cfg.training.train_seed,
        global_step=global_step,
        epoch=0,
        sample_cursor=sample_cursor,
        trainable_parameter_count=model.trainable_parameter_count,
        trainable_parameter_names=model.trainable_parameter_names,
        frozen_parameter_sha256=model.frozen_parameter_sha256,
        world_size=world_size,
        extra={
            "backend": "mock",
            "action_loss": "flow_matching_velocity_mse",
            "uses_ground_truth_future_input": False,
        },
    )


def run_mock_training(
    cfg: Thought3Config,
    *,
    resume: bool,
    device: str = "cpu",
    world_size: int = 1,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    if cfg.runtime.backend != "mock" or device != "cpu":
        raise Thought3TrainingError("Phase B training is backend=mock, device=cpu only")
    if world_size != 1:
        raise Thought3TrainingError(
            "Phase B executes one CPU worker; DDP partitioning is tested separately"
        )
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples, split_fingerprint, cache_fingerprint = load_mock_training_samples(cfg)
    train_samples = _ordered_samples(
        [sample for sample in samples if sample.split == "train"],
        cfg.training.train_seed,
    )
    development_samples = _ordered_samples(
        [sample for sample in samples if sample.split == "development"],
        cfg.training.train_seed,
    )
    if not development_samples:
        raise Thought3TrainingError("development split is empty")
    model = build_mock_model(cfg)
    model.adapter.capture_diagnostics = True
    model.train()
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    checkpoints_root = output / "checkpoints"
    latest = find_latest_checkpoint(checkpoints_root) if resume else None
    start_step = 0
    sample_cursor = 0
    if latest is not None:
        manifest = load_adapter_checkpoint(
            latest,
            adapter=model.adapter,
            optimizer=optimizer,
            expected={
                "adapter_fingerprint": cfg.adapter_structural_fingerprint,
                "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
                "cache_fingerprint": cache_fingerprint,
                "config_fingerprint": cfg.fingerprint,
                "dataset_stats_sha256": cfg.backbone.dataset_stats_sha256,
                "split_fingerprint": split_fingerprint,
                "variant": cfg.variant,
                "k": cfg.sampler.active_k,
            },
        )
        start_step = manifest.global_step
        sample_cursor = manifest.sample_cursor
    elif not resume and checkpoints_root.exists() and any(checkpoints_root.iterdir()):
        raise FileExistsError(
            f"training checkpoints already exist; pass --resume: {checkpoints_root}"
        )

    metrics_path = output / "train_metrics.jsonl"
    existing_metrics = load_jsonl(metrics_path) if metrics_path.is_file() else []
    if existing_metrics and not resume:
        raise FileExistsError(
            f"training metrics already exist; pass --resume: {metrics_path}"
        )
    initial_validation_loss = evaluate_mock_action_loss(
        model,
        development_samples,
        seed=cfg.training.train_seed,
    )
    started = time.perf_counter()
    new_metrics: list[dict[str, Any]] = []
    execution_stop = cfg.training.max_steps
    if stop_after_steps is not None:
        if stop_after_steps <= 0:
            raise Thought3TrainingError("stop_after_steps must be positive")
        execution_stop = min(execution_stop, stop_after_steps)
    if execution_stop < start_step:
        raise Thought3TrainingError(
            "stop_after_steps precedes the resumed checkpoint"
        )
    for step in range(start_step, execution_stop):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_samples = 0
        step_started = time.perf_counter()
        for accumulation_index in range(
            cfg.training.gradient_accumulation_steps
        ):
            batch, sample_cursor = _select_batch(
                train_samples,
                cursor=sample_cursor,
                batch_size=cfg.training.microbatch_size,
            )
            loss = _loss_for_samples(
                model,
                batch,
                seed=cfg.training.train_seed,
                step=step,
            )
            if cfg.training.gate_l2:
                loss = loss + cfg.training.gate_l2 * model.adapter.gate.square()
            scaled = loss / cfg.training.gradient_accumulation_steps
            scaled.backward()
            accumulated_loss += float(loss.detach().cpu()) * len(batch)
            accumulated_samples += len(batch)
        gradient_norm = _gradient_norm(model.adapter.parameters())
        optimizer.step()
        gate = float(model.adapter.gate.detach().cpu())
        diagnostics = model.adapter.last_diagnostics
        row: dict[str, Any] = {
            "attention_residual_norm": (
                diagnostics.attention_residual_norm
                if diagnostics is not None
                else None
            ),
            "device": "cpu",
            "gate_raw": gate,
            "gate_scale": math.tanh(gate),
            "global_step": step + 1,
            "gradient_norm": gradient_norm,
            "loss": accumulated_loss / accumulated_samples,
            "nan_or_inf": False,
            "peak_memory_mb": 0.0,
            "sample_cursor": sample_cursor,
            "step_time_ms": (time.perf_counter() - step_started) * 1000.0,
            "trainable_parameter_count": model.trainable_parameter_count,
        }
        if not all(
            math.isfinite(float(row[name]))
            for name in ("loss", "gradient_norm", "gate_raw", "gate_scale")
        ):
            raise Thought3TrainingError("training metric contains NaN/Inf")
        new_metrics.append(row)
        should_checkpoint = (
            (step + 1) % cfg.training.checkpoint_interval == 0
            or step + 1 == execution_stop
        )
        if should_checkpoint:
            manifest = _checkpoint_manifest(
                cfg,
                model,
                split_fingerprint=split_fingerprint,
                cache_fingerprint=cache_fingerprint,
                global_step=step + 1,
                sample_cursor=sample_cursor,
                world_size=world_size,
            )
            save_adapter_checkpoint(
                checkpoints_root / f"step_{step + 1:08d}",
                adapter=model.adapter,
                manifest=manifest,
                optimizer=optimizer,
            )
            atomic_write_jsonl(
                metrics_path,
                [
                    *[
                        row
                        for row in existing_metrics
                        if int(row["global_step"]) <= start_step
                    ],
                    *new_metrics,
                ],
            )
    atomic_write_jsonl(
        metrics_path,
        [
            *[
                row
                for row in existing_metrics
                if int(row["global_step"]) <= start_step
            ],
            *new_metrics,
        ],
    )
    final_validation_loss = evaluate_mock_action_loss(
        model,
        development_samples,
        seed=cfg.training.train_seed,
    )
    result = {
        "adapter_fingerprint": cfg.adapter_structural_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "checkpoint": str(find_latest_checkpoint(checkpoints_root)),
        "completed_steps": execution_stop,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "device": "cpu",
        "final_validation_action_loss": final_validation_loss,
        "initial_validation_action_loss": initial_validation_loss,
        "resumed_from_step": start_step,
        "runtime_seconds": time.perf_counter() - started,
        "split_fingerprint": split_fingerprint,
        "train_sample_count": len(train_samples),
        "trainable_parameter_count": model.trainable_parameter_count,
        "uses_ground_truth_future_input": False,
        "validation_sample_count": len(development_samples),
        "variant": cfg.variant,
        "status": (
            "complete"
            if execution_stop == cfg.training.max_steps
            else "intentional_test_interruption"
        ),
    }
    atomic_write_json(output / "training_manifest.json", result)
    model.close()
    return result
