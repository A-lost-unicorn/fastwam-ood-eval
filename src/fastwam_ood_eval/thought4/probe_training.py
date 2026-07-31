"""Deterministic probe training and mandatory control baselines."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from fastwam_ood_eval.thought4.feature_hooks import FeatureHookError
from fastwam_ood_eval.thought4.probe_models import build_probe, probe_parameter_count


class ProbeTrainingError(RuntimeError):
    """Raised when probe data/training violates the frozen protocol."""


TaskType = Literal["regression", "binary"]


@dataclass(frozen=True)
class ProbeDataset:
    features: Any
    targets: Any
    valid_mask: Any | None
    episode_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def validate(self) -> None:
        import torch

        if not isinstance(self.features, torch.Tensor) or self.features.ndim != 2:
            raise ProbeTrainingError("features must be a [N,D] Tensor")
        if not isinstance(self.targets, torch.Tensor) or self.targets.ndim < 2:
            raise ProbeTrainingError("targets must be a [N,...] Tensor")
        count = self.features.shape[0]
        if self.targets.shape[0] != count:
            raise ProbeTrainingError("feature/target sample counts differ")
        if len(self.episode_ids) != count or len(self.sample_ids) != count:
            raise ProbeTrainingError("identity counts differ from tensor rows")
        if len(set(self.sample_ids)) != count:
            raise ProbeTrainingError("sample IDs must be unique within a split")
        if self.valid_mask is not None:
            if not isinstance(self.valid_mask, torch.Tensor):
                raise ProbeTrainingError("valid_mask must be a Tensor")
            if self.valid_mask.shape[0] != count:
                raise ProbeTrainingError("valid_mask sample count differs")
        for name, tensor in (
            ("features", self.features),
            ("targets", self.targets),
        ):
            if not tensor.is_floating_point() or not bool(tensor.isfinite().all().item()):
                raise ProbeTrainingError(f"{name} must be finite floating point")
        if self.features.requires_grad:
            raise ProbeTrainingError("probe features must be detached")


@dataclass(frozen=True)
class ProbeTrainingSpec:
    kind: str
    task_type: TaskType
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    batch_size: int
    seed: int

    def validate(self) -> None:
        if self.kind not in {"linear", "mlp"}:
            raise ProbeTrainingError("probe kind must be linear or mlp")
        if self.task_type not in {"regression", "binary"}:
            raise ProbeTrainingError("task_type must be regression or binary")
        if min(
            self.hidden_dim,
            self.max_epochs,
            self.patience,
            self.batch_size,
        ) <= 0:
            raise ProbeTrainingError("probe integer hyperparameters must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.seed < 0:
            raise ProbeTrainingError("invalid optimizer hyperparameters/seed")


@dataclass
class TrainedProbe:
    model: Any
    best_epoch: int
    best_development_loss: float
    history: tuple[dict[str, float], ...]
    parameter_count: int
    shuffled_labels: bool
    feature_mean: Any
    feature_std: Any
    target_mean: Any
    target_std: Any


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_targets(dataset: ProbeDataset) -> tuple[Any, Any | None]:
    targets = dataset.targets.reshape(dataset.targets.shape[0], -1)
    if dataset.valid_mask is None:
        return targets, None
    mask = dataset.valid_mask
    while mask.ndim < dataset.targets.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(dataset.targets).reshape(dataset.targets.shape[0], -1)
    return targets, mask


def _train_only_standardization(
    train: ProbeDataset,
    train_target: Any,
    train_mask: Any | None,
    *,
    task_type: TaskType,
    floor: float = 1e-6,
) -> tuple[Any, Any, Any, Any]:
    import torch

    feature_mean = train.features.float().mean(dim=0)
    feature_std = train.features.float().std(dim=0, unbiased=False)
    feature_std = feature_std.clamp_min(floor)
    target = train_target.float()
    if task_type == "binary":
        valid_target = target if train_mask is None else target[train_mask.bool()]
        if valid_target.numel() == 0:
            raise ProbeTrainingError("binary target has no valid train label")
        if bool(((valid_target < 0) | (valid_target > 1)).any().item()):
            raise ProbeTrainingError("binary targets must lie in [0,1]")
        # BCEWithLogits consumes probabilities directly, so only features are
        # standardized for classification tasks.
        target_mean = torch.zeros(target.shape[1], dtype=torch.float32)
        target_std = torch.ones(target.shape[1], dtype=torch.float32)
    else:
        if train_mask is None:
            target_mean = target.mean(dim=0)
            target_std = target.std(dim=0, unbiased=False)
        else:
            valid = train_mask.float()
            counts = valid.sum(dim=0)
            if bool((counts <= 0).any().item()):
                raise ProbeTrainingError(
                    "at least one flattened target has no valid train label"
                )
            target_mean = (target * valid).sum(dim=0) / counts
            variance = (
                (target - target_mean).square() * valid
            ).sum(dim=0) / counts
            target_std = variance.sqrt()
        target_std = target_std.clamp_min(floor)
    for name, value in (
        ("feature_mean", feature_mean),
        ("feature_std", feature_std),
        ("target_mean", target_mean),
        ("target_std", target_std),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ProbeTrainingError(f"{name} contains NaN/Inf")
    return feature_mean, feature_std, target_mean, target_std


def _masked_loss(
    prediction: Any,
    target: Any,
    mask: Any | None,
    *,
    task_type: TaskType,
) -> Any:
    import torch
    from torch.nn import functional as F

    if task_type == "regression":
        losses = (prediction.float() - target.float()).square()
    else:
        losses = F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )
    if mask is None:
        return losses.mean()
    valid = mask.to(device=losses.device, dtype=losses.dtype)
    denominator = valid.sum()
    if float(denominator.item()) <= 0:
        raise ProbeTrainingError("batch has no valid labels")
    return (losses * valid).sum() / denominator


def _check_split_disjoint(
    train: ProbeDataset,
    development: ProbeDataset,
) -> None:
    overlap_samples = set(train.sample_ids) & set(development.sample_ids)
    overlap_episodes = set(train.episode_ids) & set(development.episode_ids)
    if overlap_samples or overlap_episodes:
        raise ProbeTrainingError(
            "train/development split leaks samples or episodes"
        )


def shuffled_targets(targets: Any, *, seed: int) -> Any:
    import torch

    if targets.shape[0] < 2:
        raise ProbeTrainingError("shuffled-label control needs at least two samples")
    permutation = shuffled_permutation(targets.shape[0], seed=seed)
    return targets[permutation.to(targets.device)]


def shuffled_permutation(count: int, *, seed: int) -> Any:
    import torch

    if count < 2:
        raise ProbeTrainingError("shuffled-label control needs at least two samples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    identity = torch.arange(count)
    for _attempt in range(128):
        permutation = torch.randperm(count, generator=generator)
        if not bool((permutation == identity).any().item()):
            return permutation
    # Deterministic fail-safe that is a derangement for every count >= 2.
    return identity.roll(1)


def train_probe(
    train: ProbeDataset,
    development: ProbeDataset,
    spec: ProbeTrainingSpec,
    *,
    shuffled_label_control: bool = False,
    device: str = "cpu",
) -> TrainedProbe:
    import torch

    train.validate()
    development.validate()
    spec.validate()
    _check_split_disjoint(train, development)
    _set_seed(spec.seed)
    train_target, train_mask = _flatten_targets(train)
    development_target, development_mask = _flatten_targets(development)
    feature_mean, feature_std, target_mean, target_std = (
        _train_only_standardization(
            train,
            train_target,
            train_mask,
            task_type=spec.task_type,
        )
    )
    if shuffled_label_control:
        permutation = shuffled_permutation(
            train_target.shape[0], seed=spec.seed + 1009
        )
        train_target = train_target[permutation.to(train_target.device)]
        if train_mask is not None:
            train_mask = train_mask[permutation.to(train_mask.device)]
    model = build_probe(
        spec.kind,
        input_dim=int(train.features.shape[1]),
        output_dim=int(train_target.shape[1]),
        hidden_dim=spec.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    train_features = (
        (train.features.detach().float() - feature_mean) / feature_std
    ).to(device)
    train_target = (
        (train_target.detach().float() - target_mean) / target_std
    ).to(device)
    train_mask = train_mask.detach().to(device) if train_mask is not None else None
    development_features = (
        (development.features.detach().float() - feature_mean) / feature_std
    ).to(device)
    development_target = (
        (development_target.detach().float() - target_mean) / target_std
    ).to(device)
    development_mask = (
        development_mask.detach().to(device)
        if development_mask is not None
        else None
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed + 1)
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(spec.max_epochs):
        model.train()
        permutation = torch.randperm(
            train_features.shape[0], generator=generator
        )
        losses: list[float] = []
        for start in range(0, len(permutation), spec.batch_size):
            indices = permutation[start : start + spec.batch_size].to(device)
            prediction = model(train_features[indices])
            loss = _masked_loss(
                prediction,
                train_target[indices],
                train_mask[indices] if train_mask is not None else None,
                task_type=spec.task_type,
            )
            if not bool(torch.isfinite(loss).item()):
                raise ProbeTrainingError("probe train loss is NaN/Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prediction = model(development_features)
            development_loss = _masked_loss(
                prediction,
                development_target,
                development_mask,
                task_type=spec.task_type,
            )
        value = float(development_loss.cpu())
        if not math.isfinite(value):
            raise ProbeTrainingError("probe development loss is NaN/Inf")
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": sum(losses) / len(losses),
                "development_loss": value,
            }
        )
        if value < best_loss - 1e-12:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= spec.patience:
            break
    if best_state is None:
        raise ProbeTrainingError("probe training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedProbe(
        model=model,
        best_epoch=best_epoch,
        best_development_loss=best_loss,
        history=tuple(history),
        parameter_count=probe_parameter_count(model),
        shuffled_labels=shuffled_label_control,
        feature_mean=feature_mean.detach().cpu(),
        feature_std=feature_std.detach().cpu(),
        target_mean=target_mean.detach().cpu(),
        target_std=target_std.detach().cpu(),
    )


def target_baselines(
    train_targets: Any,
    evaluation_targets: Any,
    *,
    train_mask: Any | None = None,
) -> dict[str, Any]:
    """Return constant-zero and train-target-mean predictions."""

    import torch

    if train_targets.ndim < 2 or evaluation_targets.ndim < 2:
        raise ProbeTrainingError("baseline targets must have [N,...] shape")
    if train_targets.shape[1:] != evaluation_targets.shape[1:]:
        raise ProbeTrainingError("train/evaluation target shapes differ")
    if not (
        bool(train_targets.isfinite().all().item())
        and bool(evaluation_targets.isfinite().all().item())
    ):
        raise ProbeTrainingError("baseline targets contain NaN/Inf")
    if train_mask is None:
        mean = train_targets.float().mean(dim=0, keepdim=True)
    else:
        mask = train_mask
        while mask.ndim < train_targets.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(train_targets).float()
        counts = mask.sum(dim=0, keepdim=True)
        if bool((counts <= 0).any().item()):
            raise ProbeTrainingError(
                "target-mean baseline has a target dimension without valid labels"
            )
        mean = (train_targets.float() * mask).sum(dim=0, keepdim=True) / counts
    return {
        "constant_zero": torch.zeros_like(evaluation_targets),
        "target_mean": mean.expand_as(evaluation_targets).clone(),
    }
