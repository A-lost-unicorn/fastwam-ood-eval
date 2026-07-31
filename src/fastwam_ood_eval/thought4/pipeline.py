"""Probe-panel orchestration over already extracted frozen features."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought4.probe_evaluation import (
    depth_metrics,
    episode_grouped_bootstrap,
    paired_condition_gap,
    regression_metrics,
    rotation_metrics_6d,
    trajectory_metrics,
)
from fastwam_ood_eval.thought4.probe_training import (
    ProbeDataset,
    ProbeTrainingSpec,
    TrainedProbe,
    target_baselines,
    train_probe,
)
from fastwam_ood_eval.thought4.schemas import sha256_canonical
from fastwam_ood_eval.thought4.video_feature_extractor import tensor_sha256


class DiagnosisPipelineError(RuntimeError):
    """Raised when an extracted bundle cannot support the frozen probe panel."""


@dataclass(frozen=True)
class ProbeExample:
    sample_id: str
    episode_id: str
    split: str
    condition: str
    source: str
    module_path: str
    layer_index: int | None
    denoise_step_index: int | None
    pooling: str
    feature: Any
    labels: Mapping[str, Any]
    masks: Mapping[str, Any]

    @property
    def feature_key(self) -> str:
        layer = "none" if self.layer_index is None else str(self.layer_index)
        denoise = (
            "none"
            if self.denoise_step_index is None
            else str(self.denoise_step_index)
        )
        return (
            f"{self.source}|{self.module_path}|layer={layer}|"
            f"denoise={denoise}|pool={self.pooling}"
        )


@dataclass
class ProbePanelOutput:
    result: dict[str, Any]
    linear_models: dict[tuple[str, str, int], Any]


def _stack(
    examples: Sequence[ProbeExample],
    target_name: str,
    *,
    distinguish_condition: bool = False,
) -> ProbeDataset:
    import torch

    if not examples:
        raise DiagnosisPipelineError("cannot build an empty probe dataset")
    features = torch.stack(
        [torch.as_tensor(value.feature).detach().float().reshape(-1) for value in examples]
    )
    targets = torch.stack(
        [torch.as_tensor(value.labels[target_name]).detach().float() for value in examples]
    )
    mask_values = [value.masks.get(target_name) for value in examples]
    if any(value is not None for value in mask_values):
        if not all(value is not None for value in mask_values):
            raise DiagnosisPipelineError("target mask is present for only some samples")
        mask = torch.stack(
            [torch.as_tensor(value, dtype=torch.bool) for value in mask_values]
        )
    else:
        mask = None
    dataset = ProbeDataset(
        features=features.detach(),
        targets=targets.detach(),
        valid_mask=mask,
        episode_ids=tuple(value.episode_id for value in examples),
        sample_ids=tuple(
            f"{value.sample_id}:{value.condition}"
            if distinguish_condition
            else value.sample_id
            for value in examples
        ),
    )
    dataset.validate()
    return dataset


def _predict(trained: TrainedProbe, dataset: ProbeDataset) -> Any:
    import torch

    model = trained.model
    model.eval()
    device = next(model.parameters()).device
    feature_mean = trained.feature_mean.to(device)
    feature_std = trained.feature_std.to(device)
    target_mean = trained.target_mean.to(device)
    target_std = trained.target_std.to(device)
    with torch.no_grad():
        standardized = (
            dataset.features.detach().float().to(device) - feature_mean
        ) / feature_std
        prediction = model(standardized)
        prediction = prediction.float() * target_std + target_mean
    return prediction.cpu().reshape(dataset.targets.shape)


def _per_sample_rmse(prediction: Any, target: Any, mask: Any | None) -> list[float]:
    import torch

    delta = (prediction.detach().float() - target.detach().float()).square()
    if mask is None:
        return [
            float(value)
            for value in delta.reshape(delta.shape[0], -1).mean(dim=1).sqrt()
        ]
    valid = mask
    while valid.ndim < delta.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(delta).float()
    numerator = (delta * valid).reshape(delta.shape[0], -1).sum(dim=1)
    denominator = valid.reshape(delta.shape[0], -1).sum(dim=1).clamp_min(1)
    return [float(value) for value in (numerator / denominator).sqrt()]


def _target_metrics(
    target_name: str,
    prediction: Any,
    dataset: ProbeDataset,
) -> dict[str, float]:
    import torch

    mask = dataset.valid_mask
    if target_name == "depth":
        return depth_metrics(
            prediction.clamp_min(1e-6),
            dataset.targets.clamp_min(1e-6),
        )
    if target_name.endswith("_6d"):
        return rotation_metrics_6d(prediction, dataset.targets, mask)
    if target_name == "action_translation_trajectory":
        return trajectory_metrics(
            prediction,
            dataset.targets,
            mask,
        )
    if target_name == "action_se3_trajectory":
        # Layout [H,10] = xyz + rotation6d + gripper.
        if prediction.shape[-1] != 10 or dataset.targets.shape[-1] != 10:
            raise DiagnosisPipelineError("SE(3) target must have last dimension 10")
        return trajectory_metrics(
            prediction[..., :3],
            dataset.targets[..., :3],
            mask,
            rotation_prediction=prediction[..., 3:9],
            rotation_target=dataset.targets[..., 3:9],
            # The probe is trained as direct continuous regression to g∈[0,1],
            # not as a logit classifier.
            gripper_prediction=prediction[..., 9:10].clamp(0.0, 1.0),
            gripper_target=dataset.targets[..., 9:10],
        )
    return regression_metrics(prediction, dataset.targets, mask)


def _baseline_metrics(
    target_name: str,
    train: ProbeDataset,
    evaluation: ProbeDataset,
) -> dict[str, dict[str, float]]:
    return {
        name: _target_metrics(target_name, prediction, evaluation)
        for name, prediction in target_baselines(
            train.targets,
            evaluation.targets,
            train_mask=train.valid_mask,
        ).items()
    }


def run_probe_panel(
    examples: Sequence[ProbeExample],
    *,
    source: str,
    probe_models: Sequence[str],
    seeds: Sequence[int],
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    device: str = "cpu",
) -> ProbePanelOutput:
    """Train on Clean/train, tune on Clean/development, evaluate all test conditions."""

    if source not in {"A", "B"}:
        raise DiagnosisPipelineError("probe source must be A or B")
    selected = [value for value in examples if value.source == source]
    if not selected:
        raise DiagnosisPipelineError(f"no examples for source {source}")
    groups: dict[str, list[ProbeExample]] = {}
    for value in selected:
        groups.setdefault(value.feature_key, []).append(value)
    rows: list[dict[str, Any]] = []
    linear_models: dict[tuple[str, str, int], Any] = {}
    for feature_key, group in sorted(groups.items()):
        clean_train = [
            value
            for value in group
            if value.split == "train" and value.condition == "clean"
        ]
        clean_development = [
            value
            for value in group
            if value.split == "development" and value.condition == "clean"
        ]
        if not clean_train or not clean_development:
            raise DiagnosisPipelineError(
                f"{feature_key} lacks Clean train/development examples"
            )
        target_names = sorted(
            set.intersection(*(set(value.labels) for value in group))
        )
        if not target_names:
            raise DiagnosisPipelineError(f"{feature_key} has no shared labels")
        for target_name in target_names:
            pooling = group[0].pooling
            if (
                source == "A"
                and target_name == "depth"
                and pooling not in {"spatial_mean", "foreground_mean"}
            ):
                continue
            if (
                source == "A"
                and target_name.startswith("relative_camera_")
                and pooling != "spatial_mean"
            ):
                continue
            if (
                source == "A"
                and target_name.startswith("eef_object_")
                and pooling not in {"spatial_mean", "robot_object_roi"}
            ):
                continue
            camera_pose_target = target_name.startswith("relative_camera_")
            training_conditions = (
                {"clean", "camera"} if camera_pose_target else {"clean"}
            )
            train_examples = [
                value
                for value in group
                if value.split == "train"
                and value.condition in training_conditions
            ]
            development_examples = [
                value
                for value in group
                if value.split == "development"
                and value.condition in training_conditions
            ]
            train = _stack(
                train_examples,
                target_name,
                distinguish_condition=camera_pose_target,
            )
            development = _stack(
                development_examples,
                target_name,
                distinguish_condition=camera_pose_target,
            )
            evaluations = {
                condition: _stack(
                    [
                        value
                        for value in group
                        if value.split == "test" and value.condition == condition
                    ],
                    target_name,
                )
                for condition in ("clean", "camera", "lighting", "robot_init")
            }
            for kind in probe_models:
                for seed in seeds:
                    spec = ProbeTrainingSpec(
                        kind=kind,
                        task_type="regression",
                        hidden_dim=hidden_dim,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        max_epochs=max_epochs,
                        patience=patience,
                        batch_size=batch_size,
                        seed=int(seed),
                    )
                    trained = train_probe(
                        train,
                        development,
                        spec,
                        shuffled_label_control=False,
                        device=device,
                    )
                    shuffled = train_probe(
                        train,
                        development,
                        spec,
                        shuffled_label_control=True,
                        device=device,
                    )
                    development_prediction = _predict(trained, development)
                    development_metrics = _target_metrics(
                        target_name, development_prediction, development
                    )
                    condition_metrics: dict[str, Any] = {}
                    error_by_condition: dict[str, dict[str, float]] = {}
                    episode_by_condition: dict[str, dict[str, str]] = {}
                    for condition, dataset in evaluations.items():
                        prediction = _predict(trained, dataset)
                        sample_errors = _per_sample_rmse(
                            prediction, dataset.targets, dataset.valid_mask
                        )
                        error_by_condition[condition] = dict(
                            zip(dataset.sample_ids, sample_errors)
                        )
                        episode_by_condition[condition] = dict(
                            zip(dataset.sample_ids, dataset.episode_ids)
                        )
                        interval = episode_grouped_bootstrap(
                            sample_errors,
                            dataset.episode_ids,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + int(seed),
                        )
                        shuffled_prediction = _predict(shuffled, dataset)
                        condition_metrics[condition] = {
                            "metrics": _target_metrics(
                                target_name, prediction, dataset
                            ),
                            "rmse_grouped_bootstrap": asdict(interval),
                            "shuffled_label_control": _target_metrics(
                                target_name, shuffled_prediction, dataset
                            ),
                            "baselines": _baseline_metrics(
                                target_name, train, dataset
                            ),
                            "sample_count": int(dataset.features.shape[0]),
                            "episode_count": len(set(dataset.episode_ids)),
                        }
                    clean_rmse = condition_metrics["clean"][
                        "rmse_grouped_bootstrap"
                    ]["estimate"]
                    gaps = {
                        condition: (
                            condition_metrics[condition][
                                "rmse_grouped_bootstrap"
                            ]["estimate"]
                            - clean_rmse
                        )
                        for condition in ("camera", "lighting", "robot_init")
                    }
                    exact_state_paired_gaps = {
                        condition: asdict(
                            paired_condition_gap(
                                error_by_condition["clean"],
                                error_by_condition[condition],
                                episode_by_pair=episode_by_condition["clean"],
                                replicates=bootstrap_replicates,
                                seed=bootstrap_seed + int(seed) + offset,
                            )
                        )
                        for offset, condition in enumerate(
                            ("camera", "lighting"), start=1
                        )
                    }
                    metadata = group[0]
                    row = {
                        "source": source,
                        "feature_key": feature_key,
                        "module_path": metadata.module_path,
                        "layer_index": metadata.layer_index,
                        "denoise_step_index": metadata.denoise_step_index,
                        "pooling": metadata.pooling,
                        "target": target_name,
                        "training_conditions": sorted(training_conditions),
                        "probe_kind": kind,
                        "seed": int(seed),
                        "parameter_count": trained.parameter_count,
                        "best_epoch": trained.best_epoch,
                        "development_loss": trained.best_development_loss,
                        "standardization": {
                            "fit_split": "train",
                            "fit_conditions": sorted(training_conditions),
                            "feature_floor": 1e-6,
                            "target_floor": 1e-6,
                            "feature_mean_sha256": tensor_sha256(
                                trained.feature_mean
                            ),
                            "feature_std_sha256": tensor_sha256(
                                trained.feature_std
                            ),
                            "target_mean_sha256": tensor_sha256(
                                trained.target_mean
                            ),
                            "target_std_sha256": tensor_sha256(
                                trained.target_std
                            ),
                            "feature_constant_dimension_count": int(
                                (trained.feature_std <= 1e-6).sum().item()
                            ),
                            "target_constant_dimension_count": int(
                                (trained.target_std <= 1e-6).sum().item()
                            ),
                        },
                        "development_metrics": development_metrics,
                        "condition_metrics": condition_metrics,
                        "gaps_vs_clean_rmse": gaps,
                        "exact_state_paired_rmse_gaps": exact_state_paired_gaps,
                        "robot_init_exact_state_pair": False,
                        "test_used_for_selection": False,
                    }
                    row["row_sha256"] = sha256_canonical(row)
                    rows.append(row)
                    if kind == "linear":
                        linear_models[(feature_key, target_name, int(seed))] = (
                            trained
                        )
    payload: dict[str, Any] = {
        "schema_version": f"thought4.phase4.source_{source.lower()}_probe.v1",
        "source": source,
        "training_rule": {
            "relative_camera_pose": ["clean", "camera"],
            "all_other_targets": ["clean"],
        },
        "selection_split": "development",
        "evaluation_split": "test",
        "future_rgb_read": False,
        "rows": rows,
    }
    payload["result_sha256"] = sha256_canonical(payload)
    return ProbePanelOutput(payload, linear_models)


def select_intervention_feature(
    video_probe_result: Mapping[str, Any],
    *,
    target: str,
    seed: int,
) -> dict[str, Any]:
    """Select actual cached K/V by cross-seed development loss only."""

    probe_candidates = [
        row
        for row in video_probe_result.get("rows", [])
        if row.get("probe_kind") == "linear"
        and row.get("target") == target
        and row.get("source") == "A"
    ]
    if not probe_candidates:
        raise DiagnosisPipelineError("no eligible development linear Video probe")
    # Phase 4-C prioritizes the exact K/V tensors consumed by Action DiT.  The
    # upstream norm1 hidden is probed and reported, but cannot win intervention
    # selection merely because it has a lower development loss.
    candidates = [
        row
        for row in probe_candidates
        if str(row.get("module_path", "")).startswith("mot.video_kv_cache.")
        and str(row.get("module_path", "")).endswith((".k", ".v"))
    ]
    if not candidates:
        raise DiagnosisPipelineError(
            "no actual action-consumed Video K/V cache candidate for intervention"
        )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(str(row["feature_key"]), []).append(row)
    expected_seeds = {int(row["seed"]) for row in candidates}
    if int(seed) not in expected_seeds:
        raise DiagnosisPipelineError(
            "requested intervention basis seed is absent from probe panel"
        )
    summaries: list[
        tuple[float, int, str, str, list[Mapping[str, Any]]]
    ] = []
    for feature_key, values in grouped.items():
        observed_seeds = [int(row["seed"]) for row in values]
        if len(observed_seeds) != len(set(observed_seeds)):
            raise DiagnosisPipelineError(
                f"duplicate linear probe seed for {feature_key}"
            )
        if set(observed_seeds) != expected_seeds:
            raise DiagnosisPipelineError(
                f"incomplete probe seed panel for {feature_key}"
            )
        mean_loss = sum(float(row["development_loss"]) for row in values) / len(
            values
        )
        metadata = values[0]
        summaries.append(
            (
                mean_loss,
                -int(
                    metadata["layer_index"]
                    if metadata["layer_index"] is not None
                    else -1
                ),
                str(metadata["module_path"]),
                feature_key,
                values,
            )
        )
    # Lower mean development loss is better. Test/OOD metrics are not read.
    mean_loss, _layer_order, _module_order, _feature_order, selected_rows = min(
        summaries, key=lambda value: value[:4]
    )
    selected = next(
        row for row in selected_rows if int(row["seed"]) == int(seed)
    )
    return {
        "feature_key": selected["feature_key"],
        "module_path": selected["module_path"],
        "layer_index": selected["layer_index"],
        "pooling": selected["pooling"],
        "target": selected["target"],
        "basis_seed": int(seed),
        "selection_seeds": sorted(expected_seeds),
        "selection_seed_count": len(expected_seeds),
        "selection_split": "development",
        "candidate_scope": "action_consumed_video_kv_only",
        "test_or_ood_read": False,
        "tie_break": "later_layer_then_module_path_then_feature_key",
        "development_loss_mean_across_seeds": mean_loss,
        "basis_seed_development_loss": selected["development_loss"],
    }
