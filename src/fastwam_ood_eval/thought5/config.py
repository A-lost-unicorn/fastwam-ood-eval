"""Strict, versioned configuration for Phase 5 audit/smoke/pilot/formal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from fastwam_ood_eval.thought5 import FORMAL_VARIANTS, VARIANTS
from fastwam_ood_eval.thought5.schemas import object_sha256


CONFIG_SCHEMA = "thought5.phase5.config.v1"
SELECTED_FEATURE = "mot.video_kv_cache.15.v"
INJECTION_MODULE = "video_expert.blocks.15.self_attn.v"
FROZEN_LORA_TARGETS = (
    "video_expert.blocks.15.self_attn.k",
    "video_expert.blocks.15.self_attn.v",
)


class Thought5ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    stage: str
    output_dir: Path
    seed: int
    protocol_frozen: bool


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    device: str
    dtype: str
    action_denoise_steps: int


@dataclass(frozen=True)
class BackboneConfig:
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_stats_path: Path
    dataset_stats_sha256: str
    fastwam_commit: str
    frozen_parameter_sha256: str
    selected_feature: str
    injection_module: str
    hidden_dim: int
    token_height: int
    token_width: int


@dataclass(frozen=True)
class MethodConfig:
    lora_targets: tuple[str, ...]
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    geo_projector_hidden_dim: int
    ray_pose_hidden_dim: int
    train_action_dit: bool
    use_gt_depth_at_inference: bool


@dataclass(frozen=True)
class CohortConfig:
    suite: str
    dataset_root: Path
    dataset_revision: str
    classification_path: Path
    train_tasks: tuple[int, ...]
    development_tasks: tuple[int, ...]
    formal_tasks: tuple[int, ...]
    train_episodes_per_task: int
    development_episodes_per_task: int
    formal_episodes_per_task: int
    frames_per_episode: int
    horizon: int
    conditions: tuple[str, ...]
    exact_state_conditions: tuple[str, ...]
    target_object_by_task: Mapping[int, str]
    split_seed: int
    seed_namespace: str
    thought3_split_manifest: Path
    thought4_formal_manifest: Path


@dataclass(frozen=True)
class TrainingConfig:
    variants: tuple[str, ...]
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    checkpoint_rule: str
    lambda_by_variant: Mapping[str, Mapping[str, float]]
    shuffled_geometry_seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_replicates: int
    bootstrap_seed: int
    task_bootstrap_seed: int
    h1_min_gap_reduction: float
    clean_noninferiority_margin: float
    g4_equivalence_fraction: float
    fixed_video_layer: int
    action_seeds: tuple[int, ...]
    future_k: int
    future_probe_model: str
    future_probe_projection_dim: int
    future_probe_projection_seed: int
    future_probe_ridge_alphas: tuple[float, ...]
    thought3_adapter_steps: int
    thought3_adapter_lr: float
    thought3_adapter_seed: int
    rollout_max_steps: int
    rollout_wait_steps: int
    rollout_control_horizon: int
    rollout_image_size: tuple[int, int]
    rollout_save_failure_videos: bool


@dataclass(frozen=True)
class Thought5Config:
    schema_version: str
    experiment: ExperimentConfig
    runtime: RuntimeConfig
    backbone: BackboneConfig
    method: MethodConfig
    cohort: CohortConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    raw: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return object_sha256(self.raw)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Thought5ConfigError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise Thought5ConfigError(
            f"{name} keys differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _path(value: Any) -> Path:
    return Path(str(value))


def load_thought5_config(path: str | Path) -> Thought5Config:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    _exact_keys(
        root,
        {
            "schema_version",
            "experiment",
            "runtime",
            "backbone",
            "method",
            "cohort",
            "training",
            "evaluation",
        },
        "config",
    )
    if root["schema_version"] != CONFIG_SCHEMA:
        raise Thought5ConfigError("unsupported Thought5 config schema")
    experiment = _mapping(root["experiment"], "experiment")
    runtime = _mapping(root["runtime"], "runtime")
    backbone = _mapping(root["backbone"], "backbone")
    method = _mapping(root["method"], "method")
    cohort = _mapping(root["cohort"], "cohort")
    training = _mapping(root["training"], "training")
    evaluation = _mapping(root["evaluation"], "evaluation")
    cfg = Thought5Config(
        schema_version=CONFIG_SCHEMA,
        experiment=ExperimentConfig(
            name=str(experiment["name"]),
            stage=str(experiment["stage"]),
            output_dir=_path(experiment["output_dir"]),
            seed=int(experiment["seed"]),
            protocol_frozen=bool(experiment["protocol_frozen"]),
        ),
        runtime=RuntimeConfig(
            backend=str(runtime["backend"]),
            device=str(runtime["device"]),
            dtype=str(runtime["dtype"]),
            action_denoise_steps=int(runtime["action_denoise_steps"]),
        ),
        backbone=BackboneConfig(
            checkpoint_path=_path(backbone["checkpoint_path"]),
            checkpoint_sha256=str(backbone["checkpoint_sha256"]),
            dataset_stats_path=_path(backbone["dataset_stats_path"]),
            dataset_stats_sha256=str(backbone["dataset_stats_sha256"]),
            fastwam_commit=str(backbone["fastwam_commit"]),
            frozen_parameter_sha256=str(backbone["frozen_parameter_sha256"]),
            selected_feature=str(backbone["selected_feature"]),
            injection_module=str(backbone["injection_module"]),
            hidden_dim=int(backbone["hidden_dim"]),
            token_height=int(backbone["token_height"]),
            token_width=int(backbone["token_width"]),
        ),
        method=MethodConfig(
            lora_targets=tuple(str(v) for v in method["lora_targets"]),
            lora_rank=int(method["lora_rank"]),
            lora_alpha=float(method["lora_alpha"]),
            lora_dropout=float(method["lora_dropout"]),
            geo_projector_hidden_dim=int(method["geo_projector_hidden_dim"]),
            ray_pose_hidden_dim=int(method["ray_pose_hidden_dim"]),
            train_action_dit=bool(method["train_action_dit"]),
            use_gt_depth_at_inference=bool(method["use_gt_depth_at_inference"]),
        ),
        cohort=CohortConfig(
            suite=str(cohort["suite"]),
            dataset_root=_path(cohort["dataset_root"]),
            dataset_revision=str(cohort["dataset_revision"]),
            classification_path=_path(cohort["classification_path"]),
            train_tasks=tuple(int(v) for v in cohort["train_tasks"]),
            development_tasks=tuple(int(v) for v in cohort["development_tasks"]),
            formal_tasks=tuple(int(v) for v in cohort["formal_tasks"]),
            train_episodes_per_task=int(cohort["train_episodes_per_task"]),
            development_episodes_per_task=int(
                cohort["development_episodes_per_task"]
            ),
            formal_episodes_per_task=int(cohort["formal_episodes_per_task"]),
            frames_per_episode=int(cohort["frames_per_episode"]),
            horizon=int(cohort["horizon"]),
            conditions=tuple(str(v) for v in cohort["conditions"]),
            exact_state_conditions=tuple(
                str(v) for v in cohort["exact_state_conditions"]
            ),
            target_object_by_task={
                int(key): str(value)
                for key, value in _mapping(
                    cohort["target_object_by_task"], "target_object_by_task"
                ).items()
            },
            split_seed=int(cohort["split_seed"]),
            seed_namespace=str(cohort["seed_namespace"]),
            thought3_split_manifest=_path(cohort["thought3_split_manifest"]),
            thought4_formal_manifest=_path(cohort["thought4_formal_manifest"]),
        ),
        training=TrainingConfig(
            variants=tuple(str(v) for v in training["variants"]),
            max_steps=int(training["max_steps"]),
            batch_size=int(training["batch_size"]),
            gradient_accumulation=int(training["gradient_accumulation"]),
            optimizer=str(training["optimizer"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            checkpoint_rule=str(training["checkpoint_rule"]),
            lambda_by_variant={
                str(key): {str(k): float(v) for k, v in value.items()}
                for key, value in _mapping(
                    training["lambda_by_variant"], "lambda_by_variant"
                ).items()
            },
            shuffled_geometry_seed=int(training["shuffled_geometry_seed"]),
        ),
        evaluation=EvaluationConfig(
            bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
            bootstrap_seed=int(evaluation["bootstrap_seed"]),
            task_bootstrap_seed=int(evaluation["task_bootstrap_seed"]),
            h1_min_gap_reduction=float(evaluation["h1_min_gap_reduction"]),
            clean_noninferiority_margin=float(
                evaluation["clean_noninferiority_margin"]
            ),
            g4_equivalence_fraction=float(
                evaluation["g4_equivalence_fraction"]
            ),
            fixed_video_layer=int(evaluation["fixed_video_layer"]),
            action_seeds=tuple(int(v) for v in evaluation["action_seeds"]),
            future_k=int(evaluation["future_k"]),
            future_probe_model=str(evaluation["future_probe_model"]),
            future_probe_projection_dim=int(
                evaluation["future_probe_projection_dim"]
            ),
            future_probe_projection_seed=int(
                evaluation["future_probe_projection_seed"]
            ),
            future_probe_ridge_alphas=tuple(
                float(value)
                for value in evaluation["future_probe_ridge_alphas"]
            ),
            thought3_adapter_steps=int(evaluation["thought3_adapter_steps"]),
            thought3_adapter_lr=float(evaluation["thought3_adapter_lr"]),
            thought3_adapter_seed=int(evaluation["thought3_adapter_seed"]),
            rollout_max_steps=int(evaluation["rollout_max_steps"]),
            rollout_wait_steps=int(evaluation["rollout_wait_steps"]),
            rollout_control_horizon=int(evaluation["rollout_control_horizon"]),
            rollout_image_size=tuple(
                int(value) for value in evaluation["rollout_image_size"]
            ),
            rollout_save_failure_videos=bool(
                evaluation["rollout_save_failure_videos"]
            ),
        ),
        raw=root,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Thought5Config) -> None:
    if cfg.experiment.stage not in {"audit", "smoke", "pilot", "formal"}:
        raise Thought5ConfigError("invalid stage")
    if cfg.runtime.backend not in {"mock", "fastwam"}:
        raise Thought5ConfigError("runtime backend must be mock or fastwam")
    if cfg.backbone.selected_feature != SELECTED_FEATURE:
        raise Thought5ConfigError("selected feature must remain frozen at layer-15 V")
    for label, digest in (
        ("checkpoint", cfg.backbone.checkpoint_sha256),
        ("dataset stats", cfg.backbone.dataset_stats_sha256),
        ("frozen parameters", cfg.backbone.frozen_parameter_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise Thought5ConfigError(f"{label} SHA-256 is invalid")
    if cfg.backbone.injection_module != INJECTION_MODULE:
        raise Thought5ConfigError("ray/pose injection module changed")
    if cfg.backbone.hidden_dim != 3072 or (
        cfg.backbone.token_height,
        cfg.backbone.token_width,
    ) != (7, 14):
        raise Thought5ConfigError("Fast-WAM layer-15 feature shape must be 98x3072")
    if cfg.method.lora_targets != FROZEN_LORA_TARGETS:
        raise Thought5ConfigError("LoRA targets differ from the preregistered K/V pair")
    if cfg.method.train_action_dit or cfg.method.use_gt_depth_at_inference:
        raise Thought5ConfigError("Action DiT/GT-depth inference are forbidden in v1")
    if cfg.method.lora_rank != 8 or cfg.method.lora_dropout != 0:
        raise Thought5ConfigError("v1 freezes LoRA rank=8 and dropout=0")
    tasks = (
        set(cfg.cohort.train_tasks),
        set(cfg.cohort.development_tasks),
        set(cfg.cohort.formal_tasks),
    )
    if any(not group for group in tasks):
        raise Thought5ConfigError("train/development/formal task lists must be non-empty")
    if cfg.experiment.stage in {"audit", "formal"} and any(
        left & right
        for index, left in enumerate(tasks)
        for right in tasks[index + 1 :]
    ):
        raise Thought5ConfigError("formal train/development/test tasks must be disjoint")
    if set().union(*tasks) - set(range(10)):
        raise Thought5ConfigError("LIBERO Goal task indices must be in [0,9]")
    if set().union(*tasks) - set(cfg.cohort.target_object_by_task):
        raise Thought5ConfigError("every selected task needs a target-object mapping")
    if cfg.cohort.conditions != ("clean", "camera", "lighting", "robot_init"):
        raise Thought5ConfigError("condition order is frozen")
    if cfg.cohort.exact_state_conditions != ("clean", "camera", "lighting"):
        raise Thought5ConfigError("exact-state conditions are frozen")
    if cfg.experiment.stage == "formal":
        if len(cfg.cohort.formal_tasks) < 2:
            raise Thought5ConfigError("formal evaluation requires multiple tasks")
        if cfg.training.variants != FORMAL_VARIANTS:
            raise Thought5ConfigError("formal controls must be B0/B1/G1/G2/G3")
    elif any(value not in VARIANTS for value in cfg.training.variants):
        raise Thought5ConfigError("unknown control variant")
    expected_lambdas = {"lambda_repa", "lambda_equiv", "lambda_pose_aux"}
    for variant in cfg.training.variants:
        values = cfg.training.lambda_by_variant.get(variant)
        if variant == "B0":
            continue
        if values is None or set(values) != expected_lambdas:
            raise Thought5ConfigError(f"missing exact lambda contract for {variant}")
        if any(value < 0 for value in values.values()):
            raise Thought5ConfigError("lambda values must be non-negative")
    if any(
        cfg.training.lambda_by_variant.get(variant, {}).get(key, 0) != 0
        for variant in ("B1",)
        for key in expected_lambdas
    ):
        raise Thought5ConfigError("B1 must have all auxiliary lambdas zero")
    if cfg.evaluation.fixed_video_layer != 15 or cfg.evaluation.future_k != 1:
        raise Thought5ConfigError("frozen evaluation layer/K changed")
    if (
        cfg.evaluation.future_probe_model != "linear_ridge"
        or cfg.evaluation.future_probe_projection_dim != 128
        or cfg.evaluation.future_probe_projection_seed != 5597
        or cfg.evaluation.future_probe_ridge_alphas
        != (0.0001, 0.01, 1.0, 100.0)
    ):
        raise Thought5ConfigError("future-geometry probe protocol changed")
    if (
        cfg.evaluation.thought3_adapter_steps != 200
        or cfg.evaluation.thought3_adapter_lr != 3e-4
        or cfg.evaluation.thought3_adapter_seed != 3407
    ):
        raise Thought5ConfigError("Thought3 adapter recipe must be reused unchanged")
    if cfg.evaluation.h1_min_gap_reduction != 0.25:
        raise Thought5ConfigError("H1 threshold must remain 25%")
    if cfg.evaluation.g4_equivalence_fraction != 0.8:
        raise Thought5ConfigError("G4 equivalence threshold must remain 80%")
    if (
        cfg.evaluation.rollout_max_steps != 400
        or cfg.evaluation.rollout_wait_steps != 30
        or cfg.evaluation.rollout_control_horizon != 10
        or cfg.evaluation.rollout_image_size != (256, 256)
    ):
        raise Thought5ConfigError(
            "v1 rollout semantics must remain 400/30/10 at 256x256"
        )


def config_summary(cfg: Thought5Config) -> dict[str, Any]:
    return {
        "schema_version": cfg.schema_version,
        "fingerprint": cfg.fingerprint,
        "experiment": asdict(cfg.experiment),
        "selected_feature": cfg.backbone.selected_feature,
        "lora_targets": list(cfg.method.lora_targets),
        "variants": list(cfg.training.variants),
        "task_split": {
            "train": list(cfg.cohort.train_tasks),
            "development": list(cfg.cohort.development_tasks),
            "formal": list(cfg.cohort.formal_tasks),
        },
    }
