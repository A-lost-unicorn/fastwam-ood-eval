"""Strict configuration and frozen-protocol validation for Thought4."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fastwam_ood_eval.thought4 import ALLOWED_CONDITIONS, THOUGHT4_CONFIG_SCHEMA
from fastwam_ood_eval.thought4.schemas import sha256_canonical


class Thought4ConfigError(ValueError):
    """Raised for an unsafe or ambiguous Thought4 configuration."""


OFFICIAL_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
OFFICIAL_CHECKPOINT_SHA256 = (
    "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
)
OFFICIAL_DATASET_REVISION = "117413dc0ca99c7cd64036c4eaa4a316c537d692"
FROZEN_VIDEO_LAYERS = (0, 7, 15, 22, 29)
FROZEN_ACTION_HOOKS = (
    "action_expert.action_encoder",
    "action_expert.blocks.15.norm1",
    "action_expert.blocks.29.norm1",
    "action_expert.head",
)
FROZEN_POOLING = (
    "spatial_mean",
    "foreground_mean",
    "robot_object_roi",
)
FORMAL_CONDITION_TASK_IDS = {
    "clean": (1,),
    "camera": (691, 697, 698, 706, 711),
    "lighting": (2313, 2314, 2334, 2337, 2351),
    "robot_init": (282, 283, 284, 285, 294),
}
SMOKE_CONDITION_TASK_IDS = {
    "clean": (1,),
    "camera": (691,),
    "lighting": (2313,),
    "robot_init": (282,),
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int
    mode: str


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    device: str
    action_denoise_steps: int
    feature_dtype: str
    shard_size: int


@dataclass(frozen=True)
class BackboneConfig:
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_stats_path: Path
    fastwam_commit: str
    frozen_parameter_sha256: str
    video_layers: tuple[int, ...]
    action_hooks: tuple[str, ...]


@dataclass(frozen=True)
class CohortConfig:
    suite: str
    task_ids: tuple[int, ...]
    dataset_root: Path
    dataset_revision: str
    condition_task_ids: tuple[tuple[str, tuple[int, ...]], ...]
    target_object_name: str
    split_seed: int
    train_base_states: int
    development_base_states: int
    test_base_states: int
    frames_per_episode: int
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class RenderingConfig:
    camera_name: str
    image_height: int
    image_width: int
    require_depth: bool
    exact_state_conditions: tuple[str, ...]


@dataclass(frozen=True)
class ProbeConfig:
    models: tuple[str, ...]
    pooling: tuple[str, ...]
    mlp_hidden_dim: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    batch_size: int
    seeds: tuple[int, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    horizon: int


@dataclass(frozen=True)
class InterventionConfig:
    enabled: bool
    source_preference: str
    target_label: str
    layer_selection_split: str
    rank_energy_threshold: float
    max_rank: int
    donor_seed: int
    action_seeds: tuple[int, ...]
    norm_ratio_tolerance: float
    replay_floor_repeats: int


@dataclass(frozen=True)
class Thought4Config:
    schema_version: str
    experiment: ExperimentConfig
    runtime: RuntimeConfig
    backbone: BackboneConfig
    cohort: CohortConfig
    rendering: RenderingConfig
    probe: ProbeConfig
    intervention: InterventionConfig

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        for section in ("experiment", "backbone", "cohort"):
            for key, value in list(payload[section].items()):
                if isinstance(value, Path):
                    payload[section][key] = str(value)
        return sha256_canonical(payload)


DEFAULTS: dict[str, Any] = {
    "schema_version": THOUGHT4_CONFIG_SCHEMA,
    "experiment": {
        "name": "phase4_geometry_action_diagnosis_v1",
        "output_dir": "outputs/thought4/phase4_geometry_action_diagnosis_v1",
        "seed": 4407,
        "mode": "formal",
    },
    "runtime": {
        "backend": "fastwam",
        "device": "cuda:0",
        "action_denoise_steps": 20,
        "feature_dtype": "float32",
        "shard_size": 16,
    },
    "backbone": {
        "checkpoint_path": "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "dataset_stats_path": (
            "checkpoints/fastwam_release/"
            "libero_uncond_2cam224_dataset_stats.json"
        ),
        "fastwam_commit": OFFICIAL_FASTWAM_COMMIT,
        "frozen_parameter_sha256": (
            "ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8"
        ),
        "video_layers": list(FROZEN_VIDEO_LAYERS),
        "action_hooks": list(FROZEN_ACTION_HOOKS),
    },
    "cohort": {
        "suite": "libero_goal",
        "task_ids": [0],
        "dataset_root": "data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        # LIBERO-Plus task-classification IDs are one-based.  Runtime adapters
        # subtract one before indexing the suite.
        "condition_task_ids": {
            "clean": [1],
            "camera": [691, 697, 698, 706, 711],
            "lighting": [2313, 2314, 2334, 2337, 2351],
            "robot_init": [282, 283, 284, 285, 294],
        },
        "target_object_name": "wooden_cabinet_1",
        "split_seed": 4407,
        "train_base_states": 40,
        "development_base_states": 12,
        "test_base_states": 12,
        "frames_per_episode": 2,
        "conditions": list(ALLOWED_CONDITIONS),
    },
    "rendering": {
        "camera_name": "agentview",
        "image_height": 224,
        "image_width": 224,
        "require_depth": True,
        "exact_state_conditions": ["clean", "camera", "lighting"],
    },
    "probe": {
        "models": ["linear", "mlp"],
        "pooling": ["spatial_mean", "foreground_mean", "robot_object_roi"],
        "mlp_hidden_dim": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "patience": 12,
        "batch_size": 32,
        "seeds": [4407, 4408, 4409],
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 4417,
        "horizon": 32,
    },
    "intervention": {
        "enabled": True,
        "source_preference": "A",
        "target_label": "eef_object_translation_camera",
        "layer_selection_split": "development",
        "rank_energy_threshold": 0.95,
        "max_rank": 32,
        "donor_seed": 4427,
        "action_seeds": [4437, 4438, 4439],
        "norm_ratio_tolerance": 0.05,
        "replay_floor_repeats": 2,
    },
}

_SECTION_KEYS = {
    key: set(value)
    for key, value in DEFAULTS.items()
    if isinstance(value, Mapping)
}


def _merge_strict(base: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    unknown_top = set(incoming) - set(base)
    if unknown_top:
        raise Thought4ConfigError(f"unknown top-level keys: {sorted(unknown_top)}")
    merged = deepcopy(base)
    for section, value in incoming.items():
        if section == "schema_version":
            merged[section] = value
            continue
        if not isinstance(value, Mapping):
            raise Thought4ConfigError(f"{section} must be a mapping")
        unknown = set(value) - _SECTION_KEYS[section]
        if unknown:
            raise Thought4ConfigError(
                f"unknown keys in {section}: {sorted(unknown)}"
            )
        merged[section].update(value)
    return merged


def _tuple_strings(value: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise Thought4ConfigError(f"{name} must be a non-empty list of strings")
    return result


def _as_config(data: Mapping[str, Any]) -> Thought4Config:
    merged = _merge_strict(DEFAULTS, data)
    cfg = Thought4Config(
        schema_version=str(merged["schema_version"]),
        experiment=ExperimentConfig(
            name=str(merged["experiment"]["name"]),
            output_dir=Path(merged["experiment"]["output_dir"]),
            seed=int(merged["experiment"]["seed"]),
            mode=str(merged["experiment"]["mode"]),
        ),
        runtime=RuntimeConfig(
            backend=str(merged["runtime"]["backend"]),
            device=str(merged["runtime"]["device"]),
            action_denoise_steps=int(merged["runtime"]["action_denoise_steps"]),
            feature_dtype=str(merged["runtime"]["feature_dtype"]),
            shard_size=int(merged["runtime"]["shard_size"]),
        ),
        backbone=BackboneConfig(
            checkpoint_path=Path(merged["backbone"]["checkpoint_path"]),
            checkpoint_sha256=str(merged["backbone"]["checkpoint_sha256"]),
            dataset_stats_path=Path(merged["backbone"]["dataset_stats_path"]),
            fastwam_commit=str(merged["backbone"]["fastwam_commit"]),
            frozen_parameter_sha256=str(
                merged["backbone"]["frozen_parameter_sha256"]
            ),
            video_layers=tuple(int(v) for v in merged["backbone"]["video_layers"]),
            action_hooks=_tuple_strings(
                merged["backbone"]["action_hooks"], "backbone.action_hooks"
            ),
        ),
        cohort=CohortConfig(
            suite=str(merged["cohort"]["suite"]),
            task_ids=tuple(int(v) for v in merged["cohort"]["task_ids"]),
            dataset_root=Path(merged["cohort"]["dataset_root"]),
            dataset_revision=str(merged["cohort"]["dataset_revision"]),
            condition_task_ids=tuple(
                sorted(
                    (
                        str(key),
                        tuple(int(item) for item in value),
                    )
                    for key, value in merged["cohort"][
                        "condition_task_ids"
                    ].items()
                )
            ),
            target_object_name=str(merged["cohort"]["target_object_name"]),
            split_seed=int(merged["cohort"]["split_seed"]),
            train_base_states=int(merged["cohort"]["train_base_states"]),
            development_base_states=int(
                merged["cohort"]["development_base_states"]
            ),
            test_base_states=int(merged["cohort"]["test_base_states"]),
            frames_per_episode=int(merged["cohort"]["frames_per_episode"]),
            conditions=_tuple_strings(
                merged["cohort"]["conditions"], "cohort.conditions"
            ),
        ),
        rendering=RenderingConfig(
            camera_name=str(merged["rendering"]["camera_name"]),
            image_height=int(merged["rendering"]["image_height"]),
            image_width=int(merged["rendering"]["image_width"]),
            require_depth=bool(merged["rendering"]["require_depth"]),
            exact_state_conditions=_tuple_strings(
                merged["rendering"]["exact_state_conditions"],
                "rendering.exact_state_conditions",
            ),
        ),
        probe=ProbeConfig(
            models=_tuple_strings(merged["probe"]["models"], "probe.models"),
            pooling=_tuple_strings(merged["probe"]["pooling"], "probe.pooling"),
            mlp_hidden_dim=int(merged["probe"]["mlp_hidden_dim"]),
            learning_rate=float(merged["probe"]["learning_rate"]),
            weight_decay=float(merged["probe"]["weight_decay"]),
            max_epochs=int(merged["probe"]["max_epochs"]),
            patience=int(merged["probe"]["patience"]),
            batch_size=int(merged["probe"]["batch_size"]),
            seeds=tuple(int(v) for v in merged["probe"]["seeds"]),
            bootstrap_replicates=int(merged["probe"]["bootstrap_replicates"]),
            bootstrap_seed=int(merged["probe"]["bootstrap_seed"]),
            horizon=int(merged["probe"]["horizon"]),
        ),
        intervention=InterventionConfig(
            enabled=bool(merged["intervention"]["enabled"]),
            source_preference=str(merged["intervention"]["source_preference"]),
            target_label=str(merged["intervention"]["target_label"]),
            layer_selection_split=str(
                merged["intervention"]["layer_selection_split"]
            ),
            rank_energy_threshold=float(
                merged["intervention"]["rank_energy_threshold"]
            ),
            max_rank=int(merged["intervention"]["max_rank"]),
            donor_seed=int(merged["intervention"]["donor_seed"]),
            action_seeds=tuple(
                int(v) for v in merged["intervention"]["action_seeds"]
            ),
            norm_ratio_tolerance=float(
                merged["intervention"]["norm_ratio_tolerance"]
            ),
            replay_floor_repeats=int(
                merged["intervention"]["replay_floor_repeats"]
            ),
        ),
    )
    validate_config(cfg)
    return cfg


def load_thought4_config(path: str | Path) -> Thought4Config:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Thought4ConfigError("Thought4 YAML root must be a mapping")
    return _as_config(value)


def validate_config(cfg: Thought4Config) -> None:
    if cfg.schema_version != THOUGHT4_CONFIG_SCHEMA:
        raise Thought4ConfigError(
            f"schema_version must be {THOUGHT4_CONFIG_SCHEMA}"
        )
    if cfg.experiment.mode not in {"smoke", "formal"}:
        raise Thought4ConfigError("experiment.mode must be smoke or formal")
    if cfg.experiment.seed < 0:
        raise Thought4ConfigError("experiment.seed must be non-negative")
    output = cfg.experiment.output_dir.resolve()
    allowed = Path("outputs/thought4").resolve()
    if output == allowed or allowed not in output.parents:
        raise Thought4ConfigError(
            "Thought4 output_dir must be a child of outputs/thought4/"
        )
    if cfg.runtime.backend not in {"mock", "fastwam"}:
        raise Thought4ConfigError("runtime.backend must be mock or fastwam")
    if cfg.runtime.backend == "fastwam" and not cfg.runtime.device.startswith("cuda:"):
        raise Thought4ConfigError("Fast-WAM Thought4 runs require explicit cuda:N")
    if cfg.runtime.action_denoise_steps != 20:
        raise Thought4ConfigError("action denoise schedule is frozen to 20 steps")
    if cfg.runtime.feature_dtype != "float32":
        raise Thought4ConfigError("pooled feature dtype is frozen to float32")
    if cfg.runtime.shard_size <= 0:
        raise Thought4ConfigError("runtime.shard_size must be positive")
    if cfg.backbone.fastwam_commit != OFFICIAL_FASTWAM_COMMIT:
        raise Thought4ConfigError("Fast-WAM commit differs from frozen official commit")
    if cfg.backbone.checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise Thought4ConfigError("checkpoint SHA differs from frozen official checkpoint")
    if cfg.experiment.mode == "formal":
        if cfg.backbone.video_layers != FROZEN_VIDEO_LAYERS:
            raise Thought4ConfigError(
                f"formal video layers must remain frozen to {FROZEN_VIDEO_LAYERS}"
            )
        if cfg.backbone.action_hooks != FROZEN_ACTION_HOOKS:
            raise Thought4ConfigError(
                "formal action hook paths differ from frozen protocol"
            )
    else:
        if cfg.backbone.video_layers != (15,):
            raise Thought4ConfigError("smoke must use exactly Video layer 15")
        if cfg.backbone.action_hooks != ("action_expert.blocks.15.norm1",):
            raise Thought4ConfigError(
                "smoke must use exactly Action block 15 norm1"
            )
    if not cfg.cohort.task_ids or min(cfg.cohort.task_ids) < 0:
        raise Thought4ConfigError("cohort.task_ids must be non-empty/non-negative")
    if cfg.cohort.suite != "libero_goal" or cfg.cohort.task_ids != (0,):
        raise Thought4ConfigError("v1 cohort is frozen to libero_goal task 0")
    if cfg.cohort.dataset_revision != OFFICIAL_DATASET_REVISION:
        raise Thought4ConfigError("dataset revision differs from frozen LIBERO source")
    condition_task_ids = dict(cfg.cohort.condition_task_ids)
    if set(condition_task_ids) != set(ALLOWED_CONDITIONS):
        raise Thought4ConfigError(
            "condition_task_ids must contain clean/camera/lighting/robot_init"
        )
    if any(not values for values in condition_task_ids.values()):
        raise Thought4ConfigError("each condition task-ID panel must be non-empty")
    if min(value for values in condition_task_ids.values() for value in values) <= 0:
        raise Thought4ConfigError(
            "LIBERO-Plus classification IDs must be one-based positive integers"
        )
    if len(condition_task_ids["clean"]) != 1:
        raise Thought4ConfigError("clean must contain exactly one base task ID")
    expected_task_ids = (
        FORMAL_CONDITION_TASK_IDS
        if cfg.experiment.mode == "formal"
        else SMOKE_CONDITION_TASK_IDS
    )
    if condition_task_ids != expected_task_ids:
        raise Thought4ConfigError(
            f"{cfg.experiment.mode} condition task-ID panel differs from frozen v1"
        )
    if cfg.cohort.target_object_name != "wooden_cabinet_1":
        raise Thought4ConfigError("v1 target object must be wooden_cabinet_1")
    expected_conditions = (
        set(ALLOWED_CONDITIONS)
        if cfg.experiment.mode == "formal"
        else {"clean", "camera", "lighting"}
    )
    if set(cfg.cohort.conditions) != expected_conditions:
        raise Thought4ConfigError(
            f"{cfg.experiment.mode} conditions must be {sorted(expected_conditions)}"
        )
    counts = (
        cfg.cohort.train_base_states,
        cfg.cohort.development_base_states,
        cfg.cohort.test_base_states,
    )
    if min(counts) <= 0 or cfg.cohort.frames_per_episode <= 0:
        raise Thought4ConfigError("cohort counts and frames_per_episode must be positive")
    expected_counts = (40, 12, 12) if cfg.experiment.mode == "formal" else (2, 2, 2)
    expected_frames = 2 if cfg.experiment.mode == "formal" else 1
    if counts != expected_counts or cfg.cohort.frames_per_episode != expected_frames:
        raise Thought4ConfigError(
            f"{cfg.experiment.mode} cohort size/frame plan differs from frozen v1"
        )
    if set(cfg.rendering.exact_state_conditions) != {
        "clean",
        "camera",
        "lighting",
    }:
        raise Thought4ConfigError(
            "exact_state_conditions must be clean/camera/lighting only"
        )
    if cfg.rendering.image_height <= 0 or cfg.rendering.image_width <= 0:
        raise Thought4ConfigError("render dimensions must be positive")
    if (
        cfg.rendering.camera_name != "agentview"
        or (cfg.rendering.image_height, cfg.rendering.image_width) != (224, 224)
        or not cfg.rendering.require_depth
    ):
        raise Thought4ConfigError("v1 rendering is frozen to agentview RGB-D 224x224")
    if cfg.probe.models != ("linear", "mlp"):
        raise Thought4ConfigError("probe order is frozen to linear then mlp")
    if cfg.probe.pooling != FROZEN_POOLING:
        raise Thought4ConfigError("probe pooling rules/order differ from frozen v1")
    if (
        not cfg.probe.seeds
        or min(cfg.probe.seeds) < 0
        or len(set(cfg.probe.seeds)) != len(cfg.probe.seeds)
    ):
        raise Thought4ConfigError("probe seeds must be non-negative")
    if min(
        cfg.probe.mlp_hidden_dim,
        cfg.probe.max_epochs,
        cfg.probe.patience,
        cfg.probe.batch_size,
        cfg.probe.bootstrap_replicates,
        cfg.probe.horizon,
    ) <= 0:
        raise Thought4ConfigError("probe integer hyperparameters must be positive")
    if cfg.probe.horizon != 32:
        raise Thought4ConfigError("probe/action horizon is frozen to 32")
    if cfg.probe.learning_rate <= 0 or cfg.probe.weight_decay < 0:
        raise Thought4ConfigError("invalid probe optimizer hyperparameters")
    if not cfg.intervention.enabled or cfg.intervention.source_preference != "A":
        raise Thought4ConfigError("v1 requires one source-A intervention")
    if cfg.intervention.target_label not in {
        "depth",
        "relative_camera_translation",
        "relative_camera_rotation_6d",
        "eef_object_translation_camera",
        "eef_object_translation_world",
    }:
        raise Thought4ConfigError("invalid intervention geometry target label")
    if cfg.intervention.target_label != "eef_object_translation_camera":
        raise Thought4ConfigError(
            "v1 intervention target is frozen to camera-frame EEF-object translation"
        )
    if cfg.intervention.layer_selection_split != "development":
        raise Thought4ConfigError("intervention layer selection must use development")
    if not 0.0 < cfg.intervention.rank_energy_threshold <= 1.0:
        raise Thought4ConfigError("rank_energy_threshold must be in (0,1]")
    if cfg.intervention.max_rank <= 0:
        raise Thought4ConfigError("intervention.max_rank must be positive")
    if cfg.intervention.norm_ratio_tolerance < 0:
        raise Thought4ConfigError("norm ratio tolerance must be non-negative")
    if (
        not cfg.intervention.action_seeds
        or min(cfg.intervention.action_seeds) < 0
        or len(set(cfg.intervention.action_seeds))
        != len(cfg.intervention.action_seeds)
    ):
        raise Thought4ConfigError(
            "intervention action seeds must be unique/non-negative"
        )
    if cfg.intervention.replay_floor_repeats < 2:
        raise Thought4ConfigError("replay_floor_repeats must be at least two")


def config_to_dict(cfg: Thought4Config) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["experiment"]["output_dir"] = str(cfg.experiment.output_dir)
    payload["backbone"]["checkpoint_path"] = str(cfg.backbone.checkpoint_path)
    payload["backbone"]["dataset_stats_path"] = str(
        cfg.backbone.dataset_stats_path
    )
    payload["cohort"]["dataset_root"] = str(cfg.cohort.dataset_root)
    payload["cohort"]["condition_task_ids"] = {
        key: list(values) for key, values in cfg.cohort.condition_task_ids
    }
    payload["config_fingerprint"] = cfg.fingerprint
    return payload
