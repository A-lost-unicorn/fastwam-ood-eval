"""Typed YAML configuration loading and validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


ALLOWED_BACKENDS = {"libero", "libero_plus", "mock"}
ALLOWED_PRECISIONS = {"fp32", "fp16", "bf16"}
ALLOWED_LEVELS = {"easy", "medium", "hard"}
ALLOWED_CATEGORIES = {
    "camera_viewpoints",
    "light_conditions",
    "background_textures",
    "robot_initial_states",
    "objects_layout",
}
ALLOWED_POLICY_VARIANTS = {"fastwam", "joint_wam", "idm", "mock"}
ALLOWED_VARIANT_SELECTIONS = {"sample", "all_once"}


@dataclass(frozen=True)
class PolicyConfig:
    """Policy identity needed for scientifically valid cross-model comparisons.

    ``test_time_future_imagination`` is metadata about the upstream architecture,
    not a cosmetic video-recording switch. Fast-WAM uses current-frame video
    tokens, Joint WAM jointly denoises future-video/action tokens, and IDM first
    predicts a future video before recovering actions.
    """

    variant: str = "fastwam"
    test_time_future_imagination: bool = False
    comparison_group: str | None = None
    training_recipe_id: str | None = None


def _required(data: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in data:
        raise ConfigError(f"Missing required setting: {section}.{key}")
    return data[key]


def _as_list(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    raise ConfigError(f"{name} must be a scalar or list, got {type(value).__name__}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int = 0
    overwrite: bool = False
    resume: bool = True
    save_video: bool = True
    save_failure_video_only: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class HardwareConfig:
    devices: tuple[int, ...] = (0,)
    workers_per_gpu: int = 1
    precision: str = "bf16"
    max_gpu_memory_gb: float = 23.0
    enable_tf32: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    path: Path | None
    model_name: str
    config_path: Path | None
    dataset_stats_path: Path | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    suite: str
    suite_config: Path
    tasks: tuple[int, ...] | None
    episodes_per_task: int
    max_steps: int
    num_steps_wait: int
    control_horizon: int
    image_size: tuple[int, int]


@dataclass(frozen=True)
class PerturbationConfig:
    enabled: bool = False
    categories: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()
    variant_selection: str = "sample"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingConfig:
    fps: int = 24
    save_observations: bool = False
    save_actions: bool = True
    save_robot_state: bool = True
    video_format: str = "mp4"


@dataclass(frozen=True)
class EvalConfig:
    experiment: ExperimentConfig
    hardware: HardwareConfig
    checkpoint: CheckpointConfig
    benchmark: BenchmarkConfig
    perturbation: PerturbationConfig
    recording: RecordingConfig
    source_path: Path
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                **self.experiment.__dict__,
                "output_dir": str(self.experiment.output_dir),
            },
            "hardware": {**self.hardware.__dict__, "devices": list(self.hardware.devices)},
            "checkpoint": {
                **self.checkpoint.__dict__,
                "path": str(self.checkpoint.path) if self.checkpoint.path else None,
                "config_path": str(self.checkpoint.config_path) if self.checkpoint.config_path else None,
                "dataset_stats_path": (
                    str(self.checkpoint.dataset_stats_path)
                    if self.checkpoint.dataset_stats_path
                    else None
                ),
            },
            "benchmark": {
                **self.benchmark.__dict__,
                "suite_config": str(self.benchmark.suite_config),
                "tasks": list(self.benchmark.tasks) if self.benchmark.tasks is not None else "all",
                "image_size": list(self.benchmark.image_size),
            },
            "perturbation": {
                "enabled": self.perturbation.enabled,
                "category": list(self.perturbation.categories),
                "level": list(self.perturbation.levels),
                "variant_selection": self.perturbation.variant_selection,
                "parameters": self.perturbation.parameters,
            },
            "recording": self.recording.__dict__,
            "policy": self.policy.__dict__,
        }


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid override value {raw!r}: {exc}") from exc


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply Hydra-like ``section.key=value`` overrides to a mapping."""
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must use key=value syntax: {item!r}")
        dotted, raw = item.split("=", 1)
        keys = [part for part in dotted.split(".") if part]
        if not keys:
            raise ConfigError(f"Override has an empty key: {item!r}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            child = cursor.get(key)
            if child is None:
                child = {}
                cursor[key] = child
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot override below non-mapping key: {dotted!r}")
            cursor = child
        cursor[keys[-1]] = _parse_override_value(raw)
    return result


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _build(data: Mapping[str, Any], source_path: Path) -> EvalConfig:
    for section in ("experiment", "hardware", "checkpoint", "benchmark", "perturbation", "recording"):
        if not isinstance(data.get(section), Mapping):
            raise ConfigError(f"Missing or invalid top-level section: {section}")

    ex = data["experiment"]
    hw = data["hardware"]
    ck = data["checkpoint"]
    bm = data["benchmark"]
    pt = data["perturbation"]
    rec = data["recording"]
    policy = data.get("policy", {})
    if not isinstance(policy, Mapping):
        raise ConfigError("Invalid top-level section: policy")

    tasks_raw = _required(bm, "tasks", "benchmark")
    if tasks_raw in (None, "all"):
        tasks = None
    else:
        task_values = _as_list(tasks_raw, name="benchmark.tasks")
        try:
            tasks = tuple(int(value) for value in task_values)
        except (TypeError, ValueError) as exc:
            raise ConfigError("benchmark.tasks must contain integer task IDs") from exc

    image_size_raw = _as_list(_required(bm, "image_size", "benchmark"), name="benchmark.image_size")
    if len(image_size_raw) != 2:
        raise ConfigError("benchmark.image_size must be [height, width]")

    cfg = EvalConfig(
        experiment=ExperimentConfig(
            name=str(_required(ex, "name", "experiment")),
            output_dir=Path(str(_required(ex, "output_dir", "experiment"))),
            seed=int(ex.get("seed", 0)),
            overwrite=bool(ex.get("overwrite", False)),
            resume=bool(ex.get("resume", True)),
            save_video=bool(ex.get("save_video", True)),
            save_failure_video_only=bool(ex.get("save_failure_video_only", True)),
            log_level=str(ex.get("log_level", "INFO")).upper(),
        ),
        hardware=HardwareConfig(
            devices=tuple(
                int(value)
                for value in _as_list(hw.get("devices", [0]), name="hardware.devices")
            ),
            workers_per_gpu=int(hw.get("workers_per_gpu", 1)),
            precision=str(hw.get("precision", "bf16")).lower(),
            max_gpu_memory_gb=float(hw.get("max_gpu_memory_gb", 23.0)),
            enable_tf32=bool(hw.get("enable_tf32", True)),
        ),
        checkpoint=CheckpointConfig(
            path=_path_or_none(ck.get("path")),
            model_name=str(ck.get("model_name", "libero_uncond_2cam224_1e-4")),
            config_path=_path_or_none(ck.get("config_path")),
            dataset_stats_path=_path_or_none(ck.get("dataset_stats_path")),
        ),
        benchmark=BenchmarkConfig(
            backend=str(_required(bm, "backend", "benchmark")).lower(),
            suite=str(_required(bm, "suite", "benchmark")),
            suite_config=Path(str(_required(bm, "suite_config", "benchmark"))),
            tasks=tasks,
            episodes_per_task=int(_required(bm, "episodes_per_task", "benchmark")),
            max_steps=int(_required(bm, "max_steps", "benchmark")),
            num_steps_wait=int(bm.get("num_steps_wait", 30)),
            control_horizon=int(_required(bm, "control_horizon", "benchmark")),
            image_size=(int(image_size_raw[0]), int(image_size_raw[1])),
        ),
        perturbation=PerturbationConfig(
            enabled=bool(pt.get("enabled", False)),
            categories=tuple(
                str(value)
                for value in _as_list(pt.get("category", []), name="perturbation.category")
            ),
            levels=tuple(
                str(value)
                for value in _as_list(pt.get("level", []), name="perturbation.level")
            ),
            variant_selection=str(pt.get("variant_selection", "sample")).lower(),
            parameters=dict(pt.get("parameters", {})),
        ),
        recording=RecordingConfig(
            fps=int(rec.get("fps", 24)),
            save_observations=bool(rec.get("save_observations", False)),
            save_actions=bool(rec.get("save_actions", True)),
            save_robot_state=bool(rec.get("save_robot_state", True)),
            video_format=str(rec.get("video_format", "mp4")),
        ),
        source_path=source_path,
        policy=PolicyConfig(
            variant=str(
                policy.get(
                    "variant",
                    "mock" if str(bm.get("backend", "")).lower() == "mock" else "fastwam",
                )
            ).lower(),
            test_time_future_imagination=bool(policy.get("test_time_future_imagination", False)),
            comparison_group=(
                str(policy["comparison_group"]).strip()
                if policy.get("comparison_group") not in (None, "")
                else None
            ),
            training_recipe_id=(
                str(policy["training_recipe_id"]).strip()
                if policy.get("training_recipe_id") not in (None, "")
                else None
            ),
        ),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: EvalConfig) -> None:
    errors: list[str] = []
    if not cfg.experiment.name.strip():
        errors.append("experiment.name must not be empty")
    if cfg.hardware.precision not in ALLOWED_PRECISIONS:
        errors.append(f"hardware.precision must be one of {sorted(ALLOWED_PRECISIONS)}")
    if not cfg.hardware.devices:
        errors.append("hardware.devices must contain at least one device")
    if len(set(cfg.hardware.devices)) != len(cfg.hardware.devices):
        errors.append("hardware.devices must not contain duplicates")
    if any(device < 0 for device in cfg.hardware.devices):
        errors.append("hardware.devices must contain non-negative device indices")
    if cfg.hardware.max_gpu_memory_gb < 0:
        errors.append("hardware.max_gpu_memory_gb must be non-negative")
    if cfg.hardware.workers_per_gpu != 1:
        errors.append("hardware.workers_per_gpu must be 1 for the supported episode sharding design")
    if cfg.benchmark.backend not in ALLOWED_BACKENDS:
        errors.append(f"benchmark.backend must be one of {sorted(ALLOWED_BACKENDS)}")
    if cfg.benchmark.episodes_per_task <= 0:
        errors.append("benchmark.episodes_per_task must be positive")
    if cfg.benchmark.max_steps <= 0 or cfg.benchmark.control_horizon <= 0:
        errors.append("benchmark.max_steps and control_horizon must be positive")
    if cfg.benchmark.num_steps_wait < 0:
        errors.append("benchmark.num_steps_wait must be non-negative")
    if cfg.benchmark.control_horizon > cfg.benchmark.max_steps:
        errors.append("benchmark.control_horizon cannot exceed max_steps")
    if any(value <= 0 for value in cfg.benchmark.image_size):
        errors.append("benchmark.image_size values must be positive")
    if cfg.perturbation.enabled:
        unknown_categories = set(cfg.perturbation.categories) - ALLOWED_CATEGORIES
        unknown_levels = set(cfg.perturbation.levels) - ALLOWED_LEVELS
        if not cfg.perturbation.categories or not cfg.perturbation.levels:
            errors.append("enabled perturbations require at least one category and level")
        if unknown_categories:
            errors.append(f"illegal perturbation categories: {sorted(unknown_categories)}")
        if unknown_levels:
            errors.append(f"illegal perturbation levels: {sorted(unknown_levels)}")
        if cfg.benchmark.backend not in {"libero_plus", "mock"}:
            errors.append("enabled perturbations require benchmark.backend=libero_plus or mock")
        if cfg.perturbation.variant_selection not in ALLOWED_VARIANT_SELECTIONS:
            errors.append(
                "perturbation.variant_selection must be one of "
                f"{sorted(ALLOWED_VARIANT_SELECTIONS)}"
            )
        if (
            cfg.perturbation.variant_selection == "all_once"
            and cfg.benchmark.episodes_per_task != 1
        ):
            errors.append(
                "perturbation.variant_selection=all_once requires "
                "benchmark.episodes_per_task=1 (the official LIBERO-Plus protocol)"
            )
    elif cfg.benchmark.backend == "libero_plus":
        errors.append("benchmark.backend=libero_plus requires perturbation.enabled=true")
    elif cfg.perturbation.variant_selection != "sample":
        errors.append("disabled perturbations require perturbation.variant_selection=sample")
    if cfg.recording.video_format not in {"mp4", "avi"}:
        errors.append("recording.video_format must be mp4 or avi")
    if cfg.policy.variant not in ALLOWED_POLICY_VARIANTS:
        errors.append(f"policy.variant must be one of {sorted(ALLOWED_POLICY_VARIANTS)}")
    expected_future = {"fastwam": False, "joint_wam": True, "idm": True, "mock": False}
    if (
        cfg.policy.variant in expected_future
        and cfg.policy.test_time_future_imagination != expected_future[cfg.policy.variant]
    ):
        errors.append(
            "policy.test_time_future_imagination is inconsistent with policy.variant: "
            f"{cfg.policy.variant} requires {expected_future[cfg.policy.variant]}"
        )
    if cfg.benchmark.backend == "mock" and cfg.policy.variant != "mock":
        errors.append("benchmark.backend=mock requires policy.variant=mock")
    if cfg.benchmark.backend != "mock" and cfg.policy.variant == "mock":
        errors.append("real benchmark backends cannot use policy.variant=mock")
    model_markers = {
        "fastwam": "_uncond_",
        "joint_wam": "_joint_",
        "idm": "_idm_",
    }
    marker = model_markers.get(cfg.policy.variant)
    if marker and marker not in cfg.checkpoint.model_name:
        errors.append(
            f"policy.variant={cfg.policy.variant} requires checkpoint.model_name containing {marker!r}; "
            "future imagination is not a runtime visualization toggle"
        )
    if marker and cfg.checkpoint.path is not None and marker not in cfg.checkpoint.path.name:
        errors.append(
            f"policy.variant={cfg.policy.variant} requires a checkpoint filename containing {marker!r}; "
            "do not load an uncond checkpoint into a future-imagination architecture"
        )
    if errors:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(errors))


def validate_runtime_paths(cfg: EvalConfig, *, require_checkpoint: bool = True) -> None:
    """Validate expensive-run prerequisites immediately before evaluation."""
    missing: list[str] = []
    if not cfg.benchmark.suite_config.exists():
        missing.append(f"suite config: {cfg.benchmark.suite_config}")
    if cfg.benchmark.backend == "libero" and not Path("third_party/LIBERO/libero").exists():
        missing.append("clean LIBERO checkout: third_party/LIBERO")
    if cfg.benchmark.backend == "libero_plus" and not Path("third_party/LIBERO-plus/libero").exists():
        missing.append("LIBERO-Plus checkout: third_party/LIBERO-plus")
    if cfg.benchmark.backend != "mock" and not Path("third_party/FastWAM/src/fastwam").exists():
        missing.append("Fast-WAM package: third_party/FastWAM/src/fastwam")
    if (
        cfg.benchmark.backend == "libero_plus"
        and require_checkpoint
        and not Path("third_party/LIBERO-plus/libero/libero/assets").exists()
    ):
        missing.append(
            "LIBERO-Plus assets: third_party/LIBERO-plus/libero/libero/assets "
            "(download the official assets.zip)"
        )
    if cfg.benchmark.backend != "mock" and require_checkpoint:
        if cfg.checkpoint.path is None or not cfg.checkpoint.path.is_file():
            missing.append(f"checkpoint: {cfg.checkpoint.path}")
        if cfg.checkpoint.dataset_stats_path is None or not cfg.checkpoint.dataset_stats_path.is_file():
            missing.append(f"dataset stats: {cfg.checkpoint.dataset_stats_path}")
        if cfg.checkpoint.config_path is None or not cfg.checkpoint.config_path.is_file():
            missing.append(f"Fast-WAM config: {cfg.checkpoint.config_path}")
    if missing:
        raise ConfigError("Missing runtime prerequisites:\n- " + "\n- ".join(missing))


def validate_hardware_inventory(
    cfg: EvalConfig,
    *,
    cuda_available: bool,
    device_memory_gb: Sequence[float],
    cuda_visible_devices: str | None = None,
) -> None:
    """Validate a real-evaluation config against the visible CUDA inventory."""
    if cfg.benchmark.backend == "mock":
        return
    errors: list[str] = []
    if not cuda_available:
        errors.append("CUDA is not available for a real benchmark backend")
    visible_count = len(device_memory_gb)
    if len(cfg.hardware.devices) > visible_count:
        errors.append(
            f"configuration requests {len(cfg.hardware.devices)} GPUs {cfg.hardware.devices}, "
            f"but only {visible_count} CUDA devices are visible"
        )

    # With CUDA_VISIBLE_DEVICES set, torch re-numbers the selected devices from
    # zero. Otherwise configuration indices address the physical torch inventory.
    inventory_indices = (
        tuple(range(len(cfg.hardware.devices)))
        if cuda_visible_devices not in (None, "")
        else cfg.hardware.devices
    )
    for configured_device, inventory_index in zip(cfg.hardware.devices, inventory_indices):
        if inventory_index >= visible_count:
            errors.append(
                f"configured GPU {configured_device} resolves to unavailable CUDA index {inventory_index}"
            )
            continue
        available_gb = float(device_memory_gb[inventory_index])
        if available_gb + 1e-6 < cfg.hardware.max_gpu_memory_gb:
            errors.append(
                f"GPU {configured_device} has {available_gb:.1f} GiB, below configured memory budget "
                f"{cfg.hardware.max_gpu_memory_gb:.1f} GiB"
            )
    if errors:
        raise ConfigError("Hardware inventory mismatch:\n- " + "\n- ".join(errors))


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> EvalConfig:
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse YAML {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {source}")
    return _build(apply_overrides(raw, overrides), source)
