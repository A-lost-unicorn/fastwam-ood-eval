"""Standalone Thought 3 YAML configuration and invariant validation."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fastwam_ood_eval.thought3 import THOUGHT3_CONFIG_SCHEMA
from fastwam_ood_eval.thought3.safety import (
    Thought3SafetyError,
    ensure_standard_training_source,
    ensure_thought3_output_path,
)
from fastwam_ood_eval.thought3.schemas import (
    ALLOWED_K,
    NATIVE_FUTURE_SHAPE,
    canonical_json,
)


class Thought3ConfigError(ValueError):
    """Raised when a Thought 3 configuration violates the frozen design."""


VALID_VARIANTS = ("B0", "A0", "A1", "A2", "A4", "A-shuffle")
VALID_BACKENDS = ("mock", "fastwam")
VALID_CACHE_DTYPES = ("bfloat16", "float32")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int = 3407


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "mock"
    device: str = "cpu"
    action_denoise_steps: int = 20
    online_use_cache: bool = False
    max_gpu_memory_gb: float = 43.0


@dataclass(frozen=True)
class BackboneConfig:
    checkpoint_path: Path | None
    checkpoint_sha256: str
    dataset_stats_path: Path | None
    dataset_stats_sha256: str
    fastwam_commit: str
    model_config_path: Path | None
    model_config_sha256: str
    num_video_frames: int = 9
    image_height: int = 224
    image_width: int = 448


@dataclass(frozen=True)
class SamplerConfig:
    active_k: int
    cache_k: tuple[int, ...] = ALLOWED_K
    shift: float = 5.0
    num_train_timesteps: int = 1000
    global_cache_seed: int = 3407
    rand_device: str = "cpu"
    latent_shape: tuple[int, int, int, int] = NATIVE_FUTURE_SHAPE
    cache_dtype: str = "bfloat16"


@dataclass(frozen=True)
class AdapterConfig:
    enabled: bool
    injection: str = "action_encoder_output"
    input_channels: int = 48
    action_hidden_dim: int = 1024
    future_dim: int = 256
    attention_dim: int = 512
    num_heads: int = 8
    max_projected_grid: tuple[int, int, int] = (2, 7, 14)
    zero_init_gate: bool = True
    lora_enabled: bool = False


@dataclass(frozen=True)
class DataConfig:
    dataset_roots: tuple[Path, ...]
    dataset_revision: str
    inventory_path: Path | None
    split_seed: int = 3407
    development_fraction: float = 0.1
    mock_sample_count: int = 16
    camera_keys: tuple[str, ...] = ("image", "wrist_image")


@dataclass(frozen=True)
class CacheConfig:
    root: Path
    shard_size: int = 512
    pilot_limit: int | None = None
    required_free_space_fraction: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 100
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    train_seed: int = 3407
    gradient_checkpointing: bool = False
    gate_l2: float = 0.0
    checkpoint_interval: int = 25


@dataclass(frozen=True)
class Thought3Config:
    variant: str
    experiment: ExperimentConfig
    runtime: RuntimeConfig
    backbone: BackboneConfig
    sampler: SamplerConfig
    adapter: AdapterConfig
    data: DataConfig
    cache: CacheConfig
    training: TrainingConfig
    source_path: Path
    schema_version: str = THOUGHT3_CONFIG_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)

        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            return value

        return normalize(payload)

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        payload.pop("source_path", None)
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def adapter_structural_payload(self) -> dict[str, Any]:
        return {
            "injection": self.adapter.injection,
            "input_channels": self.adapter.input_channels,
            "action_hidden_dim": self.adapter.action_hidden_dim,
            "future_dim": self.adapter.future_dim,
            "attention_dim": self.adapter.attention_dim,
            "num_heads": self.adapter.num_heads,
            "max_projected_grid": list(self.adapter.max_projected_grid),
            "zero_init_gate": self.adapter.zero_init_gate,
            "lora_enabled": self.adapter.lora_enabled,
        }

    @property
    def adapter_structural_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.adapter_structural_payload).encode("utf-8")
        ).hexdigest()


DEFAULTS: dict[str, Any] = {
    "schema_version": THOUGHT3_CONFIG_SCHEMA,
    "variant": "A1",
    "experiment": {
        "name": "thought3_smoke",
        "output_dir": "outputs/thought3/smoke",
        "seed": 3407,
    },
    "runtime": {
        "backend": "mock",
        "device": "cpu",
        "action_denoise_steps": 20,
        "online_use_cache": False,
        "max_gpu_memory_gb": 43.0,
    },
    "backbone": {
        "checkpoint_path": "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        "checkpoint_sha256": "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579",
        "dataset_stats_path": (
            "checkpoints/fastwam_release/"
            "libero_uncond_2cam224_dataset_stats.json"
        ),
        "dataset_stats_sha256": "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638",
        "fastwam_commit": "45d8e1458921d83f8ad6cf9ce993d371208dabd0",
        "model_config_path": "third_party/FastWAM/configs/model/fastwam.yaml",
        "model_config_sha256": "ab3c2ffde9933e7576c747fecce82bd7d28c9c6478c1b53fcac02b3012be416c",
        "num_video_frames": 9,
        "image_height": 224,
        "image_width": 448,
    },
    "sampler": {
        "active_k": 1,
        "cache_k": [1, 2, 4],
        "shift": 5.0,
        "num_train_timesteps": 1000,
        "global_cache_seed": 3407,
        "rand_device": "cpu",
        "latent_shape": [48, 2, 14, 28],
        "cache_dtype": "bfloat16",
    },
    "adapter": {
        "enabled": True,
        "injection": "action_encoder_output",
        "input_channels": 48,
        "action_hidden_dim": 1024,
        "future_dim": 256,
        "attention_dim": 512,
        "num_heads": 8,
        "max_projected_grid": [2, 7, 14],
        "zero_init_gate": True,
        "lora_enabled": False,
    },
    "data": {
        "dataset_roots": [],
        "dataset_revision": "mock-v1",
        "inventory_path": None,
        "split_seed": 3407,
        "development_fraction": 0.1,
        "mock_sample_count": 16,
        "camera_keys": ["image", "wrist_image"],
    },
    "cache": {
        "root": "outputs/thought3/cache/smoke",
        "shard_size": 512,
        "pilot_limit": None,
        "required_free_space_fraction": 0.2,
    },
    "training": {
        "max_steps": 100,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "train_seed": 3407,
        "gradient_checkpointing": False,
        "gate_l2": 0.0,
        "checkpoint_interval": 25,
    },
}


_ALLOWED_KEYS: dict[str, set[str]] = {
    "experiment": {"name", "output_dir", "seed"},
    "runtime": {
        "backend",
        "device",
        "action_denoise_steps",
        "online_use_cache",
        "max_gpu_memory_gb",
    },
    "backbone": {
        "checkpoint_path",
        "checkpoint_sha256",
        "dataset_stats_path",
        "dataset_stats_sha256",
        "fastwam_commit",
        "model_config_path",
        "model_config_sha256",
        "num_video_frames",
        "image_height",
        "image_width",
    },
    "sampler": {
        "active_k",
        "cache_k",
        "shift",
        "num_train_timesteps",
        "global_cache_seed",
        "rand_device",
        "latent_shape",
        "cache_dtype",
    },
    "adapter": {
        "enabled",
        "injection",
        "input_channels",
        "action_hidden_dim",
        "future_dim",
        "attention_dim",
        "num_heads",
        "max_projected_grid",
        "zero_init_gate",
        "lora_enabled",
    },
    "data": {
        "dataset_roots",
        "dataset_revision",
        "inventory_path",
        "split_seed",
        "development_fraction",
        "mock_sample_count",
        "camera_keys",
    },
    "cache": {
        "root",
        "shard_size",
        "pilot_limit",
        "required_free_space_fraction",
    },
    "training": {
        "max_steps",
        "microbatch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "train_seed",
        "gradient_checkpointing",
        "gate_l2",
        "checkpoint_interval",
    },
}


def _reject_unknown_keys(data: Mapping[str, Any]) -> None:
    top_level = {"schema_version", "variant", *_ALLOWED_KEYS}
    unknown_top = sorted(set(data) - top_level)
    if unknown_top:
        raise Thought3ConfigError(
            f"Unknown Thought3 top-level config keys: {unknown_top}"
        )
    for section, allowed in _ALLOWED_KEYS.items():
        values = data.get(section)
        if isinstance(values, Mapping):
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise Thought3ConfigError(
                    f"Unknown Thought3 {section} config keys: {unknown}"
                )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise Thought3ConfigError(f"Invalid override value: {raw!r}") from exc


def apply_overrides(
    data: Mapping[str, Any],
    overrides: Sequence[str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(data))
    for override in overrides:
        if "=" not in override:
            raise Thought3ConfigError(
                f"Override must use dotted.key=value syntax: {override!r}"
            )
        dotted, raw_value = override.split("=", 1)
        keys = [key.strip() for key in dotted.split(".") if key.strip()]
        if not keys:
            raise Thought3ConfigError(f"Override key must not be empty: {override!r}")
        cursor = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise Thought3ConfigError(
                    f"Cannot override below non-mapping key: {dotted!r}"
                )
            cursor = child
        cursor[keys[-1]] = _parse_override_value(raw_value)
    return result


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _sha(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise Thought3ConfigError(f"{name} must be a 64-character SHA-256 digest")
    return normalized


def _tuple_int(value: Any, *, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Thought3ConfigError(f"{name} must contain exactly {length} integers")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise Thought3ConfigError(f"{name} must contain integers") from exc


def _build(data: Mapping[str, Any], source_path: Path) -> Thought3Config:
    _reject_unknown_keys(data)
    for section in (
        "experiment",
        "runtime",
        "backbone",
        "sampler",
        "adapter",
        "data",
        "cache",
        "training",
    ):
        if not isinstance(data.get(section), Mapping):
            raise Thought3ConfigError(f"Missing or invalid top-level section: {section}")
    ex = data["experiment"]
    runtime = data["runtime"]
    backbone = data["backbone"]
    sampler = data["sampler"]
    adapter = data["adapter"]
    dataset = data["data"]
    cache = data["cache"]
    training = data["training"]
    cfg = Thought3Config(
        schema_version=str(data.get("schema_version", THOUGHT3_CONFIG_SCHEMA)),
        variant=str(data.get("variant", "A1")),
        experiment=ExperimentConfig(
            name=str(ex.get("name", "thought3_smoke")),
            output_dir=Path(str(ex["output_dir"])),
            seed=int(ex.get("seed", 3407)),
        ),
        runtime=RuntimeConfig(
            backend=str(runtime.get("backend", "mock")).lower(),
            device=str(runtime.get("device", "cpu")),
            action_denoise_steps=int(runtime.get("action_denoise_steps", 20)),
            online_use_cache=bool(runtime.get("online_use_cache", False)),
            max_gpu_memory_gb=float(runtime.get("max_gpu_memory_gb", 43.0)),
        ),
        backbone=BackboneConfig(
            checkpoint_path=_path_or_none(backbone.get("checkpoint_path")),
            checkpoint_sha256=_sha(
                backbone.get("checkpoint_sha256"), "backbone.checkpoint_sha256"
            ),
            dataset_stats_path=_path_or_none(backbone.get("dataset_stats_path")),
            dataset_stats_sha256=_sha(
                backbone.get("dataset_stats_sha256"),
                "backbone.dataset_stats_sha256",
            ),
            fastwam_commit=str(backbone.get("fastwam_commit", "")).strip(),
            model_config_path=_path_or_none(backbone.get("model_config_path")),
            model_config_sha256=_sha(
                backbone.get("model_config_sha256"),
                "backbone.model_config_sha256",
            ),
            num_video_frames=int(backbone.get("num_video_frames", 9)),
            image_height=int(backbone.get("image_height", 224)),
            image_width=int(backbone.get("image_width", 448)),
        ),
        sampler=SamplerConfig(
            active_k=int(sampler.get("active_k", 1)),
            cache_k=tuple(int(value) for value in sampler.get("cache_k", ALLOWED_K)),
            shift=float(sampler.get("shift", 5.0)),
            num_train_timesteps=int(sampler.get("num_train_timesteps", 1000)),
            global_cache_seed=int(sampler.get("global_cache_seed", 3407)),
            rand_device=str(sampler.get("rand_device", "cpu")),
            latent_shape=_tuple_int(
                sampler.get("latent_shape", NATIVE_FUTURE_SHAPE),
                name="sampler.latent_shape",
                length=4,
            ),
            cache_dtype=str(sampler.get("cache_dtype", "bfloat16")).lower(),
        ),
        adapter=AdapterConfig(
            enabled=bool(adapter.get("enabled", True)),
            injection=str(adapter.get("injection", "action_encoder_output")),
            input_channels=int(adapter.get("input_channels", 48)),
            action_hidden_dim=int(adapter.get("action_hidden_dim", 1024)),
            future_dim=int(adapter.get("future_dim", 256)),
            attention_dim=int(adapter.get("attention_dim", 512)),
            num_heads=int(adapter.get("num_heads", 8)),
            max_projected_grid=_tuple_int(
                adapter.get("max_projected_grid", (2, 7, 14)),
                name="adapter.max_projected_grid",
                length=3,
            ),
            zero_init_gate=bool(adapter.get("zero_init_gate", True)),
            lora_enabled=bool(adapter.get("lora_enabled", False)),
        ),
        data=DataConfig(
            dataset_roots=tuple(
                Path(str(value)) for value in dataset.get("dataset_roots", [])
            ),
            dataset_revision=str(dataset.get("dataset_revision", "")).strip(),
            inventory_path=_path_or_none(dataset.get("inventory_path")),
            split_seed=int(dataset.get("split_seed", 3407)),
            development_fraction=float(dataset.get("development_fraction", 0.1)),
            mock_sample_count=int(dataset.get("mock_sample_count", 16)),
            camera_keys=tuple(
                str(value) for value in dataset.get("camera_keys", ("image", "wrist_image"))
            ),
        ),
        cache=CacheConfig(
            root=Path(str(cache["root"])),
            shard_size=int(cache.get("shard_size", 512)),
            pilot_limit=(
                None
                if cache.get("pilot_limit") in (None, "")
                else int(cache["pilot_limit"])
            ),
            required_free_space_fraction=float(
                cache.get("required_free_space_fraction", 0.2)
            ),
        ),
        training=TrainingConfig(
            max_steps=int(training.get("max_steps", 100)),
            microbatch_size=int(training.get("microbatch_size", 1)),
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 8)
            ),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-2)),
            train_seed=int(training.get("train_seed", 3407)),
            gradient_checkpointing=bool(
                training.get("gradient_checkpointing", False)
            ),
            gate_l2=float(training.get("gate_l2", 0.0)),
            checkpoint_interval=int(training.get("checkpoint_interval", 25)),
        ),
        source_path=source_path,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Thought3Config) -> None:
    errors: list[str] = []
    if cfg.schema_version != THOUGHT3_CONFIG_SCHEMA:
        errors.append(f"schema_version must be {THOUGHT3_CONFIG_SCHEMA}")
    if cfg.variant not in VALID_VARIANTS:
        errors.append(f"variant must be one of {VALID_VARIANTS}")
    if cfg.runtime.backend not in VALID_BACKENDS:
        errors.append(f"runtime.backend must be one of {VALID_BACKENDS}")
    expected_k = {"B0": 0, "A0": 0, "A1": 1, "A2": 2, "A4": 4}
    if cfg.variant in expected_k and cfg.sampler.active_k != expected_k[cfg.variant]:
        errors.append(
            f"variant={cfg.variant} requires sampler.active_k={expected_k[cfg.variant]}"
        )
    if cfg.variant == "A-shuffle" and cfg.sampler.active_k not in ALLOWED_K:
        errors.append("A-shuffle requires sampler.active_k in [1,2,4]")
    expected_adapter = cfg.variant != "B0"
    if cfg.adapter.enabled != expected_adapter:
        errors.append(
            f"variant={cfg.variant} requires adapter.enabled={str(expected_adapter).lower()}"
        )
    if tuple(cfg.sampler.cache_k) != ALLOWED_K:
        errors.append("sampler.cache_k must be exactly [1,2,4]")
    if tuple(cfg.sampler.latent_shape) != NATIVE_FUTURE_SHAPE:
        errors.append(f"sampler.latent_shape must be {NATIVE_FUTURE_SHAPE}")
    if cfg.sampler.shift != 5.0 or cfg.sampler.num_train_timesteps != 1000:
        errors.append("first protocol freezes sampler shift=5 and num_train_timesteps=1000")
    if cfg.sampler.global_cache_seed < 0:
        errors.append("sampler.global_cache_seed must be non-negative")
    if cfg.sampler.rand_device != "cpu":
        errors.append("first protocol freezes sampler.rand_device=cpu")
    if cfg.sampler.cache_dtype not in VALID_CACHE_DTYPES:
        errors.append(f"sampler.cache_dtype must be one of {VALID_CACHE_DTYPES}")
    if cfg.runtime.action_denoise_steps != 20:
        errors.append("runtime.action_denoise_steps must remain 20")
    if cfg.runtime.online_use_cache:
        errors.append("runtime.online_use_cache must be false")
    if cfg.runtime.max_gpu_memory_gb <= 0 or cfg.runtime.max_gpu_memory_gb > 43.0:
        errors.append("runtime.max_gpu_memory_gb must be in (0,43]")
    if cfg.backbone.num_video_frames != 9:
        errors.append("backbone.num_video_frames must remain 9")
    if (cfg.backbone.image_height, cfg.backbone.image_width) != (224, 448):
        errors.append("backbone image size must remain [224,448]")
    if len(cfg.backbone.fastwam_commit) != 40:
        errors.append("backbone.fastwam_commit must be a 40-character Git commit")
    if cfg.adapter.injection != "action_encoder_output":
        errors.append("Phase B supports only adapter.injection=action_encoder_output")
    if cfg.adapter.enabled:
        if min(
            cfg.adapter.input_channels,
            cfg.adapter.action_hidden_dim,
            cfg.adapter.future_dim,
            cfg.adapter.attention_dim,
            cfg.adapter.num_heads,
        ) <= 0:
            errors.append("adapter dimensions and heads must be positive")
        if cfg.adapter.attention_dim % cfg.adapter.num_heads:
            errors.append("adapter.attention_dim must be divisible by num_heads")
        if not cfg.adapter.zero_init_gate:
            errors.append("first protocol requires adapter.zero_init_gate=true")
        if cfg.adapter.lora_enabled:
            errors.append("Phase B first protocol requires adapter.lora_enabled=false")
    if not cfg.data.dataset_revision:
        errors.append("data.dataset_revision must not be empty")
    if not 0 < cfg.data.development_fraction < 1:
        errors.append("data.development_fraction must be between 0 and 1")
    if cfg.data.split_seed < 0 or cfg.data.mock_sample_count <= 0:
        errors.append("data split seed must be non-negative and mock_sample_count positive")
    if len(cfg.data.camera_keys) != 2 or len(set(cfg.data.camera_keys)) != 2:
        errors.append("data.camera_keys must contain two unique cameras")
    if cfg.runtime.backend == "fastwam" and not cfg.data.dataset_roots:
        errors.append("fastwam backend requires standard LIBERO data.dataset_roots")
    if cfg.cache.shard_size <= 0:
        errors.append("cache.shard_size must be positive")
    if cfg.cache.pilot_limit is not None and cfg.cache.pilot_limit <= 0:
        errors.append("cache.pilot_limit must be positive when set")
    if not 0 <= cfg.cache.required_free_space_fraction <= 1:
        errors.append("cache.required_free_space_fraction must be in [0,1]")
    if min(
        cfg.training.max_steps,
        cfg.training.microbatch_size,
        cfg.training.gradient_accumulation_steps,
        cfg.training.checkpoint_interval,
    ) <= 0:
        errors.append("training steps, batch settings and checkpoint interval must be positive")
    if cfg.training.learning_rate <= 0 or cfg.training.weight_decay < 0:
        errors.append("training learning_rate must be positive and weight_decay non-negative")
    if cfg.training.gate_l2 < 0 or cfg.training.train_seed < 0:
        errors.append("training gate_l2 and train_seed must be non-negative")
    try:
        ensure_thought3_output_path(cfg.experiment.output_dir)
        ensure_thought3_output_path(cfg.cache.root)
        for root in cfg.data.dataset_roots:
            ensure_standard_training_source(root)
    except Thought3SafetyError as exc:
        errors.append(str(exc))
    if errors:
        raise Thought3ConfigError("; ".join(errors))


def load_thought3_config(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> Thought3Config:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise Thought3ConfigError(f"Invalid YAML: {source}") from exc
    if not isinstance(loaded, Mapping):
        raise Thought3ConfigError(f"Thought3 config root must be a mapping: {source}")
    merged = _deep_merge(DEFAULTS, loaded)
    merged = apply_overrides(merged, overrides)
    return _build(merged, source.resolve())
