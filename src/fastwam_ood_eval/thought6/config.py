"""Strict, versioned Phase 6 configuration with no sigma override surface."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from fastwam_ood_eval.thought6 import (
    ACTION_DENOISE_STEPS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FUTURE_K,
    SIGMA_THRESHOLD,
)
from fastwam_ood_eval.thought6.schemas import Thought6Error, canonical_json_bytes


CONFIG_SCHEMA = "thought6.config.v1"
OUTPUT_ROOT = Path("outputs/thought6/phase6_sigma_aware_future_fusion_v1")
FROZEN_BACKBONE_SHA256 = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
FROZEN_PARAMETER_SHA256 = "ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8"
STAGES = {"audit", "phase6a", "phase6b", "phase6c_stage1", "phase6c_stage2"}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class Thought6Config:
    source_path: Path
    stage: str
    experiment_name: str
    output_dir: Path
    seed: int
    device: str
    dtype: str
    max_gpu_memory_gb: float
    backbone_checkpoint: Path
    backbone_checkpoint_sha256: str
    backbone_frozen_parameter_sha256: str
    dataset_stats_path: Path
    dataset_stats_sha256: str
    fastwam_commit: str
    thought3_config_path: Path
    authoritative_phase2_result: Path
    adapter_manifest_path: Path
    adapter_checkpoint_path: Path
    adapter_file_sha256: str
    adapter_state_sha256: str
    adapter_parameter_count: int
    scheduler_source_path: Path
    suite_dataset_roots: Mapping[str, Path]
    suite_dataset_revisions: Mapping[str, str | None]
    classification_path: Path
    tasks_per_suite: int
    utility_episodes_per_task: int
    utility_flow_slots: int
    stage1_states_per_task: int
    stage2_total_states_per_task: int
    bootstrap_replicates: int
    bootstrap_seed: int
    clean_utility_margin: float
    clean_rollout_margin_pp: float
    rollout_max_steps: Mapping[str, int]

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        payload.pop("source_path", None)
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(key): normalize(nested) for key, nested in value.items()}
            if isinstance(value, tuple):
                return [normalize(nested) for nested in value]
            return value

        payload = normalize(asdict(self))
        payload.update(
            {
                "schema_version": CONFIG_SCHEMA,
                "sigma_threshold": SIGMA_THRESHOLD,
                "sigma_threshold_source": "code_constant_not_configurable",
                "action_denoise_steps": ACTION_DENOISE_STEPS,
                "future_k": FUTURE_K,
            }
        )
        return payload


def _reject_sigma_override(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"sigma_0", "sigma_threshold", "gate_threshold"}:
                raise Thought6Error(
                    f"{path}.{key} is forbidden: sigma threshold is fixed in code at 0.5"
                )
            _reject_sigma_override(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_sigma_override(nested, path=f"{path}[{index}]")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Thought6Error(f"{name} must be a mapping")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Thought6Error(f"{name} must be a non-empty path string")
    return Path(value)


def load_thought6_config(path: str | Path) -> Thought6Config:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    _reject_sigma_override(root)
    if root.get("schema_version") != CONFIG_SCHEMA:
        raise Thought6Error("Thought6 config schema mismatch")
    experiment = _mapping(root.get("experiment"), "experiment")
    runtime = _mapping(root.get("runtime"), "runtime")
    backbone = _mapping(root.get("backbone"), "backbone")
    adapter = _mapping(root.get("adapter"), "adapter")
    scheduler = _mapping(root.get("scheduler"), "scheduler")
    cohort = _mapping(root.get("cohort"), "cohort")
    statistics = _mapping(root.get("statistics"), "statistics")
    gates = _mapping(root.get("gates"), "gates")
    rollout = _mapping(root.get("rollout"), "rollout")
    stage = str(experiment.get("stage"))
    if stage not in STAGES:
        raise Thought6Error(f"invalid Thought6 stage: {stage}")
    output_dir = _path(experiment.get("output_dir"), "experiment.output_dir")
    if output_dir != OUTPUT_ROOT:
        raise Thought6Error(f"Thought6 output must be the frozen namespace {OUTPUT_ROOT}")
    if int(runtime.get("action_denoise_steps", -1)) != ACTION_DENOISE_STEPS:
        raise Thought6Error("Phase 6 freezes 20 action denoising steps")
    if int(scheduler.get("future_k", -1)) != FUTURE_K:
        raise Thought6Error("Phase 6 freezes K=1")
    if float(scheduler.get("shift", -1)) != 5.0:
        raise Thought6Error("Phase 6 freezes scheduler shift=5")
    if int(scheduler.get("num_train_timesteps", -1)) != 1000:
        raise Thought6Error("Phase 6 freezes 1000 training timesteps")
    suites = tuple(cohort.get("suites", ()))
    if suites != SUITES:
        raise Thought6Error(f"Phase 6 suites must be exactly {SUITES}")
    roots = _mapping(cohort.get("dataset_roots"), "cohort.dataset_roots")
    revisions = _mapping(cohort.get("dataset_revisions"), "cohort.dataset_revisions")
    if set(roots) != set(SUITES) or set(revisions) != set(SUITES):
        raise Thought6Error("dataset roots/revisions must cover all four frozen suites")
    bootstrap_replicates = int(statistics.get("bootstrap_replicates", -1))
    bootstrap_seed = int(statistics.get("bootstrap_seed", -1))
    if bootstrap_replicates != BOOTSTRAP_REPLICATES or bootstrap_seed != BOOTSTRAP_SEED:
        raise Thought6Error("Phase 6 freezes 10,000 bootstrap replicates and seed 6607")
    cfg = Thought6Config(
        source_path=source,
        stage=stage,
        experiment_name=str(experiment["name"]),
        output_dir=output_dir,
        seed=int(experiment["seed"]),
        device=str(runtime["device"]),
        dtype=str(runtime["dtype"]),
        max_gpu_memory_gb=float(runtime["max_gpu_memory_gb"]),
        backbone_checkpoint=_path(backbone["checkpoint_path"], "backbone.checkpoint_path"),
        backbone_checkpoint_sha256=str(backbone["checkpoint_sha256"]),
        backbone_frozen_parameter_sha256=str(backbone["frozen_parameter_sha256"]),
        dataset_stats_path=_path(backbone["dataset_stats_path"], "backbone.dataset_stats_path"),
        dataset_stats_sha256=str(backbone["dataset_stats_sha256"]),
        fastwam_commit=str(backbone["fastwam_commit"]),
        thought3_config_path=_path(adapter["thought3_config_path"], "adapter.thought3_config_path"),
        authoritative_phase2_result=_path(adapter["phase2_result_path"], "adapter.phase2_result_path"),
        adapter_manifest_path=_path(adapter["manifest_path"], "adapter.manifest_path"),
        adapter_checkpoint_path=_path(adapter["checkpoint_path"], "adapter.checkpoint_path"),
        adapter_file_sha256=str(adapter["file_sha256"]),
        adapter_state_sha256=str(adapter["state_sha256"]),
        adapter_parameter_count=int(adapter["trainable_parameter_count"]),
        scheduler_source_path=_path(scheduler["source_path"], "scheduler.source_path"),
        suite_dataset_roots={suite: Path(str(roots[suite])) for suite in SUITES},
        suite_dataset_revisions={
            suite: None if revisions[suite] is None else str(revisions[suite])
            for suite in SUITES
        },
        classification_path=_path(cohort["classification_path"], "cohort.classification_path"),
        tasks_per_suite=int(cohort["tasks_per_suite"]),
        utility_episodes_per_task=int(cohort["utility_episodes_per_task"]),
        utility_flow_slots=int(cohort["utility_flow_slots"]),
        stage1_states_per_task=int(cohort["stage1_states_per_task"]),
        stage2_total_states_per_task=int(cohort["stage2_total_states_per_task"]),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        clean_utility_margin=float(gates["clean_utility_margin"]),
        clean_rollout_margin_pp=float(gates["clean_rollout_margin_pp"]),
        rollout_max_steps={str(key): int(value) for key, value in _mapping(rollout["max_steps"], "rollout.max_steps").items()},
    )
    if (
        cfg.dtype != "bfloat16"
        or cfg.tasks_per_suite != 2
        or cfg.utility_episodes_per_task < 4
        or cfg.utility_flow_slots != 32
        or cfg.stage1_states_per_task != 10
        or cfg.stage2_total_states_per_task != 20
        or cfg.clean_utility_margin != -0.002
        or cfg.clean_rollout_margin_pp != -5.0
    ):
        raise Thought6Error("Thought6 config differs from the frozen Phase 6 protocol")
    if (
        cfg.backbone_checkpoint_sha256 != FROZEN_BACKBONE_SHA256
        or cfg.backbone_frozen_parameter_sha256 != FROZEN_PARAMETER_SHA256
        or len(cfg.adapter_file_sha256) != 64
        or len(cfg.adapter_state_sha256) != 64
    ):
        raise Thought6Error("Thought6 backbone/Adapter identity differs from the frozen protocol")
    return cfg


def config_summary(cfg: Thought6Config) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "stage": cfg.stage,
        "config_fingerprint": cfg.fingerprint,
        "output_dir": str(cfg.output_dir),
        "sigma_threshold": SIGMA_THRESHOLD,
        "sigma_configurable": False,
        "real_gpu_run": False,
    }
