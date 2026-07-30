"""Schemas and pure analysis for the real K=1 online action counterfactual."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fastwam_ood_eval.thought3.schemas import (
    BaseSampleIdentity,
    canonical_json,
)


ONLINE_CF_CONFIG_SCHEMA = "thought3.k1_online_counterfactual.config.v1"
ONLINE_CF_SAMPLE_SCHEMA = "thought3.k1_online_counterfactual.sample.v1"
ONLINE_CF_AGGREGATE_SCHEMA = "thought3.k1_online_counterfactual.aggregate.v1"
ONLINE_CF_SHUFFLE_SCHEMA = "thought3.k1_online.shuffle_manifest.v1"
ONLINE_CF_DECISION_SCHEMA = "thought3.k1_online_counterfactual.decision.v1"
ONLINE_CF_CLASSIFICATIONS = (
    "future_content_sensitivity_observed",
    "latent_presence_sensitivity_only",
    "no_material_online_action_sensitivity",
)
ONLINE_CF_CONDITIONS = ("B0", "correct", "null", "shuffle")
PAIR_KEYS = (
    "correct_null",
    "correct_shuffle",
    "null_shuffle",
    "b0_null",
)


class OnlineCounterfactualError(RuntimeError):
    """Raised when the online counterfactual protocol is violated."""


@dataclass(frozen=True)
class OnlineCohortSample:
    identity: BaseSampleIdentity
    split: str

    @property
    def base_sample_id(self) -> str:
        return self.identity.base_sample_id

    @property
    def episode_id(self) -> str:
        return self.identity.episode_id


@dataclass(frozen=True)
class K1OnlineCounterfactualConfig:
    source_path: Path
    raw: Mapping[str, Any]
    experiment_name: str
    output_dir: Path
    experiment_seed: int
    device: str
    action_horizon: int
    action_denoise_steps: int
    future_k: int
    future_shift: float
    future_num_train_timesteps: int
    rand_device: str
    max_gpu_memory_gb: float
    warmup_b0_calls: int
    warmup_future_action_calls: int
    thought3_config_path: Path
    thought3_config_sha256: str
    thought3_config_fingerprint: str
    e6_gate_path: Path
    e6_gate_sha256: str
    e6_checkpoint_dir: Path
    e6_adapter_sha256: str
    e6_checkpoint_manifest_sha256: str
    e6_adapter_state_sha256: str
    e6_checkpoint_config_fingerprint: str
    e6_adapter_fingerprint: str
    backbone_checkpoint_sha256: str
    dataset_stats_sha256: str
    split_fingerprint: str
    fastwam_commit: str
    frozen_parameter_sha256: str
    checkpoint_selection_disclosure: str
    action_seed_namespace: str
    future_seed_namespace: str
    shuffle_namespace: str
    cohort: tuple[OnlineCohortSample, ...]
    cohort_selection: str
    shuffle_seed: int
    expected_shuffle_mapping_sha256: str
    replay_repeats: int
    replay_hard_max_linf: float
    replay_l2_multiplier: float
    replay_absolute_l2_floor: float
    stable_sample_min_count: int
    action_hash_change_min_count: int
    null_kind: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(dict(self.raw)).encode("utf-8")
        ).hexdigest()

    @property
    def cohort_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "schema_version": (
                        "thought3.k1_online.cohort_manifest.v1"
                    ),
                    "selection": self.cohort_selection,
                    "samples": [
                        {
                            "base_sample_id": sample.base_sample_id,
                            "episode_id": sample.episode_id,
                            "identity": sample.identity.to_dict(),
                            "split": sample.split,
                        }
                        for sample in self.cohort
                    ],
                }
            ).encode("utf-8")
        ).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OnlineCounterfactualError(f"{name} must be a mapping")
    return value


def _sha256(value: object, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise OnlineCounterfactualError(
            f"{name} must be a 64-character SHA-256"
        )
    return text


def _git_sha(value: object, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise OnlineCounterfactualError(
            f"{name} must be a 40-character Git SHA"
        )
    return text


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise OnlineCounterfactualError(
            f"{name} keys changed: {sorted(set(value) ^ expected)}"
        )


def load_k1_online_counterfactual_config(
    path: str | Path,
) -> K1OnlineCounterfactualConfig:
    """Load and validate the standalone Phase 1 schema."""

    source_path = Path(path)
    value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    root = _mapping(value, "online counterfactual config")
    if root.get("schema_version") != ONLINE_CF_CONFIG_SCHEMA:
        raise OnlineCounterfactualError(
            f"schema_version must be {ONLINE_CF_CONFIG_SCHEMA}"
        )
    _strict_keys(
        root,
        {
            "schema_version",
            "experiment",
            "runtime",
            "source",
            "seeds",
            "cohort",
            "shuffle",
            "replay_floor",
            "null",
            "scope",
        },
        "config",
    )
    experiment = _mapping(root["experiment"], "experiment")
    runtime = _mapping(root["runtime"], "runtime")
    source = _mapping(root["source"], "source")
    seeds = _mapping(root["seeds"], "seeds")
    cohort_raw = _mapping(root["cohort"], "cohort")
    shared = _mapping(cohort_raw["shared_identity"], "cohort.shared_identity")
    shuffle = _mapping(root["shuffle"], "shuffle")
    replay = _mapping(root["replay_floor"], "replay_floor")
    null = _mapping(root["null"], "null")
    scope = _mapping(root["scope"], "scope")
    _strict_keys(
        experiment,
        {"name", "output_dir", "seed"},
        "experiment",
    )
    _strict_keys(
        runtime,
        {
            "backend",
            "device",
            "action_horizon",
            "action_denoise_steps",
            "future_k",
            "future_shift",
            "future_num_train_timesteps",
            "rand_device",
            "max_gpu_memory_gb",
            "warmup_b0_calls",
            "warmup_future_action_calls",
        },
        "runtime",
    )
    _strict_keys(
        source,
        {
            "thought3_config_path",
            "thought3_config_sha256",
            "thought3_config_fingerprint",
            "e6_gate_path",
            "e6_gate_sha256",
            "e6_checkpoint_dir",
            "e6_adapter_sha256",
            "e6_checkpoint_manifest_sha256",
            "e6_adapter_state_sha256",
            "e6_checkpoint_config_fingerprint",
            "e6_adapter_fingerprint",
            "backbone_checkpoint_sha256",
            "dataset_stats_sha256",
            "split_fingerprint",
            "fastwam_commit",
            "frozen_parameter_sha256",
            "checkpoint_selection_disclosure",
        },
        "source",
    )
    _strict_keys(
        seeds,
        {"action_namespace", "future_namespace", "shuffle_namespace"},
        "seeds",
    )
    _strict_keys(
        cohort_raw,
        {
            "selection",
            "sample_count",
            "shared_identity",
            "samples",
        },
        "cohort",
    )
    _strict_keys(
        shared,
        {
            "camera_keys",
            "checkpoint_sha256",
            "dataset_revision",
            "frame_index",
            "language",
            "preprocessing_sha256",
            "sampler_config_sha256",
            "split_manifest_sha256",
            "stats_sha256",
            "suite",
            "task_id",
            "task_name",
            "timestamp_ns",
        },
        "cohort.shared_identity",
    )
    _strict_keys(
        shuffle,
        {
            "seed",
            "expected_mapping_sha256",
            "require_other_episode",
            "require_one_to_one",
            "require_same_task",
            "reuse_recipient_future_noise_seed",
        },
        "shuffle",
    )
    _strict_keys(
        replay,
        {
            "repeats",
            "hard_max_linf",
            "l2_multiplier",
            "absolute_l2_floor",
            "stable_sample_min_count",
            "action_hash_change_min_count",
        },
        "replay_floor",
    )
    _strict_keys(
        null,
        {"kind", "tensor_substitute", "run_video_dit"},
        "null",
    )
    _strict_keys(
        scope,
        {
            "read_action_target",
            "read_training_future_cache",
            "read_future_rgb",
            "read_development",
            "read_ood",
            "read_rollout_success",
            "decode_future_video",
            "start_rollout",
        },
        "scope",
    )
    if any(bool(value) for value in scope.values()):
        raise OnlineCounterfactualError(
            "all forbidden online counterfactual scope flags must be false"
        )
    samples: list[OnlineCohortSample] = []
    for row_value in cohort_raw.get("samples", ()):
        row = _mapping(row_value, "cohort sample")
        _strict_keys(
            row,
            {
                "base_sample_id",
                "demonstration_id",
                "episode_index",
                "split",
            },
            "cohort sample",
        )
        identity = BaseSampleIdentity(
            dataset_revision=str(shared["dataset_revision"]),
            suite=str(shared["suite"]),
            task_id=str(shared["task_id"]),
            task_name=str(shared["task_name"]),
            demonstration_id=str(row["demonstration_id"]),
            episode_index=int(row["episode_index"]),
            frame_index=int(shared["frame_index"]),
            timestamp_ns=int(shared["timestamp_ns"]),
            camera_keys=tuple(str(value) for value in shared["camera_keys"]),
            language=str(shared["language"]),
            checkpoint_sha256=_sha256(
                shared["checkpoint_sha256"],
                "cohort.checkpoint_sha256",
            ),
            stats_sha256=_sha256(
                shared["stats_sha256"], "cohort.stats_sha256"
            ),
            sampler_config_sha256=_sha256(
                shared["sampler_config_sha256"],
                "cohort.sampler_config_sha256",
            ),
            preprocessing_sha256=_sha256(
                shared["preprocessing_sha256"],
                "cohort.preprocessing_sha256",
            ),
            split_manifest_sha256=_sha256(
                shared["split_manifest_sha256"],
                "cohort.split_manifest_sha256",
            ),
        )
        expected_base_id = _sha256(
            row["base_sample_id"], "cohort.base_sample_id"
        )
        if identity.base_sample_id != expected_base_id:
            raise OnlineCounterfactualError(
                "cohort base_sample_id does not match its frozen identity"
            )
        samples.append(
            OnlineCohortSample(
                identity=identity,
                split=str(row["split"]),
            )
        )
    cfg = K1OnlineCounterfactualConfig(
        source_path=source_path,
        raw=root,
        experiment_name=str(experiment["name"]),
        output_dir=Path(str(experiment["output_dir"])),
        experiment_seed=int(experiment["seed"]),
        device=str(runtime["device"]),
        action_horizon=int(runtime["action_horizon"]),
        action_denoise_steps=int(runtime["action_denoise_steps"]),
        future_k=int(runtime["future_k"]),
        future_shift=float(runtime["future_shift"]),
        future_num_train_timesteps=int(
            runtime["future_num_train_timesteps"]
        ),
        rand_device=str(runtime["rand_device"]),
        max_gpu_memory_gb=float(runtime["max_gpu_memory_gb"]),
        warmup_b0_calls=int(runtime["warmup_b0_calls"]),
        warmup_future_action_calls=int(
            runtime["warmup_future_action_calls"]
        ),
        thought3_config_path=Path(str(source["thought3_config_path"])),
        thought3_config_sha256=_sha256(
            source["thought3_config_sha256"],
            "source.thought3_config_sha256",
        ),
        thought3_config_fingerprint=_sha256(
            source["thought3_config_fingerprint"],
            "source.thought3_config_fingerprint",
        ),
        e6_gate_path=Path(str(source["e6_gate_path"])),
        e6_gate_sha256=_sha256(
            source["e6_gate_sha256"], "source.e6_gate_sha256"
        ),
        e6_checkpoint_dir=Path(str(source["e6_checkpoint_dir"])),
        e6_adapter_sha256=_sha256(
            source["e6_adapter_sha256"],
            "source.e6_adapter_sha256",
        ),
        e6_checkpoint_manifest_sha256=_sha256(
            source["e6_checkpoint_manifest_sha256"],
            "source.e6_checkpoint_manifest_sha256",
        ),
        e6_adapter_state_sha256=_sha256(
            source["e6_adapter_state_sha256"],
            "source.e6_adapter_state_sha256",
        ),
        e6_checkpoint_config_fingerprint=_sha256(
            source["e6_checkpoint_config_fingerprint"],
            "source.e6_checkpoint_config_fingerprint",
        ),
        e6_adapter_fingerprint=_sha256(
            source["e6_adapter_fingerprint"],
            "source.e6_adapter_fingerprint",
        ),
        backbone_checkpoint_sha256=_sha256(
            source["backbone_checkpoint_sha256"],
            "source.backbone_checkpoint_sha256",
        ),
        dataset_stats_sha256=_sha256(
            source["dataset_stats_sha256"],
            "source.dataset_stats_sha256",
        ),
        split_fingerprint=_sha256(
            source["split_fingerprint"], "source.split_fingerprint"
        ),
        fastwam_commit=_git_sha(
            source["fastwam_commit"], "source.fastwam_commit"
        ),
        frozen_parameter_sha256=_sha256(
            source["frozen_parameter_sha256"],
            "source.frozen_parameter_sha256",
        ),
        checkpoint_selection_disclosure=str(
            source["checkpoint_selection_disclosure"]
        ),
        action_seed_namespace=str(seeds["action_namespace"]),
        future_seed_namespace=str(seeds["future_namespace"]),
        shuffle_namespace=str(seeds["shuffle_namespace"]),
        cohort=tuple(samples),
        cohort_selection=str(cohort_raw["selection"]),
        shuffle_seed=int(shuffle["seed"]),
        expected_shuffle_mapping_sha256=_sha256(
            shuffle["expected_mapping_sha256"],
            "shuffle.expected_mapping_sha256",
        ),
        replay_repeats=int(replay["repeats"]),
        replay_hard_max_linf=float(replay["hard_max_linf"]),
        replay_l2_multiplier=float(replay["l2_multiplier"]),
        replay_absolute_l2_floor=float(replay["absolute_l2_floor"]),
        stable_sample_min_count=int(replay["stable_sample_min_count"]),
        action_hash_change_min_count=int(
            replay["action_hash_change_min_count"]
        ),
        null_kind=str(null["kind"]),
    )
    sample_ids = [sample.base_sample_id for sample in cfg.cohort]
    episode_ids = [sample.episode_id for sample in cfg.cohort]
    declared_sample_count = int(cohort_raw["sample_count"])
    if (
        declared_sample_count != 8
        or len(cfg.cohort) != declared_sample_count
        or len(set(sample_ids)) != 8
        or len(set(episode_ids)) != 8
        or any(sample.split != "train" for sample in cfg.cohort)
        or runtime["backend"] != "fastwam"
        or cfg.device != "cuda:0"
        or cfg.action_horizon != 32
        or cfg.action_denoise_steps != 20
        or cfg.future_k != 1
        or cfg.future_shift != 5.0
        or cfg.future_num_train_timesteps != 1000
        or cfg.rand_device != "cpu"
        or cfg.max_gpu_memory_gb != 23.8
        or cfg.warmup_b0_calls != 1
        or cfg.warmup_future_action_calls != 1
        or cfg.experiment_seed != 3407
        or cfg.replay_repeats != 2
        or cfg.replay_hard_max_linf != 1e-5
        or cfg.replay_l2_multiplier != 10.0
        or cfg.replay_absolute_l2_floor != 1e-7
        or cfg.stable_sample_min_count != 6
        or cfg.action_hash_change_min_count != 6
        or cfg.null_kind
        != "formal_parameter_free_injection_null_mask_v1"
        or null.get("tensor_substitute") is not False
        or null.get("run_video_dit") is not False
        or shuffle.get("require_other_episode") is not True
        or shuffle.get("require_one_to_one") is not True
        or shuffle.get("require_same_task") is not True
        or shuffle.get("reuse_recipient_future_noise_seed") is not True
    ):
        raise OnlineCounterfactualError(
            "frozen K=1 online counterfactual design changed"
        )
    manifest = build_episode_derangement(cfg)
    if manifest["fingerprint"] != cfg.expected_shuffle_mapping_sha256:
        raise OnlineCounterfactualError(
            "deterministic shuffle mapping fingerprint changed"
        )
    return cfg


def stable_online_seed(namespace: str, seed: int, sample_id: str) -> int:
    if seed < 0:
        raise OnlineCounterfactualError("seed must be non-negative")
    digest = hashlib.sha256(
        f"{namespace}\0{seed}\0{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_episode_derangement(
    cfg: K1OnlineCounterfactualConfig,
) -> dict[str, Any]:
    """Build the frozen one-to-one other-episode permutation."""

    ordered = sorted(
        cfg.cohort,
        key=lambda sample: hashlib.sha256(
            (
                f"{cfg.shuffle_namespace}\0{cfg.shuffle_seed}\0"
                f"{sample.base_sample_id}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    donor_by_target = {
        sample.base_sample_id: ordered[(index + 1) % len(ordered)]
        for index, sample in enumerate(ordered)
    }
    mapping = [
        {
            "target_base_sample_id": sample.base_sample_id,
            "donor_base_sample_id": donor_by_target[
                sample.base_sample_id
            ].base_sample_id,
        }
        for sample in cfg.cohort
    ]
    targets = {sample.base_sample_id: sample for sample in cfg.cohort}
    if (
        {row["target_base_sample_id"] for row in mapping} != set(targets)
        or {row["donor_base_sample_id"] for row in mapping} != set(targets)
        or any(
            row["target_base_sample_id"] == row["donor_base_sample_id"]
            or targets[row["target_base_sample_id"]].episode_id
            == targets[row["donor_base_sample_id"]].episode_id
            for row in mapping
        )
    ):
        raise OnlineCounterfactualError(
            "cannot construct one-to-one other-episode derangement"
        )
    payload = {
        "schema_version": ONLINE_CF_SHUFFLE_SCHEMA,
        "seed": cfg.shuffle_seed,
        "mapping": mapping,
    }
    return {
        **payload,
        "fingerprint": hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }


def action_sha256(action: Any) -> str:
    import torch

    value = action.detach().cpu().contiguous()
    # Keep the implementation torch-version agnostic while hashing exact bytes.
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _component_l2(delta: Any, start: int, stop: int) -> float:
    if stop <= start:
        return 0.0
    return float(
        delta[..., start:stop]
        .square()
        .sum(dim=-1)
        .sqrt()
        .mean()
        .cpu()
    )


def action_pair_metrics(reference: Any, intervention: Any) -> dict[str, Any]:
    """Compare two [horizon,7] policy chunks without denormalizing them."""

    import torch

    if (
        reference.shape != intervention.shape
        or reference.ndim != 2
        or reference.shape[-1] != 7
    ):
        raise OnlineCounterfactualError(
            "action chunks must share [horizon,7] shape"
        )
    first = reference.detach().float().cpu().contiguous()
    second = intervention.detach().float().cpu().contiguous()
    finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
    delta = second - first
    flat_first = first.reshape(1, -1)
    flat_second = second.reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_first, flat_second, dim=-1, eps=1e-12
    )
    return {
        "action_cosine": float(cosine.reshape(())),
        "finite": finite,
        "gripper_difference": float(delta[:, 6].abs().mean()),
        "l1": float(delta.abs().mean()),
        "l2": float(delta.square().mean().sqrt()),
        "linf": float(delta.abs().max()),
        "per_timestep_l2": [
            float(value)
            for value in delta.square().sum(dim=-1).sqrt().tolist()
        ],
        "reference_action_sha256": action_sha256(first),
        "rotation_difference": _component_l2(delta, 3, 6),
        "translation_difference": _component_l2(delta, 0, 3),
        "intervention_action_sha256": action_sha256(second),
        "eef_trajectory_difference": {
            "status": "unavailable",
            "reason": (
                "policy chunks are normalized actions and no verified "
                "LIBERO forward-kinematics mapping is available"
            ),
        },
    }


def delta_direction_cosine(
    *,
    correct: Any,
    null: Any,
    shuffle: Any,
) -> float | None:
    import torch

    first = (correct.detach().float() - null.detach().float()).reshape(1, -1)
    second = (
        correct.detach().float() - shuffle.detach().float()
    ).reshape(1, -1)
    if float(first.norm()) <= 1e-12 or float(second.norm()) <= 1e-12:
        return None
    return float(
        torch.nn.functional.cosine_similarity(
            first, second, dim=-1, eps=1e-12
        ).reshape(())
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise OnlineCounterfactualError("cannot summarize empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise OnlineCounterfactualError(
            "summary values must be nonempty and finite"
        )
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized),
        "median": statistics.median(normalized),
        "p50": _quantile(normalized, 0.50),
        "p95": _quantile(normalized, 0.95),
    }


def compute_replay_floor(
    replay_rows: Sequence[Mapping[str, Any]],
    cfg: K1OnlineCounterfactualConfig,
) -> dict[str, Any]:
    if len(replay_rows) != len(cfg.cohort):
        raise OnlineCounterfactualError("replay rows do not cover the cohort")
    l2 = [float(row["metrics"]["l2"]) for row in replay_rows]
    linf = [float(row["metrics"]["linf"]) for row in replay_rows]
    finite = all(bool(row["metrics"]["finite"]) for row in replay_rows)
    p95 = _quantile(l2, 0.95)
    threshold = max(
        cfg.replay_absolute_l2_floor,
        cfg.replay_l2_multiplier * p95,
    )
    return {
        "definition_frozen_before_interventions": True,
        "finite": finite,
        "hard_max_linf": cfg.replay_hard_max_linf,
        "hard_passed": finite and max(linf) <= cfg.replay_hard_max_linf,
        "l2": summarize_values(l2),
        "linf": summarize_values(linf),
        "material_l2_threshold": threshold,
        "multiplier": cfg.replay_l2_multiplier,
        "absolute_l2_floor": cfg.replay_absolute_l2_floor,
    }


def classify_online_action_sensitivity(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    replay_floor: Mapping[str, Any],
    cfg: K1OnlineCounterfactualConfig,
) -> dict[str, Any]:
    """Apply the frozen A/B/C decision rule after replay passes."""

    if not bool(replay_floor["hard_passed"]):
        raise OnlineCounterfactualError(
            "replay floor failed; sensitivity classification is forbidden"
        )
    if len(sample_rows) != len(cfg.cohort):
        raise OnlineCounterfactualError(
            "condition rows do not cover the frozen cohort"
        )
    null_parity_linf = [
        float(row["pairs"]["b0_null"]["linf"]) for row in sample_rows
    ]
    if (
        any(not math.isfinite(value) for value in null_parity_linf)
        or max(null_parity_linf) > cfg.replay_hard_max_linf
    ):
        raise OnlineCounterfactualError(
            "formal null failed B0 parity; sensitivity classification is "
            "forbidden"
        )
    threshold = float(replay_floor["material_l2_threshold"])
    correct_null = [
        float(row["pairs"]["correct_null"]["l2"]) for row in sample_rows
    ]
    correct_shuffle = [
        float(row["pairs"]["correct_shuffle"]["l2"])
        for row in sample_rows
    ]
    correct_null_exceeds = sum(value > threshold for value in correct_null)
    correct_shuffle_exceeds = sum(
        value > threshold for value in correct_shuffle
    )
    shuffle_hash_changes = sum(
        row["actions"]["correct"]["sha256"]
        != row["actions"]["shuffle"]["sha256"]
        for row in sample_rows
    )
    if (
        correct_null_exceeds >= cfg.stable_sample_min_count
        and correct_shuffle_exceeds >= cfg.stable_sample_min_count
        and shuffle_hash_changes >= cfg.action_hash_change_min_count
    ):
        classification = ONLINE_CF_CLASSIFICATIONS[0]
        next_branch = "A"
    elif correct_null_exceeds >= cfg.stable_sample_min_count:
        classification = ONLINE_CF_CLASSIFICATIONS[1]
        next_branch = "B"
    else:
        classification = ONLINE_CF_CLASSIFICATIONS[2]
        next_branch = "C"
    directions = [
        float(row["correct_null_vs_correct_shuffle_delta_cosine"])
        for row in sample_rows
        if row.get("correct_null_vs_correct_shuffle_delta_cosine")
        is not None
    ]
    return {
        "schema_version": ONLINE_CF_DECISION_SCHEMA,
        "classification": classification,
        "next_branch": next_branch,
        "thresholds": {
            "action_hash_change_min_count": (
                cfg.action_hash_change_min_count
            ),
            "material_l2_threshold": threshold,
            "stable_sample_min_count": cfg.stable_sample_min_count,
        },
        "values": {
            "b0_null_hard_parity_passed": True,
            "b0_null_linf": summarize_values(null_parity_linf),
            "correct_null_exceeds_replay_floor": correct_null_exceeds,
            "correct_shuffle_exceeds_replay_floor": (
                correct_shuffle_exceeds
            ),
            "correct_vs_shuffle_action_hash_changes": shuffle_hash_changes,
            "direction_consistency": (
                summarize_values(directions) if directions else None
            ),
        },
        "claim_boundary": (
            "technical action sensitivity only; no rollout, success, ID, "
            "OOD, or K-comparison claim"
        ),
    }


def aggregate_online_counterfactual(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    replay_floor: Mapping[str, Any],
    cfg: K1OnlineCounterfactualConfig,
) -> dict[str, Any]:
    decision = classify_online_action_sensitivity(
        sample_rows,
        replay_floor=replay_floor,
        cfg=cfg,
    )
    pairs: dict[str, Any] = {}
    for pair in PAIR_KEYS:
        pairs[pair] = {
            metric: summarize_values(
                [
                    float(row["pairs"][pair][metric])
                    for row in sample_rows
                ]
            )
            for metric in (
                "l1",
                "l2",
                "linf",
                "action_cosine",
                "translation_difference",
                "rotation_difference",
                "gripper_difference",
            )
        }
    latency_keys = (
        "preprocessing_ms",
        "context_construction_ms",
        "current_encoding_ms",
        "future_video_dit_ms",
        "action_context_cache_ms",
        "adapter_ms",
        "action_dit_ms",
        "condition_total_ms",
        "policy_total_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    latency = {
        condition: {
            key: summarize_values(
                [
                    float(row["latency"][condition][key])
                    for row in sample_rows
                ]
            )
            for key in latency_keys
        }
        for condition in ONLINE_CF_CONDITIONS
    }
    hash_changes = {
        "correct_vs_null": sum(
            row["actions"]["correct"]["sha256"]
            != row["actions"]["null"]["sha256"]
            for row in sample_rows
        ),
        "correct_vs_shuffle": sum(
            row["actions"]["correct"]["sha256"]
            != row["actions"]["shuffle"]["sha256"]
            for row in sample_rows
        ),
        "b0_vs_null": sum(
            row["actions"]["B0"]["sha256"]
            != row["actions"]["null"]["sha256"]
            for row in sample_rows
        ),
    }
    return {
        "schema_version": ONLINE_CF_AGGREGATE_SCHEMA,
        "cohort_fingerprint": cfg.cohort_fingerprint,
        "condition_count": len(ONLINE_CF_CONDITIONS),
        "decision": decision,
        "hash_change_sample_counts": hash_changes,
        "latency": latency,
        "pair_metrics": pairs,
        "replay_floor": dict(replay_floor),
        "sample_count": len(sample_rows),
    }


def validate_online_sample_result(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != ONLINE_CF_SAMPLE_SCHEMA:
        raise OnlineCounterfactualError("sample result schema mismatch")
    if set(row.get("actions", {})) != set(ONLINE_CF_CONDITIONS):
        raise OnlineCounterfactualError("sample result action conditions mismatch")
    if set(row.get("pairs", {})) != set(PAIR_KEYS):
        raise OnlineCounterfactualError("sample result pair set mismatch")
    for condition in ONLINE_CF_CONDITIONS:
        action = row["actions"][condition]
        if (
            not action.get("finite")
            or len(action.get("tensor", ())) != 32
            or any(len(step) != 7 for step in action.get("tensor", ()))
        ):
            raise OnlineCounterfactualError(
                f"invalid {condition} action tensor"
            )
        import torch

        tensor = torch.tensor(action["tensor"], dtype=torch.float32)
        if action.get("sha256") != action_sha256(tensor):
            raise OnlineCounterfactualError(
                f"{condition} action SHA-256 does not match its tensor"
            )
    for pair in PAIR_KEYS:
        if not row["pairs"][pair].get("finite"):
            raise OnlineCounterfactualError(
                f"non-finite pair metric: {pair}"
            )
    input_hash_keys = {
        "context_mask_sha256",
        "context_sha256",
        "current_latent_sha256",
        "current_rgb_sha256",
        "proprio_sha256",
    }
    for section in (
        "target_input_identity",
        "shuffle_donor_input_identity",
    ):
        identity = _mapping(row.get(section), section)
        observed_keys = set(identity)
        if section == "shuffle_donor_input_identity":
            observed_keys.discard("base_sample_id")
        if observed_keys != input_hash_keys or any(
            len(str(identity[key])) != 64 for key in input_hash_keys
        ):
            raise OnlineCounterfactualError(
                f"{section} tensor hashes are incomplete"
            )
