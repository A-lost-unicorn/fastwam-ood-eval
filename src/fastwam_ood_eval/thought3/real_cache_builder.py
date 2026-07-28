"""Phase D real Fast-WAM cache builder.

The builder deliberately exposes only a current-observation loader.  It decodes
one timestamp from each of the two LIBERO camera videos, reproduces the upstream
training preprocessing for that current frame, and never requests a later RGB
frame or action target.
"""

from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from fastwam_ood_eval.thought3 import (
    THOUGHT3_CACHE_SCHEMA,
    THOUGHT3_CACHE_SHARD_SCHEMA,
)
from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.future_cache import (
    CacheValidationError,
    atomic_save_safetensors,
    shard_paths,
    validate_cache_shard,
)
from fastwam_ood_eval.thought3.future_sampler import (
    VideoOnlyFutureSampler,
    tensor_sha256,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    sha256_file,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_standard_training_source,
    ensure_thought3_output_path,
)
from fastwam_ood_eval.thought3.schemas import (
    FUTURE_SOURCE_KIND,
    CachePlanEntry,
    FutureLatentRecord,
)


REAL_BUILD_REPORT = "real_cache_build_report.json"


class RealCacheBuildError(RuntimeError):
    """Raised when a Phase D real-cache hard invariant fails."""


@dataclass(frozen=True)
class CurrentObservation:
    image: torch.Tensor
    proprio: torch.Tensor
    source: Mapping[str, Any]


@dataclass(frozen=True)
class GeneratedCacheValue:
    latent: torch.Tensor
    mask: torch.Tensor
    record: FutureLatentRecord
    telemetry: Mapping[str, Any]


def _scalar(value: Any) -> Any:
    if hasattr(value, "numel") and int(value.numel()) == 1:
        return value.item()
    return value


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "max": None,
            "mean": None,
            "min": None,
        }
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "max": max(normalized),
        "mean": sum(normalized) / len(normalized),
        "min": min(normalized),
    }


def _nearest_existing_parent(path: Path) -> Path:
    value = path
    while not value.exists():
        if value.parent == value:
            return value
        value = value.parent
    return value


def _check_disk_budget(
    root: Path,
    *,
    estimated_bytes: int,
    reserve_fraction: float,
) -> None:
    usage = shutil.disk_usage(_nearest_existing_parent(root))
    usable = usage.free * (1.0 - reserve_fraction)
    if estimated_bytes > usable:
        raise RealCacheBuildError(
            "insufficient real-cache disk budget: "
            f"need~{estimated_bytes} bytes, usable={int(usable)} bytes "
            f"after reserve_fraction={reserve_fraction}"
        )


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _verify_frozen_inputs(cfg: Thought3Config) -> dict[str, Any]:
    checkpoint = cfg.backbone.checkpoint_path
    stats = cfg.backbone.dataset_stats_path
    model_config = cfg.backbone.model_config_path
    if checkpoint is None or stats is None or model_config is None:
        raise RealCacheBuildError(
            "real cache requires checkpoint, dataset stats and model config paths"
        )
    expected = (
        (checkpoint, cfg.backbone.checkpoint_sha256, "checkpoint"),
        (stats, cfg.backbone.dataset_stats_sha256, "dataset stats"),
        (model_config, cfg.backbone.model_config_sha256, "model config"),
    )
    observed: dict[str, Any] = {}
    for path, digest, label in expected:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise RealCacheBuildError(
                f"{label} SHA-256 mismatch: expected={digest}, actual={actual}"
            )
        observed[label] = {"path": str(path), "sha256": actual}
    fastwam_root = Path("third_party/FastWAM").resolve()
    actual_commit = _git_head(fastwam_root)
    if actual_commit != cfg.backbone.fastwam_commit:
        raise RealCacheBuildError(
            "Fast-WAM commit mismatch: "
            f"expected={cfg.backbone.fastwam_commit}, actual={actual_commit}"
        )
    observed["fastwam"] = {
        "commit": actual_commit,
        "path": str(fastwam_root),
    }
    return observed


def preprocess_current_camera_frames(
    frames: Mapping[str, torch.Tensor],
    *,
    processor: Any,
    robot_video_dataset: Any,
) -> torch.Tensor:
    """Reproduce the official current-frame training transform without a future window."""

    processed: list[torch.Tensor] = []
    transforms = processor.val_transforms
    for meta in processor.shape_meta["images"]:
        camera_key = str(meta["key"])
        lerobot_key = f"observation.images.{camera_key}"
        if lerobot_key not in frames:
            raise RealCacheBuildError(
                f"current-only decode missing camera {lerobot_key}"
            )
        frame = frames[lerobot_key]
        if tuple(frame.shape) != (3, 512, 512):
            raise RealCacheBuildError(
                f"{lerobot_key} must be [3,512,512], got {tuple(frame.shape)}"
            )
        # Match BaseLerobotDataset._get_image exactly: float [0,1] -> uint8.
        image = (frame * 255).to(torch.uint8).unsqueeze(0)
        current_transforms = (
            transforms[camera_key]
            if isinstance(transforms, dict)
            else transforms
        )
        for transform in current_transforms:
            image = transform(image)
        expected = (1, *tuple(int(value) for value in meta["shape"]))
        if tuple(image.shape) != expected:
            raise RealCacheBuildError(
                f"{camera_key} transform expected {expected}, "
                f"got {tuple(image.shape)}"
            )
        processed.append(image)

    if len(processed) != 2:
        raise RealCacheBuildError("Phase D requires exactly two current cameras")
    video = torch.cat(processed, dim=-1)
    video = robot_video_dataset.resize_transform(video)
    video = robot_video_dataset.crop_transform(video)
    video = robot_video_dataset.normalize_transform(video)
    if tuple(video.shape) != (1, 3, 224, 448):
        raise RealCacheBuildError(
            "current-only model image must be [1,3,224,448], "
            f"got {tuple(video.shape)}"
        )
    if not video.isfinite().all():
        raise RealCacheBuildError("current-only model image contains NaN/Inf")
    return video.contiguous()


def preprocess_current_proprio(
    raw_state: torch.Tensor,
    *,
    processor: Any,
) -> torch.Tensor:
    """Apply the official LIBERO state transform/normalizer to one current state."""

    state_meta = list(processor.shape_meta["state"])
    if len(state_meta) != 1:
        raise RealCacheBuildError("Phase D expects one merged proprio state")
    state_key = str(state_meta[0]["key"])
    state = raw_state.detach().float()
    if tuple(state.shape) != (8,):
        raise RealCacheBuildError(
            f"raw current proprio must be [8], got {tuple(state.shape)}"
        )
    batch: dict[str, Any] = {
        "state": {state_key: state.unsqueeze(0)}
    }
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    batch = processor.action_state_merger.forward(batch)
    proprio = batch["state"]
    if tuple(proprio.shape) != (1, 8) or not proprio.isfinite().all():
        raise RealCacheBuildError(
            "normalized current proprio must be finite [1,8]"
        )
    return proprio.contiguous()


class CurrentOnlyLiberoSource:
    """Read exactly two current RGB frames and one current proprio vector."""

    def __init__(self, cfg: Thought3Config, upstream_cfg: Any) -> None:
        from hydra.utils import instantiate
        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )

        if len(cfg.data.dataset_roots) != 1:
            raise RealCacheBuildError(
                "Phase D current source accepts exactly one dataset root"
            )
        root = ensure_standard_training_source(cfg.data.dataset_roots[0])
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.cfg = cfg
        self.root = root
        self.dataset = instantiate(
            upstream_cfg.data.train,
            dataset_dirs=[str(root.resolve())],
            processor=None,
            text_embedding_cache_dir=None,
        )
        self.processor = instantiate(upstream_cfg.data.train.processor).eval()
        stats = cfg.backbone.dataset_stats_path
        assert stats is not None
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(stats))
        )
        datasets = self.dataset.lerobot_dataset.multi_dataset._datasets
        if len(datasets) != 1:
            raise RealCacheBuildError("Phase D expected one inner LeRobot dataset")
        self.inner = datasets[0]
        requested_backend = str(self.inner.video_backend)
        # TorchCodec is installed but cannot link FFmpeg in this environment.
        # Pinning PyAV is the same path used by its fallback, without repeated
        # failed dynamic-library loads.
        self.inner.video_backend = "pyav"
        import av
        import torchvision

        self.telemetry: dict[str, Any] = {
            "actual_future_read": False,
            "action_target_read": False,
            "configured_video_backend": requested_backend,
            "current_camera_frames_decoded": 0,
            "decode_backend": "pyav",
            "future_rgb_frames_decoded": 0,
            "pyav_version": str(av.__version__),
            "torchvision_version": str(torchvision.__version__),
        }

    def _row_index(self, episode_index: int, frame_index: int) -> int:
        starts = self.inner.episode_data_index["from"]
        ends = self.inner.episode_data_index["to"]
        if episode_index < 0 or episode_index >= len(starts):
            raise RealCacheBuildError(
                f"episode index out of range: {episode_index}"
            )
        start = int(starts[episode_index].item())
        end = int(ends[episode_index].item())
        index = start + int(frame_index)
        if index < start or index >= end:
            raise RealCacheBuildError(
                f"frame {frame_index} is outside episode {episode_index}"
            )
        return index

    def load(self, entry: CachePlanEntry) -> CurrentObservation:
        identity = entry.identity
        index = self._row_index(identity.episode_index, identity.frame_index)
        row = self.inner.hf_dataset[index]
        observed = {
            "episode_index": int(_scalar(row["episode_index"])),
            "frame_index": int(_scalar(row["frame_index"])),
            "task_index": int(_scalar(row["task_index"])),
            "timestamp_ns": int(
                round(float(_scalar(row["timestamp"])) * 1_000_000_000)
            ),
        }
        expected = {
            "episode_index": identity.episode_index,
            "frame_index": identity.frame_index,
            "task_index": int(identity.task_id.removeprefix("task_")),
            "timestamp_ns": identity.timestamp_ns,
        }
        if observed != expected:
            raise RealCacheBuildError(
                f"inventory/dataset row mismatch: expected={expected}, "
                f"observed={observed}"
            )
        timestamp = float(_scalar(row["timestamp"]))
        query = {
            str(key): [timestamp]
            for key in self.inner.meta.video_keys
        }
        started = time.perf_counter()
        frames = self.inner._query_videos(
            query,
            identity.episode_index,
        )
        decode_ms = (time.perf_counter() - started) * 1000.0
        if set(frames) != set(query):
            raise RealCacheBuildError("current camera decode key mismatch")
        self.telemetry["current_camera_frames_decoded"] += len(frames)
        image = preprocess_current_camera_frames(
            frames,
            processor=self.processor,
            robot_video_dataset=self.dataset,
        )
        proprio = preprocess_current_proprio(
            row["observation.state"],
            processor=self.processor,
        )
        return CurrentObservation(
            image=image,
            proprio=proprio,
            source={
                "current_camera_count": len(frames),
                "current_decode_latency_ms": decode_ms,
                "dataset_index": index,
                "episode_index": identity.episode_index,
                "frame_index": identity.frame_index,
                "future_rgb_frames_decoded": 0,
                "timestamp_ns": identity.timestamp_ns,
            },
        )


class _FastWAMVideoVelocity:
    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        conditions: Mapping[str, object],
    ) -> torch.Tensor:
        return self.model.video_expert(
            x=state,
            timestep=timestep.to(device=state.device, dtype=state.dtype),
            context=conditions["context"],
            context_mask=conditions["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=True,
        )


def _cuda_measure(
    *,
    device: str,
    hard_limit_bytes: int,
    function: Any,
) -> tuple[Any, dict[str, float]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    if peak_bytes >= hard_limit_bytes:
        raise RealCacheBuildError(
            f"CUDA peak {peak_bytes / 2**30:.3f} GiB violates "
            f"hard limit {hard_limit_bytes / 2**30:.3f} GiB"
        )
    return value, {
        "latency_ms": latency_ms,
        "peak_gib": peak_bytes / 2**30,
        "peak_memory_mb": peak_bytes / 2**20,
    }


def _write_real_shard(
    cfg: Thought3Config,
    entries: Sequence[CachePlanEntry],
    values: Sequence[GeneratedCacheValue],
    *,
    cache_fingerprint: str,
) -> dict[str, Any]:
    if not entries or len(entries) != len(values):
        raise RealCacheBuildError("real shard entries/values are empty or misaligned")
    k = entries[0].k
    shard_index = entries[0].shard_index
    if any(
        entry.k != k or entry.shard_index != shard_index
        for entry in entries
    ):
        raise RealCacheBuildError("real shard entries disagree on K/index")
    paths = shard_paths(cfg.cache.root, k, shard_index)
    if paths.manifest.exists():
        raise FileExistsError(paths.manifest)
    partial = [path for path in (paths.tensor, paths.metadata) if path.exists()]
    if partial:
        raise FileExistsError(
            "uncommitted real cache artifacts require manual audit: "
            + ", ".join(str(path) for path in partial)
        )

    latents = torch.stack([value.latent for value in values])
    masks = torch.stack([value.mask for value in values])
    rows: list[dict[str, Any]] = []
    for tensor_index, (entry, value) in enumerate(
        zip(entries, values, strict=True)
    ):
        if value.record.cache_sample_id != entry.cache_sample_id:
            raise RealCacheBuildError("generated record does not match cache plan")
        rows.append(
            {
                "episode_id": entry.identity.episode_id,
                "identity": entry.identity.to_dict(),
                "record": value.record.to_dict(),
                "source_access": {
                    "action_target_read": False,
                    "actual_future_read": False,
                    "current_camera_frames_decoded": 2,
                    "future_rgb_frames_decoded": 0,
                },
                "task_id": entry.identity.task_id,
                "telemetry": dict(value.telemetry),
                "tensor_index": tensor_index,
            }
        )

    paths.tensor.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_safetensors(
        paths.tensor,
        {
            "future_latents": latents,
            "future_masks": masks,
        },
        metadata={
            "backend": "fastwam",
            "cache_fingerprint": cache_fingerprint,
            "current_observation_only": "true",
            "schema_version": THOUGHT3_CACHE_SCHEMA,
            "source_kind": FUTURE_SOURCE_KIND,
        },
    )
    atomic_write_jsonl(paths.metadata, rows)
    manifest: dict[str, Any] = {
        "backend": "fastwam",
        "cache_fingerprint": cache_fingerprint,
        "cache_sample_ids": [
            row["record"]["cache_sample_id"] for row in rows
        ],
        "cache_schema_version": THOUGHT3_CACHE_SCHEMA,
        "current_observation_only": True,
        "future_source_kind": FUTURE_SOURCE_KIND,
        "k": k,
        "metadata_file": paths.metadata.name,
        "metadata_file_sha256": sha256_file(paths.metadata),
        "sample_count": len(entries),
        "schema_version": THOUGHT3_CACHE_SHARD_SCHEMA,
        "shard_index": shard_index,
        "tensor_file": paths.tensor.name,
        "tensor_file_sha256": sha256_file(paths.tensor),
        "tensor_sha256": {
            "future_latents": tensor_sha256(latents),
            "future_masks": tensor_sha256(masks),
        },
        "uses_ground_truth_future": False,
    }
    atomic_write_json(paths.manifest, manifest)
    validate_cache_shard(
        paths,
        expected_cache_fingerprint=cache_fingerprint,
    )
    return manifest


def _inspect_existing_shards(
    cfg: Thought3Config,
    grouped: Mapping[tuple[int, int], Sequence[CachePlanEntry]],
    *,
    cache_fingerprint: str,
    resume: bool,
) -> tuple[list[tuple[int, int]], int]:
    missing: list[tuple[int, int]] = []
    skipped = 0
    for key in sorted(grouped):
        paths = shard_paths(cfg.cache.root, *key)
        if not paths.manifest.exists():
            partial = [
                path for path in (paths.tensor, paths.metadata) if path.exists()
            ]
            if partial:
                raise FileExistsError(
                    "uncommitted real cache artifacts require manual audit: "
                    + ", ".join(str(path) for path in partial)
                )
            missing.append(key)
            continue
        try:
            validate_cache_shard(
                paths,
                expected_cache_fingerprint=cache_fingerprint,
            )
        except CacheValidationError as exc:
            raise CacheValidationError(
                f"resume found corrupt committed real shard "
                f"{paths.manifest}: {exc}"
            ) from exc
        if not resume:
            raise FileExistsError(
                f"real cache shard exists; pass --resume: {paths.manifest}"
            )
        skipped += 1
    return missing, skipped


def _group_entries(
    entries: Iterable[CachePlanEntry],
) -> dict[tuple[int, int], list[CachePlanEntry]]:
    grouped: dict[tuple[int, int], list[CachePlanEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.k, entry.shard_index)].append(entry)
    for values in grouped.values():
        values.sort(key=lambda item: item.identity.base_sample_id)
    return dict(grouped)


def _load_prompt_context(
    model: Any,
    entries: Sequence[CachePlanEntry],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    languages = {entry.identity.language for entry in entries}
    if len(languages) != 1:
        raise RealCacheBuildError(
            "Phase D one-task cache requires exactly one language"
        )
    language = next(iter(languages))
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    prompt = str(DEFAULT_PROMPT).format(task=language)
    context, mask = model.encode_prompt(prompt)
    context = context.detach().to(device=device, dtype=model.torch_dtype)
    mask = mask.detach().to(device=device, dtype=torch.bool)
    if tuple(context.shape) != (1, 128, 4096):
        raise RealCacheBuildError(
            f"prompt context expected [1,128,4096], got {tuple(context.shape)}"
        )
    if tuple(mask.shape) != (1, 128):
        raise RealCacheBuildError(
            f"prompt mask expected [1,128], got {tuple(mask.shape)}"
        )
    return context, mask, prompt


def build_real_cache(
    cfg: Thought3Config,
    *,
    resume: bool,
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build a one-task Phase D cache on one physical GPU."""

    started = time.perf_counter()
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_D") != "YES":
        raise RealCacheBuildError(
            "set CONFIRM_THOUGHT3_PHASE_D=YES for real cache generation"
        )
    if (rank, world_size) != (0, 1):
        raise RealCacheBuildError(
            "Phase D real cache is intentionally single-rank; "
            "multi-GPU cache generation starts after Gate D"
        )
    if device != "cuda:0" or cfg.runtime.device != "cuda:0":
        raise RealCacheBuildError(
            "inside the single-card window Phase D requires logical cuda:0"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RealCacheBuildError(
            "Phase D requires exactly one CUDA-visible physical GPU"
        )
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = ensure_thought3_output_path(cfg.cache.root)
    entries, plan = load_cache_plan(root)
    if plan["config_fingerprint"] != cfg.fingerprint:
        raise RealCacheBuildError("real cache plan/config fingerprint mismatch")
    if int(plan["sample_count"]) != 32 or len(entries) != 96:
        raise RealCacheBuildError(
            "Phase D is frozen to 32 base samples and 96 K entries"
        )
    grouped = _group_entries(entries)
    missing, skipped = _inspect_existing_shards(
        cfg,
        grouped,
        cache_fingerprint=str(plan["cache_fingerprint"]),
        resume=resume,
    )
    if not missing:
        return {
            "assigned_shards": len(grouped),
            "backend": "fastwam",
            "built_shards": 0,
            "cache_fingerprint": plan["cache_fingerprint"],
            "complete": True,
            "model_loaded": False,
            "rank": rank,
            "resume_validation_only": True,
            "skipped_valid_shards": skipped,
            "status": "already_complete",
            "total_shards": len(grouped),
            "world_size": world_size,
        }

    estimated = (
        sum(len(grouped[key]) for key in missing)
        * math.prod(cfg.sampler.latent_shape)
        * (2 if cfg.sampler.cache_dtype == "bfloat16" else 4)
    )
    _check_disk_budget(
        root,
        estimated_bytes=estimated,
        reserve_fraction=cfg.cache.required_free_space_fraction,
    )
    provenance = _verify_frozen_inputs(cfg)
    hard_limit_bytes = int(cfg.runtime.max_gpu_memory_gb * 2**30)

    from fastwam_ood_eval.thought3.phase_c_smoke import _load_upstream_model

    torch.cuda.reset_peak_memory_stats(device)
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize(device)
    model_report = {
        **model_report,
        "load_peak_gib": (
            int(torch.cuda.max_memory_allocated(device)) / 2**30
        ),
    }
    if model_report["load_peak_gib"] >= cfg.runtime.max_gpu_memory_gb:
        raise RealCacheBuildError("model load exceeded Phase D memory gate")
    source = CurrentOnlyLiberoSource(cfg, upstream_cfg)
    context_base, context_mask_base, prompt = _load_prompt_context(
        model,
        entries,
        device=device,
    )
    if model.text_encoder is not None:
        model.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    sampler = VideoOnlyFutureSampler(
        _FastWAMVideoVelocity(model),
        shift=cfg.sampler.shift,
        num_train_timesteps=cfg.sampler.num_train_timesteps,
        rand_device=cfg.sampler.rand_device,
    )
    dtype = (
        torch.bfloat16
        if cfg.sampler.cache_dtype == "bfloat16"
        else torch.float32
    )
    missing_set = set(missing)
    shard_indices = sorted({shard_index for _, shard_index in missing})
    built = 0
    generated_entries = 0
    encoded_base_samples = 0
    decode_latencies: list[float] = []
    encode_latencies: list[float] = []
    sampling_latencies: dict[int, list[float]] = defaultdict(list)
    peak_memory: list[float] = []
    build_started = time.perf_counter()

    with torch.inference_mode():
        for shard_index in shard_indices:
            needed_k = [
                k
                for k in cfg.sampler.cache_k
                if (k, shard_index) in missing_set
            ]
            reference_entries = grouped[(needed_k[0], shard_index)]
            values_by_k: dict[int, list[GeneratedCacheValue]] = {
                k: [] for k in needed_k
            }
            for position, reference in enumerate(reference_entries):
                matching = {
                    k: grouped[(k, shard_index)][position]
                    for k in needed_k
                }
                if any(
                    entry.identity.base_sample_id
                    != reference.identity.base_sample_id
                    for entry in matching.values()
                ):
                    raise RealCacheBuildError(
                        "K shards are not positionally paired by base sample"
                    )
                observation = source.load(reference)
                decode_latencies.append(
                    float(observation.source["current_decode_latency_ms"])
                )
                image = observation.image.to(
                    device=device,
                    dtype=model.torch_dtype,
                )
                current_latent, encode_report = _cuda_measure(
                    device=device,
                    hard_limit_bytes=hard_limit_bytes,
                    function=lambda: model._encode_input_image_latents_tensor(
                        image
                    ),
                )
                if tuple(current_latent.shape) != (1, 48, 1, 14, 28):
                    raise RealCacheBuildError(
                        "current latent must be [1,48,1,14,28]"
                    )
                encode_latencies.append(encode_report["latency_ms"])
                peak_memory.append(encode_report["peak_memory_mb"])
                proprio = observation.proprio.to(
                    device=device,
                    dtype=model.torch_dtype,
                )
                context, context_mask = model._append_proprio_to_context(
                    context_base,
                    context_mask_base,
                    proprio,
                )
                conditions = {
                    "context": context,
                    "context_mask": context_mask,
                }
                initial_hashes: set[str] = set()
                for k in needed_k:
                    entry = matching[k]
                    sample, sampling_report = _cuda_measure(
                        device=device,
                        hard_limit_bytes=hard_limit_bytes,
                        function=lambda entry=entry: sampler.sample(
                            current_latent,
                            initial_noise_seeds=(
                                entry.initial_noise_seed,
                            ),
                            k=entry.k,
                            conditions=conditions,
                        ),
                    )
                    latent = sample.future_latent[0].detach().to(
                        device="cpu",
                        dtype=dtype,
                    ).contiguous()
                    if tuple(latent.shape) != tuple(cfg.sampler.latent_shape):
                        raise RealCacheBuildError(
                            f"K={k} latent shape mismatch: "
                            f"{tuple(latent.shape)}"
                        )
                    if not latent.isfinite().all():
                        raise RealCacheBuildError(
                            f"K={k} future latent contains NaN/Inf"
                        )
                    initial_hash = sample.initial_state_sha256[0]
                    initial_hashes.add(initial_hash)
                    mask = torch.ones(
                        latent.shape[1:],
                        dtype=torch.bool,
                    )
                    record = FutureLatentRecord(
                        base_sample_id=entry.identity.base_sample_id,
                        cache_sample_id=entry.cache_sample_id,
                        split=entry.split,
                        k=entry.k,
                        initial_noise_seed=entry.initial_noise_seed,
                        schedule=sample.schedule,
                        checkpoint_sha256=cfg.backbone.checkpoint_sha256,
                        stats_sha256=cfg.backbone.dataset_stats_sha256,
                        cache_fingerprint=str(plan["cache_fingerprint"]),
                        initial_state_sha256=initial_hash,
                        latent_dtype=cfg.sampler.cache_dtype,
                        latent_sha256=tensor_sha256(latent),
                        generation_latency_ms=sampling_report["latency_ms"],
                        generation_peak_memory_mb=sampling_report[
                            "peak_memory_mb"
                        ],
                        source_kind=FUTURE_SOURCE_KIND,
                        uses_ground_truth_future=False,
                    )
                    values_by_k[k].append(
                        GeneratedCacheValue(
                            latent=latent,
                            mask=mask,
                            record=record,
                            telemetry={
                                **dict(observation.source),
                                "current_encode_latency_ms": encode_report[
                                    "latency_ms"
                                ],
                                "current_latent_sha256": tensor_sha256(
                                    current_latent
                                ),
                                "sampling_latency_ms": sampling_report[
                                    "latency_ms"
                                ],
                                "sampling_peak_memory_mb": sampling_report[
                                    "peak_memory_mb"
                                ],
                            },
                        )
                    )
                    sampling_latencies[k].append(
                        sampling_report["latency_ms"]
                    )
                    peak_memory.append(
                        sampling_report["peak_memory_mb"]
                    )
                    generated_entries += 1
                    del sample
                if len(initial_hashes) != 1:
                    raise RealCacheBuildError(
                        "K values for one base sample did not use paired noise"
                    )
                encoded_base_samples += 1
                del image, current_latent, context, context_mask, proprio
                torch.cuda.empty_cache()

            for k in needed_k:
                _write_real_shard(
                    cfg,
                    grouped[(k, shard_index)],
                    values_by_k[k],
                    cache_fingerprint=str(plan["cache_fingerprint"]),
                )
                built += 1

    generated_wall_s = time.perf_counter() - build_started
    committed = sum(
        shard_paths(root, *key).manifest.is_file()
        for key in grouped
    )
    complete = committed == len(grouped)
    if not complete:
        raise RealCacheBuildError(
            f"real cache incomplete: committed={committed}/{len(grouped)}"
        )
    report: dict[str, Any] = {
        "assigned_shards": len(grouped),
        "backend": "fastwam",
        "base_samples_encoded": encoded_base_samples,
        "built_shards": built,
        "cache_fingerprint": plan["cache_fingerprint"],
        "complete": True,
        "current_decode_latency_ms": _summary(decode_latencies),
        "current_encode_latency_ms": _summary(encode_latencies),
        "current_source": dict(source.telemetry),
        "future_entries_generated": generated_entries,
        "generation_wall_s": generated_wall_s,
        "k_sampling_latency_ms": {
            str(k): _summary(values)
            for k, values in sorted(sampling_latencies.items())
        },
        "max_execution_peak_memory_mb": max(peak_memory),
        "model": model_report,
        "model_loaded": True,
        "prompt": prompt,
        "provenance": provenance,
        "rank": rank,
        "resume_validation_only": False,
        "skipped_valid_shards": skipped,
        "status": "complete",
        "throughput_base_samples_per_s_excluding_model_load": (
            encoded_base_samples / generated_wall_s
        ),
        "total_shards": len(grouped),
        "total_wall_s": time.perf_counter() - started,
        "uses_ground_truth_future": False,
        "world_size": world_size,
    }
    report_path = atomic_write_json(root / REAL_BUILD_REPORT, report)
    cache_manifest = {
        "backend": "fastwam",
        "build_report": report_path.name,
        "build_report_sha256": sha256_file(report_path),
        "cache_fingerprint": plan["cache_fingerprint"],
        "cache_schema_version": THOUGHT3_CACHE_SCHEMA,
        "complete": True,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "current_observation_only": True,
        "plan_manifest_sha256": sha256_file(
            root / "cache_plan_manifest.json"
        ),
        "total_shards": len(grouped),
        "uses_ground_truth_future": False,
        "world_size": world_size,
    }
    atomic_write_json(root / "cache_manifest.json", cache_manifest)
    del sampler, source, model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def load_real_build_report(root: str | Path) -> dict[str, Any]:
    cache_root = ensure_thought3_output_path(root)
    manifest = load_json(cache_root / "cache_manifest.json")
    path = cache_root / str(manifest["build_report"])
    if sha256_file(path) != manifest["build_report_sha256"]:
        raise CacheValidationError("real cache build report checksum mismatch")
    return load_json(path)
