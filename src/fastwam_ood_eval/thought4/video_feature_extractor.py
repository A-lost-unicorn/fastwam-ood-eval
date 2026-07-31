"""Pooling and sharded extraction helpers for source-A Video features."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought4.feature_hooks import (
    FeatureHookError,
    ScopedFeatureCapture,
    ScopedVideoKVCacheCapture,
    validate_layer_indices,
    video_kv_cache_specs,
    video_hook_specs,
)
from fastwam_ood_eval.thought4.io_utils import (
    Thought4ArtifactError,
    atomic_write_text,
    ensure_run_mutable,
)
from fastwam_ood_eval.thought4.schemas import (
    FeatureRecord,
    SampleIdentity,
    sha256_file,
)


POOLING_RULES = (
    "spatial_mean",
    "foreground_mean",
    "robot_object_roi",
    "cls",
)


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def pool_tokens(
    tensor: Any,
    *,
    rule: str,
    token_mask: Any | None = None,
    has_cls_token: bool = False,
) -> Any:
    """Apply one preregistered pooling rule to ``[B,N,C]`` tokens."""

    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise FeatureHookError("pooling expects a [B,N,C] Tensor")
    if not tensor.is_floating_point() or not bool(tensor.isfinite().all().item()):
        raise FeatureHookError("pooling input must be finite floating point")
    if rule == "cls":
        if not has_cls_token:
            raise FeatureHookError("CLS pooling requested but architecture has no CLS token")
        return tensor[:, 0, :].detach()
    if rule == "spatial_mean":
        return tensor.mean(dim=1).detach()
    if rule not in {"foreground_mean", "robot_object_roi"}:
        raise FeatureHookError(f"unknown pooling rule: {rule}")
    if token_mask is None:
        raise FeatureHookError(f"{rule} requires an explicit token mask")
    mask = torch.as_tensor(token_mask, device=tensor.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(tensor.shape[0], -1)
    if mask.shape != tensor.shape[:2]:
        raise FeatureHookError(
            f"token mask shape mismatch: {mask.shape} vs {tensor.shape[:2]}"
        )
    counts = mask.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise FeatureHookError(f"{rule} mask contains an empty sample")
    pooled = (tensor * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
    return pooled.detach()


def build_primary_camera_token_masks(
    *,
    depth: Any,
    eef_position_world: Any,
    object_position_world: Any,
    intrinsic: Any,
    camera_to_world: Any,
    token_grid: tuple[int, int] = (7, 14),
    primary_image_hw: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Build frozen foreground and EEF/object ROI masks for two-camera tokens."""

    import numpy as np
    import torch

    value = np.asarray(depth, dtype=np.float64)
    height, width = primary_image_hw
    rows, total_columns = token_grid
    if value.shape[:2] != (height, width):
        raise FeatureHookError(
            f"depth shape {value.shape} differs from primary image {(height, width)}"
        )
    if total_columns % 2:
        raise FeatureHookError("two-camera token grid must have an even width")
    primary_columns = total_columns // 2
    if height % rows or width % primary_columns:
        raise FeatureHookError("image dimensions do not divide the token grid")
    block_depth = value.reshape(
        rows,
        height // rows,
        primary_columns,
        width // primary_columns,
    ).mean(axis=(1, 3))
    finite = np.isfinite(block_depth) & (block_depth > 0)
    if not finite.any():
        raise FeatureHookError("depth has no finite positive foreground candidates")
    threshold = float(np.quantile(block_depth[finite], 0.80))
    foreground_primary = finite & (block_depth <= threshold)
    foreground = np.zeros((rows, total_columns), dtype=bool)
    foreground[:, :primary_columns] = foreground_primary

    intrinsic_value = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = np.asarray(camera_to_world, dtype=np.float64)
    if intrinsic_value.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise FeatureHookError("invalid camera matrices for ROI pooling")
    projection = np.eye(4, dtype=np.float64)
    projection[:3, :3] = intrinsic_value
    projection = projection @ np.linalg.inv(extrinsic)
    points = np.asarray(
        [eef_position_world, object_position_world], dtype=np.float64
    )
    homogeneous = np.concatenate((points, np.ones((2, 1))), axis=1)
    projected = (projection @ homogeneous.T).T
    if np.any(projected[:, 2] <= 1e-8):
        raise FeatureHookError("EEF/object point is behind the primary camera")
    pixels_xy = projected[:, :2] / projected[:, 2:3]
    roi = np.zeros((rows, total_columns), dtype=bool)
    for x, y in pixels_xy:
        row = int(np.clip(np.floor(y / (height / rows)), 0, rows - 1))
        column = int(
            np.clip(
                np.floor(x / (width / primary_columns)),
                0,
                primary_columns - 1,
            )
        )
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                candidate_row = row + row_offset
                candidate_column = column + column_offset
                if (
                    0 <= candidate_row < rows
                    and 0 <= candidate_column < primary_columns
                ):
                    roi[candidate_row, candidate_column] = True
    if not roi.any():
        raise FeatureHookError("robot/object ROI token mask is empty")
    return {
        "foreground_mean": torch.from_numpy(foreground.reshape(-1)),
        "robot_object_roi": torch.from_numpy(roi.reshape(-1)),
    }


@dataclass(frozen=True)
class ExtractedFeature:
    identity: SampleIdentity
    condition: str
    source: str
    module_path: str
    layer_index: int | None
    denoise_step_index: int | None
    pooling: str
    tensor: Any


class VideoFeatureExtractor:
    """Capture only modules that MoT actually calls during Video prefill."""

    def __init__(
        self,
        model: Any,
        layers: Sequence[int],
        *,
        expected_action_calls: int = 20,
    ) -> None:
        validate_layer_indices(model.video_expert, layers, "Video DiT")
        self.model = model
        self.layers = tuple(int(value) for value in layers)
        self.specs = video_hook_specs(self.layers, include_kv=False)
        self.cache_specs = video_kv_cache_specs(
            self.layers, expected_calls=expected_action_calls
        )

    def capture(self, forward: Any) -> dict[str, list[Any]]:
        with ScopedFeatureCapture(
            self.model, self.specs, clone=True, to_cpu=False
        ) as capture, ScopedVideoKVCacheCapture(
            self.model.mot, self.cache_specs, to_cpu=False
        ) as cache_capture:
            forward()
        return {**capture.captured, **cache_capture.captured}

    @property
    def manifest(self) -> tuple[dict[str, Any], ...]:
        module_rows = tuple(
            {
                "source": "A",
                "name": spec.name,
                "module_path": spec.module_path,
                "location": spec.location,
            }
            for spec in self.specs
        )
        cache_rows = tuple(
            {
                "source": "A",
                "name": spec.name,
                "module_path": spec.module_path,
                "location": "forward_action_with_video_cache argument",
            }
            for spec in self.cache_specs
        )
        return (*module_rows, *cache_rows)


class FeatureShardWriter:
    """Write feature shards atomically with per-tensor and per-shard checksums."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        source: str,
        shard_index: int,
        resume: bool = False,
    ) -> None:
        if source not in {"A", "B"}:
            raise Thought4ArtifactError("feature source must be A or B")
        self.run_dir = ensure_run_mutable(run_dir)
        self.source = source
        self.shard_index = int(shard_index)
        if self.shard_index < 0:
            raise Thought4ArtifactError("shard_index must be non-negative")
        directory = self.run_dir / "feature_shards" / f"source_{source.lower()}"
        self.path = directory / f"shard_{self.shard_index:05d}.pt"
        self.checksum_path = self.path.with_suffix(".sha256")
        self.resume = resume

    def is_complete(self) -> bool:
        if not self.path.is_file() or not self.checksum_path.is_file():
            return False
        expected = self.checksum_path.read_text(encoding="utf-8").strip()
        return len(expected) == 64 and sha256_file(self.path) == expected

    def write(self, features: Sequence[ExtractedFeature]) -> list[FeatureRecord]:
        import torch

        if self.is_complete():
            if self.resume:
                loaded, records = read_feature_shard(
                    self.path, expected_source=self.source
                )
                _validate_reused_features(features, loaded)
                return records
            raise Thought4ArtifactError(f"feature shard already exists: {self.path}")
        if self.path.exists() or self.checksum_path.exists():
            raise Thought4ArtifactError(
                f"incomplete feature shard exists; inspect before retry: {self.path}"
            )
        if not features:
            raise Thought4ArtifactError("cannot write an empty feature shard")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = []
        for feature in features:
            tensor = feature.tensor.detach().cpu().contiguous()
            if tensor.requires_grad:
                raise FeatureHookError("feature tensor was not detached")
            if not bool(tensor.isfinite().all().item()):
                raise FeatureHookError("feature tensor contains NaN/Inf")
            payload.append(
                {
                    "identity": {
                        "task_id": feature.identity.task_id,
                        "episode_id": feature.identity.episode_id,
                        "frame_index": feature.identity.frame_index,
                        "split": feature.identity.split,
                        "timestamp": feature.identity.timestamp,
                        "label_identity": feature.identity.label_identity,
                    },
                    "condition": feature.condition,
                    "source": feature.source,
                    "module_path": feature.module_path,
                    "layer_index": feature.layer_index,
                    "denoise_step_index": feature.denoise_step_index,
                    "pooling": feature.pooling,
                    "tensor": tensor,
                    "feature_sha256": tensor_sha256(tensor),
                }
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(descriptor)
        temporary = Path(temp_name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        shard_sha = sha256_file(self.path)
        atomic_write_text(self.checksum_path, shard_sha + "\n")
        records: list[FeatureRecord] = []
        for row, feature in zip(payload, features):
            tensor = row["tensor"]
            records.append(
                FeatureRecord(
                    identity=feature.identity,
                    condition=row["condition"],
                    source=row["source"],
                    module_path=row["module_path"],
                    layer_index=row["layer_index"],
                    denoise_step_index=row["denoise_step_index"],
                    pooling=row["pooling"],
                    tensor_shape=tuple(int(value) for value in tensor.shape),
                    tensor_dtype=str(tensor.dtype),
                    feature_sha256=row["feature_sha256"],
                    shard_path=str(self.path),
                    shard_sha256=shard_sha,
                )
            )
        return records


def _feature_record(
    *,
    feature: ExtractedFeature,
    tensor: Any,
    shard_path: Path,
    shard_sha256: str,
) -> FeatureRecord:
    return FeatureRecord(
        identity=feature.identity,
        condition=feature.condition,
        source=feature.source,
        module_path=feature.module_path,
        layer_index=feature.layer_index,
        denoise_step_index=feature.denoise_step_index,
        pooling=feature.pooling,
        tensor_shape=tuple(int(value) for value in tensor.shape),
        tensor_dtype=str(tensor.dtype),
        feature_sha256=tensor_sha256(tensor),
        shard_path=str(shard_path),
        shard_sha256=shard_sha256,
    )


def read_feature_shard(
    path: str | Path,
    *,
    expected_source: str | None = None,
) -> tuple[list[ExtractedFeature], list[FeatureRecord]]:
    """Load a completed shard only after validating both checksum levels."""

    import torch

    shard_path = Path(path)
    checksum_path = shard_path.with_suffix(".sha256")
    if not shard_path.is_file() or not checksum_path.is_file():
        raise Thought4ArtifactError(
            f"feature shard or checksum sidecar is missing: {shard_path}"
        )
    expected_shard_sha = checksum_path.read_text(encoding="utf-8").strip()
    if len(expected_shard_sha) != 64:
        raise Thought4ArtifactError(
            f"invalid feature shard checksum sidecar: {checksum_path}"
        )
    observed_shard_sha = sha256_file(shard_path)
    if observed_shard_sha != expected_shard_sha:
        raise Thought4ArtifactError(
            f"feature shard checksum mismatch: {shard_path}"
        )
    try:
        payload = torch.load(
            shard_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise Thought4ArtifactError(
            f"feature shard cannot be loaded safely: {shard_path}"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise Thought4ArtifactError("feature shard payload must be a non-empty list")
    features: list[ExtractedFeature] = []
    records: list[FeatureRecord] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise Thought4ArtifactError(
                f"feature shard row {index} is not a mapping"
            )
        try:
            identity_row = row["identity"]
            if not isinstance(identity_row, Mapping):
                raise TypeError("identity is not a mapping")
            identity = SampleIdentity(
                task_id=str(identity_row["task_id"]),
                episode_id=str(identity_row["episode_id"]),
                frame_index=int(identity_row["frame_index"]),
                split=str(identity_row["split"]),
                timestamp=float(identity_row["timestamp"]),
                label_identity=str(identity_row["label_identity"]),
            )
            tensor = row["tensor"]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("tensor is not a Tensor")
            tensor = tensor.detach().cpu().contiguous()
            if tensor.requires_grad or not tensor.is_floating_point():
                raise TypeError("tensor must be detached floating point")
            if not bool(tensor.isfinite().all().item()):
                raise ValueError("tensor contains NaN/Inf")
            feature_sha = tensor_sha256(tensor)
            if str(row["feature_sha256"]) != feature_sha:
                raise ValueError("per-feature checksum mismatch")
            source = str(row["source"])
            if expected_source is not None and source != expected_source:
                raise ValueError(
                    f"source {source} differs from expected {expected_source}"
                )
            feature = ExtractedFeature(
                identity=identity,
                condition=str(row["condition"]),
                source=source,
                module_path=str(row["module_path"]),
                layer_index=(
                    None
                    if row["layer_index"] is None
                    else int(row["layer_index"])
                ),
                denoise_step_index=(
                    None
                    if row["denoise_step_index"] is None
                    else int(row["denoise_step_index"])
                ),
                pooling=str(row["pooling"]),
                tensor=tensor,
            )
            record = _feature_record(
                feature=feature,
                tensor=tensor,
                shard_path=shard_path,
                shard_sha256=observed_shard_sha,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Thought4ArtifactError(
                f"invalid feature shard row {index}: {exc}"
            ) from exc
        features.append(feature)
        records.append(record)
    return features, records


def _validate_reused_features(
    expected: Sequence[ExtractedFeature],
    observed: Sequence[ExtractedFeature],
) -> None:
    if len(expected) != len(observed):
        raise Thought4ArtifactError(
            "resumed feature shard row count differs from frozen extraction"
        )
    for index, (first, second) in enumerate(zip(expected, observed)):
        first_metadata = (
            first.identity,
            first.condition,
            first.source,
            first.module_path,
            first.layer_index,
            first.denoise_step_index,
            first.pooling,
        )
        second_metadata = (
            second.identity,
            second.condition,
            second.source,
            second.module_path,
            second.layer_index,
            second.denoise_step_index,
            second.pooling,
        )
        if first_metadata != second_metadata:
            raise Thought4ArtifactError(
                f"resumed feature metadata differs at row {index}"
            )
        if tensor_sha256(first.tensor) != tensor_sha256(second.tensor):
            raise Thought4ArtifactError(
                f"resumed feature tensor differs at row {index}"
            )
