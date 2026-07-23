"""Durable, isolated artifacts for shadow future diagnostics.

The diagnostic namespace is intentionally separate from the evaluation worker
files consumed by :mod:`fastwam_ood_eval.analysis.aggregate`.  In particular,
this module never writes ``episode_results.jsonl``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def clone_action_chunk(actions: Any) -> Any:
    """Return storage-independent action data suitable for environment execution."""

    if hasattr(actions, "detach") and callable(actions.detach):
        detached = actions.detach()
        if hasattr(detached, "clone") and callable(detached.clone):
            return detached.clone()
    # ``list.copy`` and similar container methods are shallow: a probe could
    # mutate nested action rows through shared references. Deepcopy is cheap for
    # these small chunks and NumPy arrays implement it as independent storage.
    return copy.deepcopy(actions)


def action_chunk_hash(actions: Any) -> str:
    """Hash action values, shape and dtype without mutating the input."""

    value = actions
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    try:
        import numpy as np

        array = np.asarray(value)
        header = json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes(order="C")
    except (ImportError, TypeError, ValueError):
        header = b"json"
        payload = json.dumps(
            _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def diagnostic_id(job_id: str, config_fingerprint: str = "v1") -> str:
    payload = f"{job_id}\x1f{config_fingerprint}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def ensure_isolated_output(output_dir: Path, source_output_dir: Path | None) -> None:
    """Reject any output location that could mutate the source experiment tree."""

    if source_output_dir is None:
        return
    output = output_dir.resolve()
    source = source_output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError(
            "Diagnostic output and source experiment must be disjoint; "
            f"output={output}, source={source}"
        )


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return {str(key): _jsonable(value) for key, value in record.items()}
    if hasattr(record, "to_dict") and callable(record.to_dict):
        payload = record.to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("record.to_dict() must return a mapping")
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if is_dataclass(record):
        return _jsonable(asdict(record))
    raise TypeError(f"Unsupported diagnostic record type: {type(record).__name__}")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON without a shared fixed temporary filename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_latest_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    """Load the last valid JSON object per key, tolerating a partial crash tail."""

    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get(key) in (None, ""):
                continue
            records[str(row[key])] = row
    return records


def load_all_completed_jobs(
    output_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load durable completions across every rank.

    The protocol fingerprint is part of the identity so changing world size or
    sharding cannot rerun a job completed by a different rank, while stale
    completions from an older protocol remain harmless.
    """

    records: dict[tuple[str, str], dict[str, Any]] = {}
    order_by_key: dict[tuple[str, str], tuple[int, int, int, str]] = {}
    pattern = Path(output_dir) / "workers"
    for path in sorted(pattern.glob("rank_*/completed_jobs.jsonl")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                job_id = row.get("job_id")
                fingerprint = row.get("protocol_fingerprint")
                if job_id in (None, "") or fingerprint in (None, ""):
                    continue
                key = (str(job_id), str(fingerprint))
                order = _record_order(row, path=path, line_index=line_index)
                if key not in records or order > order_by_key[key]:
                    records[key] = row
                    order_by_key[key] = order
    return records


def _record_order(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_index: int,
) -> tuple[int, int, int, str]:
    """Order rerun attempts independently of rank-directory traversal.

    New records carry an attempt start and durable append timestamp.  Legacy
    rows sort before timestamped rows; file mtime and line order are only a
    deterministic fallback for old records.
    """

    def integer(name: str) -> int:
        value = row.get(name)
        if isinstance(value, bool):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    try:
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        mtime = 0
    return (
        integer("attempt_started_ns"),
        integer("recorded_at_ns") or mtime,
        int(line_index),
        str(path),
    )


def _to_rgb_uint8(frame: Any) -> Any:
    """Convert common tensor/PIL/NumPy/dict frame forms to HWC uint8 RGB."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - real Fast-WAM runtime includes NumPy
        raise RuntimeError("Diagnostic media writing requires numpy") from exc

    if isinstance(frame, Mapping):
        arrays = [_to_rgb_uint8(value) for value in frame.values()]
        if not arrays:
            raise ValueError("Cannot encode an empty frame mapping")
        target_h = arrays[0].shape[0]
        if any(array.shape[0] != target_h for array in arrays):
            from PIL import Image

            arrays = [
                np.asarray(Image.fromarray(array).resize((array.shape[1], target_h)))
                if array.shape[0] != target_h
                else array
                for array in arrays
            ]
        return np.concatenate(arrays, axis=1)
    if hasattr(frame, "detach") and callable(frame.detach):
        frame = frame.detach()
    if hasattr(frame, "cpu") and callable(frame.cpu):
        frame = frame.cpu()
    if hasattr(frame, "numpy") and callable(frame.numpy):
        frame = frame.numpy()
    array = np.asarray(frame)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an image-like value, got shape {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected one, three or four channels, got shape {array.shape}")
    if array.dtype.kind in "fc":
        finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        minimum = float(finite.min()) if finite.size else 0.0
        maximum = float(finite.max()) if finite.size else 0.0
        if minimum >= -1.0001 and maximum <= 1.0001:
            finite = (finite + 1.0) * 127.5 if minimum < 0.0 else finite * 255.0
        array = finite
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _resize_like(frame: Any, reference: Any) -> Any:
    import numpy as np

    if frame.shape[:2] == reference.shape[:2]:
        return frame
    from PIL import Image

    return np.asarray(
        Image.fromarray(frame).resize((reference.shape[1], reference.shape[0]), resample=Image.BILINEAR)
    )


class DiagnosticArtifactWriter:
    """Write per-probe media and durable per-probe/per-job JSONL records."""

    def __init__(
        self,
        output_dir: Path,
        worker_rank: int = 0,
        *,
        source_output_dir: Path | None = None,
        fps: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        ensure_isolated_output(self.output_dir, source_output_dir)
        self.worker_rank = int(worker_rank)
        self.worker_dir = self.output_dir / "workers" / f"rank_{self.worker_rank}"
        self.diagnostics_path = self.worker_dir / "diagnostics.jsonl"
        self.completed_jobs_path = self.worker_dir / "completed_jobs.jsonl"
        self.artifact_root = self.worker_dir / "artifacts"
        self.fps = int(fps)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "diagnostic_manifest.json"

    def prepare_manifest(
        self,
        *,
        protocol_fingerprint: str,
        config: Mapping[str, Any],
        experiment_id: str,
        source_experiment_id: str | None,
        source_output_dir: Path | None,
        resume: bool,
        overwrite: bool,
        provenance: Mapping[str, Any] | None = None,
        write: bool = True,
        planned_job_count: int | None = None,
    ) -> dict[str, Any]:
        """Validate the output namespace and atomically establish its protocol.

        A Thought 1 output tree is never adopted as a diagnostic output merely
        because the caller pointed at it.  Protocol changes similarly require an
        explicit overwrite decision so old completion markers cannot hide work.
        """

        thought1_manifest = self.output_dir / "experiment_manifest.json"
        thought1_results = list(
            (self.output_dir / "workers").glob("rank_*/episode_results.jsonl")
        )
        if thought1_manifest.is_file() or thought1_results:
            raise RuntimeError(
                "Refusing to use a Thought 1 evaluation output as a diagnostic output: "
                f"{self.output_dir}"
            )

        source_summary = self._source_manifest_summary(
            source_experiment_id=source_experiment_id,
            source_output_dir=source_output_dir,
        )
        existing: dict[str, Any] | None = None
        if self.manifest_path.is_file():
            try:
                candidate = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid diagnostic manifest: {self.manifest_path}") from exc
            if not isinstance(candidate, dict):
                raise RuntimeError(f"Invalid diagnostic manifest object: {self.manifest_path}")
            existing = candidate
            previous = str(existing.get("protocol_fingerprint", ""))
            if previous != protocol_fingerprint and resume and not overwrite:
                raise RuntimeError(
                    "Diagnostic protocol fingerprint changed while resume is enabled; "
                    "choose a fresh output directory or explicitly set overwrite=true. "
                    f"previous={previous or 'missing'}, current={protocol_fingerprint}"
                )
            if previous == protocol_fingerprint:
                if write:
                    self._write_source_manifest(source_summary)
                return existing

        payload = {
            "schema_version": 1,
            "kind": "future_shadow_diagnostics",
            "experiment_id": experiment_id,
            "source_experiment_id": source_experiment_id,
            "source_output_dir": str(source_output_dir) if source_output_dir is not None else None,
            "protocol_fingerprint": protocol_fingerprint,
            "planned_job_count": planned_job_count,
            "config": _jsonable(config),
            "provenance": _jsonable(provenance or {}),
            "status": "diagnostic_worker_outputs_pending",
        }
        if not write:
            # Non-zero ranks only validate existing state. Rank zero publishes
            # both manifests; no distributed barrier is needed for deterministic
            # job sharding and per-rank JSONL output.
            return payload
        _atomic_json_write(self.manifest_path, payload)
        self._write_source_manifest(source_summary)
        return payload

    def _source_manifest_summary(
        self,
        *,
        source_experiment_id: str | None,
        source_output_dir: Path | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "source_experiment_id": source_experiment_id,
            "source_output_dir": str(source_output_dir) if source_output_dir is not None else None,
            "source_manifest_path": None,
            "source_manifest_sha256": None,
            "source_manifest": None,
        }
        if source_output_dir is None:
            return payload
        source = Path(source_output_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Diagnostic source output does not exist: {source}")
        source_manifest = source / "experiment_manifest.json"
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Diagnostic source manifest does not exist: {source_manifest}")
        raw = source_manifest.read_text(encoding="utf-8")
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid source experiment manifest: {source_manifest}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Invalid source experiment manifest object: {source_manifest}")
        actual_id = manifest.get("experiment_id")
        if source_experiment_id is not None and str(actual_id) != source_experiment_id:
            raise RuntimeError(
                "Configured source_experiment_id does not match the read-only source manifest: "
                f"configured={source_experiment_id}, manifest={actual_id}"
            )
        payload.update(
            {
                "source_experiment_id": actual_id,
                "source_manifest_path": str(source_manifest),
                "source_manifest_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "source_manifest": {
                    "experiment_id": actual_id,
                    "status": manifest.get("status"),
                    "config_source": manifest.get("config_source"),
                    "job_count": manifest.get("job_count"),
                    "job_manifest": manifest.get("job_manifest"),
                    "provenance": manifest.get("provenance", {}),
                },
            }
        )
        return payload

    def _write_source_manifest(self, payload: Mapping[str, Any]) -> None:
        path = self.output_dir / "source_manifest.json"
        _atomic_json_write(path, payload)

    def completed_jobs(self) -> dict[str, dict[str, Any]]:
        return load_latest_jsonl(self.completed_jobs_path, "job_id")

    def is_job_complete(self, job_id: str, *, protocol_fingerprint: str | None = None) -> bool:
        record = self.completed_jobs().get(str(job_id))
        return bool(
            record
            and record.get("status") in {"completed", "skipped"}
            and (
                protocol_fingerprint is None
                or record.get("protocol_fingerprint") == protocol_fingerprint
            )
        )

    def append_diagnostic(self, record: Any) -> dict[str, Any]:
        payload = _record_dict(record)
        if payload.get("recorded_at_ns") is None:
            payload["recorded_at_ns"] = time.time_ns()
        _append_jsonl(self.diagnostics_path, payload)
        return payload

    def mark_job_complete(
        self,
        *,
        job_id: str,
        status: str,
        termination_reason: str,
        success: bool,
        probe_count: int,
        diagnostic_id_value: str | None = None,
        protocol_fingerprint: str | None = None,
        error: str | None = None,
        attempt_id: str | None = None,
        attempt_started_ns: int | None = None,
        probe_error_count: int = 0,
    ) -> dict[str, Any]:
        if status not in {"completed", "skipped", "error", "exception"}:
            raise ValueError(f"Unsupported diagnostic completion status: {status}")
        payload = {
            "job_id": str(job_id),
            "diagnostic_id": diagnostic_id_value,
            "protocol_fingerprint": protocol_fingerprint,
            "status": status,
            "termination_reason": termination_reason,
            "success": bool(success),
            "probe_count": int(probe_count),
            "probe_error_count": int(probe_error_count),
            "error": error,
            "attempt_id": attempt_id,
            "attempt_started_ns": attempt_started_ns,
            "recorded_at_ns": time.time_ns(),
        }
        _append_jsonl(self.completed_jobs_path, payload)
        return payload

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.resolve().relative_to(self.output_dir.resolve()))

    def write_image(self, path: Path, frame: Any) -> Path:
        """Atomically write one RGB frame for direct per-probe inspection."""

        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        Image.fromarray(_to_rgb_uint8(frame)).save(temporary)
        temporary.replace(path)
        return path

    def write_video(self, path: Path, frames: Sequence[Any], *, fps: float | None = None) -> Path | None:
        if not frames:
            return None
        try:
            import imageio.v2 as imageio
        except ImportError as exc:  # pragma: no cover - Fast-WAM environment supplies imageio
            raise RuntimeError("Diagnostic video writing requires imageio") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        with imageio.get_writer(str(temporary), fps=float(fps or self.fps)) as handle:
            for frame in frames:
                handle.append_data(_to_rgb_uint8(frame))
        temporary.replace(path)
        return path

    def write_latents(self, path: Path, **arrays: Any) -> Path | None:
        values = {key: value for key, value in arrays.items() if value is not None}
        if not values:
            return None
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Diagnostic latent writing requires numpy") from exc
        serializable: dict[str, Any] = {}
        for key, value in values.items():
            if hasattr(value, "detach") and callable(value.detach):
                value = value.detach()
            if hasattr(value, "cpu") and callable(value.cpu):
                value = value.cpu()
            serializable[key] = np.asarray(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, **serializable)
        temporary.replace(path)
        return path

    def write_probe_artifacts(
        self,
        *,
        job_id: str,
        replan_index: int,
        current_frame: Any | None = None,
        predicted_frames: Sequence[Any],
        actual_frames: Sequence[Any],
        side_by_side_predicted_frames: Sequence[Any] | None = None,
        fps: float | None = None,
        predicted_latents: Any = None,
        actual_latents: Any = None,
        save_predicted: bool = True,
        save_actual: bool = True,
        save_side_by_side: bool = True,
        save_latents: bool = False,
    ) -> dict[str, str | None]:
        stem = f"{job_id}__probe_{int(replan_index):04d}"
        current_frame_path = (
            self.write_image(
                self.worker_dir / "current_frames" / f"{stem}.png",
                current_frame,
            )
            if current_frame is not None
            and (save_predicted or save_actual or save_side_by_side)
            else None
        )
        predicted_path = (
            self.write_video(
                self.worker_dir / "predicted_futures" / f"{stem}.mp4",
                list(predicted_frames),
                fps=fps,
            )
            if save_predicted
            else None
        )
        actual_path = (
            self.write_video(
                self.worker_dir / "actual_futures" / f"{stem}.mp4",
                list(actual_frames),
                fps=fps,
            )
            if save_actual
            else None
        )
        comparison: list[Any] = []
        if save_side_by_side:
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Diagnostic comparison writing requires numpy") from exc
            comparison_predictions = (
                predicted_frames
                if side_by_side_predicted_frames is None
                else side_by_side_predicted_frames
            )
            for predicted, actual in zip(comparison_predictions, actual_frames):
                predicted_rgb = _to_rgb_uint8(predicted)
                actual_rgb = _resize_like(_to_rgb_uint8(actual), predicted_rgb)
                comparison.append(np.concatenate([predicted_rgb, actual_rgb], axis=1))
        side_by_side_path = (
            self.write_video(
                self.worker_dir / "side_by_side" / f"{stem}.mp4",
                comparison,
                fps=fps,
            )
            if save_side_by_side
            else None
        )
        latent_path = (
            self.write_latents(
                self.worker_dir / "latents" / f"{stem}.npz",
                predicted=predicted_latents,
                actual=actual_latents,
            )
            if save_latents
            else None
        )
        return {
            "current_frame_path": self._relative(current_frame_path),
            "predicted_video_path": self._relative(predicted_path),
            "actual_video_path": self._relative(actual_path),
            "side_by_side_video_path": self._relative(side_by_side_path),
            "latent_path": self._relative(latent_path),
        }


__all__ = [
    "DiagnosticArtifactWriter",
    "action_chunk_hash",
    "clone_action_chunk",
    "diagnostic_id",
    "ensure_isolated_output",
    "load_all_completed_jobs",
    "load_latest_jsonl",
    "_record_order",
]
