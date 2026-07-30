"""Real Fast-WAM K=1 correct/null/shuffle action counterfactual.

This is an engineering action-sensitivity smoke, not a rollout.  It reads
eight already-consumed standard LIBERO current observations, runs the frozen
Video DiT online for correct and shuffle, injects the E6 A1 Adapter into the
20-step action denoiser, and never reads a training future cache or outcome.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.online_counterfactual import (
    ONLINE_CF_CONDITIONS,
    ONLINE_CF_SAMPLE_SCHEMA,
    K1OnlineCounterfactualConfig,
    OnlineCohortSample,
    action_pair_metrics,
    action_sha256,
    aggregate_online_counterfactual,
    build_episode_derangement,
    compute_replay_floor,
    delta_direction_cosine,
    stable_online_seed,
    validate_online_sample_result,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path
from fastwam_ood_eval.thought3.schemas import canonical_json


PREFLIGHT_SCHEMA = "thought3.k1_online_counterfactual.preflight.v1"
COHORT_MANIFEST_SCHEMA = "thought3.k1_online.cohort_manifest.v1"
RUN_STATUS_SCHEMA = "thought3.k1_online_counterfactual.run_status.v1"
ARTIFACT_MANIFEST_SCHEMA = (
    "thought3.k1_online_counterfactual.artifacts.v1"
)


class Phase1OnlineCounterfactualError(RuntimeError):
    """Raised when a real Phase 1 hard invariant fails."""


@dataclass(frozen=True)
class _CurrentEntry:
    identity: Any
    split: str


@dataclass(frozen=True)
class PreparedOnlineSample:
    cohort: OnlineCohortSample
    image: Any
    context: Any
    context_mask: Any
    current_latent: Any
    proprio: Any
    preprocessing_ms: float
    context_construction_ms: float
    current_encoding_ms: float
    preparation_peak_allocated_mib: float
    preparation_peak_reserved_mib: float
    source: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "phase": "Thought3-K1-online-counterfactual",
                "stage": stage,
                "time": _utc_now(),
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def online_counterfactual_dry_run_payload(
    cfg: K1OnlineCounterfactualConfig,
) -> dict[str, Any]:
    mapping = build_episode_derangement(cfg)
    return {
        "command": "thought3-k1-online-counterfactual",
        "config_fingerprint": cfg.fingerprint,
        "cohort_fingerprint": cfg.cohort_fingerprint,
        "conditions": list(ONLINE_CF_CONDITIONS),
        "dry_run": True,
        "main_checkpoint": str(cfg.e6_checkpoint_dir),
        "output_dir": str(cfg.output_dir),
        "sample_count": len(cfg.cohort),
        "shuffle_mapping_sha256": mapping["fingerprint"],
        "would_load_checkpoint": False,
        "would_load_fastwam": False,
        "would_run_backward": False,
        "would_run_optimizer": False,
        "would_start_rollout": False,
        "would_write": False,
        "scope": {
            "development": False,
            "future_rgb": False,
            "ood": False,
            "rollout": False,
            "success": False,
            "training_future_cache": False,
        },
    }


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_status(path: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        text=True,
    )


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _verify_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise Phase1OnlineCounterfactualError(
            f"{label} SHA mismatch: expected={expected}, observed={observed}"
        )
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "sha256": observed,
    }


def _preflight(cfg: K1OnlineCounterfactualConfig) -> tuple[Any, dict[str, Any]]:
    """Verify all frozen inputs without loading torch/model/checkpoint tensors."""

    from fastwam_ood_eval.thought3.config import load_thought3_config

    project_root = Path(".").resolve()
    fastwam_root = Path("third_party/FastWAM").resolve()
    if _git_status(project_root):
        raise Phase1OnlineCounterfactualError(
            "formal online counterfactual requires a clean project worktree"
        )
    if _git_status(fastwam_root):
        raise Phase1OnlineCounterfactualError(
            "formal online counterfactual requires a clean Fast-WAM worktree"
        )
    project_commit = _git_head(project_root)
    fastwam_commit = _git_head(fastwam_root)
    if fastwam_commit != cfg.fastwam_commit:
        raise Phase1OnlineCounterfactualError(
            "Fast-WAM commit differs from the frozen source"
        )
    files = {
        "thought3_config": _verify_file(
            cfg.thought3_config_path,
            cfg.thought3_config_sha256,
            "base Thought3 config",
        ),
        "e6_gate": _verify_file(
            cfg.e6_gate_path, cfg.e6_gate_sha256, "E6 gate"
        ),
        "e6_adapter": _verify_file(
            cfg.e6_checkpoint_dir / "adapter.safetensors",
            cfg.e6_adapter_sha256,
            "E6 Adapter",
        ),
        "e6_checkpoint_manifest": _verify_file(
            cfg.e6_checkpoint_dir / "manifest.json",
            cfg.e6_checkpoint_manifest_sha256,
            "E6 checkpoint manifest",
        ),
    }
    base_cfg = load_thought3_config(cfg.thought3_config_path)
    if (
        base_cfg.fingerprint != cfg.thought3_config_fingerprint
        or base_cfg.runtime.backend != "fastwam"
        or base_cfg.runtime.device != "cuda:0"
        or base_cfg.variant != "A1"
        or base_cfg.sampler.active_k != 1
        or base_cfg.runtime.action_denoise_steps != 20
        or base_cfg.backbone.checkpoint_sha256
        != cfg.backbone_checkpoint_sha256
        or base_cfg.backbone.dataset_stats_sha256
        != cfg.dataset_stats_sha256
        or base_cfg.backbone.fastwam_commit != cfg.fastwam_commit
    ):
        raise Phase1OnlineCounterfactualError(
            "base Thought3 config no longer matches the frozen online design"
        )
    checkpoint = load_json(cfg.e6_checkpoint_dir / "manifest.json")
    expected_checkpoint = {
        "adapter_fingerprint": cfg.e6_adapter_fingerprint,
        "backbone_checkpoint_sha256": cfg.backbone_checkpoint_sha256,
        "config_fingerprint": cfg.e6_checkpoint_config_fingerprint,
        "dataset_stats_sha256": cfg.dataset_stats_sha256,
        "fastwam_commit": cfg.fastwam_commit,
        "frozen_parameter_sha256": cfg.frozen_parameter_sha256,
        "global_step": 200,
        "k": 1,
        "split_fingerprint": cfg.split_fingerprint,
        "train_seed": 3407,
        "variant": "A1",
    }
    mismatches = {
        key: (checkpoint.get(key), expected)
        for key, expected in expected_checkpoint.items()
        if checkpoint.get(key) != expected
    }
    if (
        mismatches
        or checkpoint.get("extra", {}).get("checkpoint_kind")
        != "adapter_only"
        or checkpoint.get("extra", {}).get("contains_backbone") is not False
        or checkpoint.get("extra", {}).get("adapter_state_sha256")
        != cfg.e6_adapter_state_sha256
        or checkpoint.get("extra", {}).get("files_sha256", {}).get(
            "adapter.safetensors"
        )
        != cfg.e6_adapter_sha256
    ):
        raise Phase1OnlineCounterfactualError(
            f"E6 checkpoint provenance mismatch: {mismatches}"
        )
    e6 = load_json(cfg.e6_gate_path)
    e6_sample_ids = list(e6["fresh_cohort"]["sample_ids"])
    if (
        e6_sample_ids
        != [sample.base_sample_id for sample in cfg.cohort]
        or e6["fresh_cohort"]["selection"]
        != "train order positions 9-16 (1-based)"
        or e6["tracks"]["A1"]["result"]["checkpoint_roundtrip"][
            "adapter_state_sha256"
        ]
        != cfg.e6_adapter_state_sha256
        or int(e6["scope"]["sample_count"]) != 8
        or e6["scope"]["ood_outcomes_read"] is not False
        or e6["scope"]["success_outcomes_read"] is not False
    ):
        raise Phase1OnlineCounterfactualError(
            "E6 gate/cohort evidence differs from the frozen online source"
        )
    mapping = build_episode_derangement(cfg)
    return base_cfg, {
        "schema_version": PREFLIGHT_SCHEMA,
        "checked_at": _utc_now(),
        "checkpoint_selection_disclosure": (
            cfg.checkpoint_selection_disclosure
        ),
        "cohort_fingerprint": cfg.cohort_fingerprint,
        "config_fingerprint": cfg.fingerprint,
        "files": files,
        "fastwam_commit": fastwam_commit,
        "fastwam_worktree_clean": True,
        "main_checkpoint": {
            "adapter_fingerprint": cfg.e6_adapter_fingerprint,
            "adapter_state_sha256": cfg.e6_adapter_state_sha256,
            "adapter_tensor_sha256": cfg.e6_adapter_sha256,
            "global_step": 200,
            "path": str(cfg.e6_checkpoint_dir),
            "variant": "A1",
        },
        "project_commit": project_commit,
        "project_worktree_clean": True,
        "shuffle_mapping_sha256": mapping["fingerprint"],
        "scope": {
            "action_target_read": False,
            "development_read": False,
            "future_rgb_read": False,
            "ood_read": False,
            "rollout_started": False,
            "success_read": False,
            "training_future_cache_read": False,
        },
    }


def _cohort_manifest(cfg: K1OnlineCounterfactualConfig) -> dict[str, Any]:
    payload = {
        "schema_version": COHORT_MANIFEST_SCHEMA,
        "selection": cfg.cohort_selection,
        "samples": [
            {
                "base_sample_id": sample.base_sample_id,
                "episode_id": sample.episode_id,
                "identity": sample.identity.to_dict(),
                "split": sample.split,
            }
            for sample in cfg.cohort
        ],
    }
    fingerprint = _canonical_sha(payload)
    if fingerprint != cfg.cohort_fingerprint:
        raise Phase1OnlineCounterfactualError(
            "cohort fingerprint is not self-consistent"
        )
    return {**payload, "fingerprint": fingerprint}


def _cuda_measure(
    function: Callable[[], Any],
    *,
    device: str,
    hard_limit_gib: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if peak_allocated >= int(hard_limit_gib * 2**30):
        raise Phase1OnlineCounterfactualError(
            f"CUDA peak {peak_allocated / 2**30:.3f} GiB exceeds "
            f"{hard_limit_gib:.3f} GiB"
        )
    return value, {
        "latency_ms": elapsed_ms,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
    }


def _strict_current_observation(
    source: Any,
    rows: Any,
    entry: _CurrentEntry,
) -> Any:
    """Load only identity, current state, and current camera timestamps."""

    from fastwam_ood_eval.thought3.real_cache_builder import (
        CurrentObservation,
        _scalar,
        preprocess_current_camera_frames,
        preprocess_current_proprio,
    )

    identity = entry.identity
    index = source._row_index(
        identity.episode_index,
        identity.frame_index,
    )
    row = rows[index]
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
        raise Phase1OnlineCounterfactualError(
            "strict current-only dataset row identity mismatch"
        )
    timestamp = float(_scalar(row["timestamp"]))
    query = {
        str(key): [timestamp] for key in source.inner.meta.video_keys
    }
    started = time.perf_counter()
    frames = source.inner._query_videos(
        query,
        identity.episode_index,
    )
    decode_ms = (time.perf_counter() - started) * 1000.0
    if set(frames) != set(query):
        raise Phase1OnlineCounterfactualError(
            "strict current-only camera decode key mismatch"
        )
    source.telemetry["current_camera_frames_decoded"] += len(frames)
    image = preprocess_current_camera_frames(
        frames,
        processor=source.processor,
        robot_video_dataset=source.dataset,
    )
    proprio = preprocess_current_proprio(
        row["observation.state"],
        processor=source.processor,
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
            "selected_hf_columns": list(rows.column_names),
            "timestamp_ns": identity.timestamp_ns,
        },
    )


def _load_and_prepare(
    cfg: K1OnlineCounterfactualConfig,
    base_cfg: Any,
) -> tuple[Any, Any, tuple[PreparedOnlineSample, ...], dict[str, Any]]:
    import torch

    from fastwam_ood_eval.thought3.checkpointing import (
        load_adapter_checkpoint,
    )
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _load_upstream_model,
    )
    from fastwam_ood_eval.thought3.real_cache_builder import (
        CurrentOnlyLiberoSource,
        _load_prompt_context,
    )
    from fastwam_ood_eval.thought3.real_training import build_real_adapter
    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    device = cfg.device
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            False
        )
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.experiment_seed)
    torch.cuda.manual_seed_all(cfg.experiment_seed)

    torch.cuda.reset_peak_memory_stats(device)
    _progress("model_load_started", device=device)
    model, upstream_cfg, model_report = _load_upstream_model(base_cfg)
    torch.cuda.synchronize(device)
    model_report = {
        **model_report,
        "load_peak_allocated_mib": (
            int(torch.cuda.max_memory_allocated(device)) / 2**20
        ),
        "load_peak_reserved_mib": (
            int(torch.cuda.max_memory_reserved(device)) / 2**20
        ),
    }
    if model_report["load_peak_allocated_mib"] >= (
        cfg.max_gpu_memory_gb * 1024.0
    ):
        raise Phase1OnlineCounterfactualError(
            "model load exceeded the frozen single-RTX-4090 memory bound"
        )
    _progress("model_loaded", **model_report)
    if (
        type(model).__name__ != "FastWAM"
        or getattr(model.video_expert, "action_conditioned", None) is not False
        or str(model.video_expert.video_attention_mask_mode)
        != "first_frame_causal"
        or getattr(
            model.video_expert,
            "fuse_vae_embedding_in_latents",
            None,
        )
        is not True
    ):
        raise Phase1OnlineCounterfactualError(
            "loaded Fast-WAM architecture differs from the frozen "
            "current-only action/K=1 future contract"
        )
    frozen_before = parameter_state_sha256(
        iter(model.named_parameters())
    )
    if (
        frozen_before != cfg.frozen_parameter_sha256
        or any(parameter.requires_grad for parameter in model.parameters())
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise Phase1OnlineCounterfactualError(
            "Fast-WAM frozen parameter invariant failed before inference"
        )
    adapter = build_real_adapter(base_cfg, device=device)
    manifest = load_adapter_checkpoint(
        cfg.e6_checkpoint_dir,
        adapter=adapter,
        expected={
            "adapter_fingerprint": cfg.e6_adapter_fingerprint,
            "backbone_checkpoint_sha256": (
                cfg.backbone_checkpoint_sha256
            ),
            "config_fingerprint": (
                cfg.e6_checkpoint_config_fingerprint
            ),
            "dataset_stats_sha256": cfg.dataset_stats_sha256,
            "fastwam_commit": cfg.fastwam_commit,
            "frozen_parameter_sha256": cfg.frozen_parameter_sha256,
            "k": 1,
            "split_fingerprint": cfg.split_fingerprint,
            "variant": "A1",
        },
    )
    adapter.requires_grad_(False)
    adapter.eval()
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise Phase1OnlineCounterfactualError(
            "online Adapter must be inference-only"
        )
    entries = tuple(
        _CurrentEntry(identity=sample.identity, split=sample.split)
        for sample in cfg.cohort
    )
    source = CurrentOnlyLiberoSource(base_cfg, upstream_cfg)
    selected_columns = (
        "episode_index",
        "frame_index",
        "task_index",
        "timestamp",
        "observation.state",
    )
    available_columns = set(source.inner.hf_dataset.column_names)
    if not set(selected_columns).issubset(available_columns):
        raise Phase1OnlineCounterfactualError(
            "strict current-only source columns are unavailable"
        )
    current_rows = source.inner.hf_dataset.select_columns(
        list(selected_columns)
    )
    source.telemetry["selected_hf_columns"] = list(selected_columns)
    source.telemetry["selected_action_columns"] = []
    context_base, context_mask_base, prompt = _load_prompt_context(
        model,
        entries,
        device=device,
    )
    if model.text_encoder is not None:
        model.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    prepared: list[PreparedOnlineSample] = []
    for index, (sample, entry) in enumerate(
        zip(cfg.cohort, entries, strict=True)
    ):
        preprocessing_started = time.perf_counter()
        observation = _strict_current_observation(
            source,
            current_rows,
            entry,
        )
        preprocessing_ms = (
            time.perf_counter() - preprocessing_started
        ) * 1000.0
        image = observation.image.detach().cpu().contiguous()
        proprio_cpu = observation.proprio.detach().cpu().contiguous()
        proprio = proprio_cpu.to(
            device=device, dtype=model.torch_dtype
        )
        torch.cuda.synchronize(device)
        context_started = time.perf_counter()
        with torch.inference_mode():
            context, context_mask = model._append_proprio_to_context(
                context_base,
                context_mask_base,
                proprio,
            )
        torch.cuda.synchronize(device)
        context_ms = (time.perf_counter() - context_started) * 1000.0
        image_device = image.to(device=device, dtype=model.torch_dtype)
        if index == 0:
            with torch.inference_mode():
                warmup_latent = model._encode_input_image_latents_tensor(
                    image_device
                )
            torch.cuda.synchronize(device)
            del warmup_latent
            torch.cuda.empty_cache()

        def encode_current() -> Any:
            with torch.inference_mode():
                return model._encode_input_image_latents_tensor(
                    image_device
                )

        current_latent, encode = _cuda_measure(
            encode_current,
            device=device,
            hard_limit_gib=cfg.max_gpu_memory_gb,
        )
        if (
            tuple(current_latent.shape) != (1, 48, 1, 14, 28)
            or tuple(context.shape) != (1, 129, 4096)
            or tuple(context_mask.shape) != (1, 129)
        ):
            raise Phase1OnlineCounterfactualError(
                "prepared current/context tensor shape mismatch"
            )
        prepared.append(
            PreparedOnlineSample(
                cohort=sample,
                image=image,
                context=context.detach()
                .cpu()
                .to(dtype=model.torch_dtype)
                .contiguous(),
                context_mask=context_mask.detach()
                .cpu()
                .bool()
                .contiguous(),
                current_latent=current_latent.detach()
                .cpu()
                .to(dtype=model.torch_dtype)
                .contiguous(),
                proprio=proprio_cpu,
                preprocessing_ms=preprocessing_ms,
                context_construction_ms=context_ms,
                current_encoding_ms=float(encode["latency_ms"]),
                preparation_peak_allocated_mib=float(
                    encode["peak_allocated_mib"]
                ),
                preparation_peak_reserved_mib=float(
                    encode["peak_reserved_mib"]
                ),
                source=dict(observation.source),
            )
        )
        del (
            observation,
            image_device,
            current_latent,
            context,
            context_mask,
            proprio,
        )
        torch.cuda.empty_cache()
        _progress(
            "current_sample_prepared",
            prepared=index + 1,
            total=len(entries),
            base_sample_id=sample.base_sample_id,
        )
    telemetry = dict(source.telemetry)
    if (
        telemetry.get("action_target_read") is not False
        or telemetry.get("actual_future_read") is not False
        or int(telemetry.get("future_rgb_frames_decoded", -1)) != 0
        or int(telemetry.get("current_camera_frames_decoded", -1)) != 16
        or telemetry.get("selected_action_columns") != []
        or telemetry.get("selected_hf_columns")
        != list(selected_columns)
    ):
        raise Phase1OnlineCounterfactualError(
            f"current-only source boundary failed: {telemetry}"
        )
    return (
        model,
        adapter,
        tuple(prepared),
        {
            "adapter_checkpoint_manifest": manifest.to_dict(),
            "frozen_parameter_sha256_before": frozen_before,
            "model_load": model_report,
            "prompt": prompt,
            "preparation_warmup": {
                "current_encoding_calls": 1,
                "excluded_from_formal_latency": True,
            },
            "source_telemetry": telemetry,
        },
    )


class _CountingVideoVelocity:
    """Count the real frozen Video DiT calls made by the K-step sampler."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(
        self,
        state: Any,
        timestep: Any,
        conditions: Mapping[str, object],
    ) -> Any:
        self.calls += 1
        return self.model.video_expert(
            x=state,
            timestep=timestep.to(
                device=state.device,
                dtype=state.dtype,
            ),
            context=conditions["context"],
            context_mask=conditions["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=True,
        )


class _AdapterCudaTimer:
    """Measure only Adapter forwards, without changing their inputs."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._active = False
        self._starts: list[Any] = []
        self._ends: list[Any] = []
        self._pre_handle = adapter.register_forward_pre_hook(self._pre)
        self._post_handle = adapter.register_forward_hook(self._post)

    def _pre(self, module: Any, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        if self._active:
            import torch

            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._starts.append(event)

    def _post(
        self,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs, output
        if self._active:
            import torch

            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._ends.append(event)

    def reset(self) -> None:
        self._starts.clear()
        self._ends.clear()
        self._active = True

    def finish(self) -> tuple[float, int]:
        self._active = False
        if len(self._starts) != len(self._ends):
            raise Phase1OnlineCounterfactualError(
                "Adapter CUDA timer hook count mismatch"
            )
        return (
            sum(
                float(start.elapsed_time(end))
                for start, end in zip(
                    self._starts,
                    self._ends,
                    strict=True,
                )
            ),
            len(self._starts),
        )

    def close(self) -> None:
        self._pre_handle.remove()
        self._post_handle.remove()


def _tensor_artifact(
    output: Path,
    *,
    relative_path: Path,
    key: str,
    tensor: Any,
) -> dict[str, Any]:
    """Atomically save a CPU tensor in safetensors format."""

    from safetensors.torch import save_file

    target = output / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".safetensors",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    value = tensor.detach().cpu().contiguous()
    try:
        save_file({key: value}, str(temporary))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "bytes": target.stat().st_size,
        "key": key,
        "path": str(target),
        "sha256": sha256_file(target),
    }


def _action_payload(
    action: Any,
    *,
    output: Path,
    base_sample_id: str,
    condition: str,
) -> dict[str, Any]:
    import torch

    value = action.detach().cpu().float().contiguous()
    if tuple(value.shape) != (32, 7):
        raise Phase1OnlineCounterfactualError(
            f"{condition} action must be [32,7]"
        )
    finite = bool(torch.isfinite(value).all())
    if not finite:
        raise Phase1OnlineCounterfactualError(
            f"{condition} action contains NaN/Inf"
        )
    semantic_sha = action_sha256(value)
    artifact = _tensor_artifact(
        output,
        relative_path=(
            Path("tensors/actions")
            / f"{base_sample_id}.{condition}.safetensors"
        ),
        key="action",
        tensor=value,
    )
    return {
        "artifact": artifact,
        "dtype": str(value.dtype),
        "finite": finite,
        "sha256": semantic_sha,
        "shape": list(value.shape),
        "tensor": value.tolist(),
    }


def _action_from_payload(payload: Mapping[str, Any]) -> Any:
    import torch

    value = torch.tensor(payload["tensor"], dtype=torch.float32)
    if action_sha256(value) != payload["sha256"]:
        raise Phase1OnlineCounterfactualError(
            "stored action tensor/hash mismatch"
        )
    return value


def _verify_artifact_descriptor(
    output: Path,
    descriptor: Mapping[str, Any],
) -> None:
    path = Path(str(descriptor["path"]))
    try:
        path.resolve().relative_to(output.resolve())
    except ValueError as exc:
        raise Phase1OnlineCounterfactualError(
            "resume tensor artifact escaped the isolated output directory"
        ) from exc
    if (
        not path.is_file()
        or path.stat().st_size != int(descriptor["bytes"])
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise Phase1OnlineCounterfactualError(
            f"resume tensor artifact checksum failed: {path}"
        )


def _verify_sample_artifacts(
    output: Path,
    row: Mapping[str, Any],
) -> None:
    for condition in ONLINE_CF_CONDITIONS:
        _verify_artifact_descriptor(
            output,
            row["actions"][condition]["artifact"],
        )
    for condition in ("correct", "shuffle"):
        _verify_artifact_descriptor(
            output,
            row["future"][condition]["artifact"],
        )


def _condition_memory_started(device: str) -> float:
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _condition_memory_finished(
    *,
    device: str,
    started: float,
    hard_limit_gib: float,
) -> dict[str, float]:
    import torch

    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if peak_allocated >= int(hard_limit_gib * 2**30):
        raise Phase1OnlineCounterfactualError(
            f"CUDA peak {peak_allocated / 2**30:.3f} GiB exceeds "
            f"{hard_limit_gib:.3f} GiB"
        )
    return {
        "condition_total_ms": elapsed_ms,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
    }


def _official_b0_action(
    model: Any,
    sample: PreparedOnlineSample,
    *,
    cfg: K1OnlineCounterfactualConfig,
    action_seed: int,
) -> tuple[Any, dict[str, float]]:
    """Run the unmodified upstream FastWAM.infer_action entry point."""

    import torch

    started = _condition_memory_started(cfg.device)
    with torch.inference_mode():
        result = model.infer_action(
            prompt=None,
            input_image=sample.image,
            action_horizon=cfg.action_horizon,
            proprio=None,
            context=sample.context,
            context_mask=sample.context_mask,
            num_inference_steps=cfg.action_denoise_steps,
            seed=action_seed,
            rand_device=cfg.rand_device,
            tiled=False,
        )
    memory = _condition_memory_finished(
        device=cfg.device,
        started=started,
        hard_limit_gib=cfg.max_gpu_memory_gb,
    )
    action = result["action"].detach().cpu().float().contiguous()
    if tuple(action.shape) != (cfg.action_horizon, 7):
        raise Phase1OnlineCounterfactualError(
            "upstream B0 infer_action returned an unexpected action shape"
        )
    return action, {
        **memory,
        "adapter_call_count": 0,
        "adapter_ms": 0.0,
        "action_context_cache_ms": 0.0,
        # The public upstream API does not expose an internal timing split.
        # Its VAE encode, current-frame cache, and Action DiT are included here.
        "action_dit_ms": float(memory["condition_total_ms"]),
        "future_video_dit_ms": 0.0,
    }


def _sample_online_future(
    sampler: Any,
    velocity: _CountingVideoVelocity,
    sample: PreparedOnlineSample,
    *,
    cfg: K1OnlineCounterfactualConfig,
    future_seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Run exactly one online Video DiT update and retain native latent only."""

    import torch

    from fastwam_ood_eval.thought3.future_sampler import tensor_sha256

    model_dtype = sample.current_latent.dtype
    current = sample.current_latent.to(
        device=cfg.device,
        dtype=model_dtype,
    )
    context = sample.context.to(
        device=cfg.device,
        dtype=model_dtype,
    )
    mask = sample.context_mask.to(device=cfg.device, dtype=torch.bool)
    velocity.reset()
    started = _condition_memory_started(cfg.device)
    with torch.inference_mode():
        sampled = sampler.sample(
            current,
            initial_noise_seeds=(future_seed,),
            k=cfg.future_k,
            conditions={"context": context, "context_mask": mask},
        )
    memory = _condition_memory_finished(
        device=cfg.device,
        started=started,
        hard_limit_gib=cfg.max_gpu_memory_gb,
    )
    future = sampled.future_latent.detach().cpu().contiguous()
    if (
        velocity.calls != cfg.future_k
        or tuple(future.shape) != (1, 48, 2, 14, 28)
        or not bool(torch.isfinite(future).all())
    ):
        raise Phase1OnlineCounterfactualError(
            "online K=1 future sampler contract failed"
        )
    metadata = {
        "future_sha256": tensor_sha256(future),
        "initial_state_sha256": list(sampled.initial_state_sha256),
        "k": cfg.future_k,
        "peak_allocated_mib": memory["peak_allocated_mib"],
        "peak_reserved_mib": memory["peak_reserved_mib"],
        "schedule": sampled.schedule.to_dict(),
        "video_dit_call_count": velocity.calls,
        "video_dit_ms": memory["condition_total_ms"],
    }
    del current, context, mask, sampled
    torch.cuda.empty_cache()
    return future, metadata


def _custom_action(
    model: Any,
    injector: Any,
    adapter_timer: _AdapterCudaTimer,
    sample: PreparedOnlineSample,
    *,
    cfg: K1OnlineCounterfactualConfig,
    action_seed: int,
    future: Any | None,
    formal_null: bool,
) -> tuple[Any, dict[str, Any]]:
    """Run the frozen 20-step Action DiT with one scoped intervention."""

    import torch

    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
    )

    if formal_null == (future is not None):
        raise Phase1OnlineCounterfactualError(
            "custom action requires exactly one of formal null or future"
        )
    current = sample.current_latent.to(
        device=cfg.device,
        dtype=model.torch_dtype,
    )
    context = sample.context.to(
        device=cfg.device,
        dtype=model.torch_dtype,
    )
    context_mask = sample.context_mask.to(
        device=cfg.device,
        dtype=torch.bool,
    )
    future_device = (
        None
        if future is None
        else future.to(device=cfg.device, dtype=model.torch_dtype)
    )
    future_mask = (
        None
        if future_device is None
        else torch.ones(
            (1, 2, 14, 28),
            device=cfg.device,
            dtype=torch.bool,
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(action_seed)
    action = torch.randn(
        (1, cfg.action_horizon, 7),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=cfg.device, dtype=model.torch_dtype)

    condition_started = _condition_memory_started(cfg.device)
    torch.cuda.synchronize(cfg.device)
    cache_started = time.perf_counter()
    with torch.inference_mode():
        video_cache, attention_mask, video_seq_len = _prepare_video_cache(
            model,
            current,
            context,
            context_mask,
            action_seq_len=cfg.action_horizon,
        )
    torch.cuda.synchronize(cfg.device)
    cache_ms = (time.perf_counter() - cache_started) * 1000.0
    timesteps, deltas = (
        model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=cfg.action_denoise_steps,
            device=cfg.device,
            dtype=action.dtype,
            shift_override=None,
        )
    )
    scope = (
        injector.activate_null(expected_calls=cfg.action_denoise_steps)
        if formal_null
        else injector.activate(
            future_device,
            future_mask,
            expected_calls=cfg.action_denoise_steps,
        )
    )
    adapter_timer.reset()
    torch.cuda.synchronize(cfg.device)
    denoise_started = time.perf_counter()
    with torch.inference_mode(), scope:
        for step_t, step_delta in zip(timesteps, deltas, strict=True):
            timestep = step_t.unsqueeze(0).to(
                device=cfg.device,
                dtype=action.dtype,
            )
            prediction = _action_from_video_cache(
                model,
                action,
                timestep,
                context,
                context_mask,
                video_cache,
                attention_mask,
                video_seq_len,
            )
            action = model.infer_action_scheduler.step(
                prediction,
                step_delta,
                action,
            )
    torch.cuda.synchronize(cfg.device)
    denoise_ms = (time.perf_counter() - denoise_started) * 1000.0
    adapter_ms, adapter_calls = adapter_timer.finish()
    memory = _condition_memory_finished(
        device=cfg.device,
        started=condition_started,
        hard_limit_gib=cfg.max_gpu_memory_gb,
    )
    expected_adapter_calls = 0 if formal_null else cfg.action_denoise_steps
    if adapter_calls != expected_adapter_calls:
        raise Phase1OnlineCounterfactualError(
            "formal null/future Adapter call-count invariant failed"
        )
    value = action[0].detach().cpu().float().contiguous()
    if (
        tuple(value.shape) != (cfg.action_horizon, 7)
        or not bool(torch.isfinite(value).all())
    ):
        raise Phase1OnlineCounterfactualError(
            "custom Action DiT produced an invalid action chunk"
        )
    report = {
        **memory,
        "action_context_cache_ms": cache_ms,
        "action_dit_ms": max(0.0, denoise_ms - adapter_ms),
        "adapter_call_count": adapter_calls,
        "adapter_ms": adapter_ms,
        "denoiser_inclusive_ms": denoise_ms,
        "future_video_dit_ms": 0.0,
    }
    del (
        current,
        context,
        context_mask,
        future_device,
        future_mask,
        action,
        video_cache,
        attention_mask,
        timesteps,
        deltas,
    )
    torch.cuda.empty_cache()
    return value, report


def _latency_row(
    sample: PreparedOnlineSample,
    condition: Mapping[str, Any],
    *,
    future: Mapping[str, Any] | None,
    b0_public_api: bool = False,
    donor: PreparedOnlineSample | None = None,
) -> dict[str, Any]:
    future_ms = 0.0 if future is None else float(future["video_dit_ms"])
    condition_ms = float(condition["condition_total_ms"]) + future_ms
    preprocessing_ms = sample.preprocessing_ms
    context_construction_ms = sample.context_construction_ms
    if b0_public_api:
        if donor is not None:
            raise Phase1OnlineCounterfactualError(
                "B0 latency cannot include a shuffle donor"
            )
        current_encoding_ms = 0.0
        measurement_semantics = (
            "unmodified FastWAM.infer_action total; its VAE encode, "
            "current-cache construction, and Action DiT are inclusive in "
            "action_dit_ms"
        )
    else:
        current_encoding_ms = sample.current_encoding_ms
        measurement_semantics = (
            "shared current encoding plus separately synchronized online "
            "future/cache/Adapter/Action-DiT stages"
        )
        if donor is not None:
            preprocessing_ms += donor.preprocessing_ms
            context_construction_ms += donor.context_construction_ms
            current_encoding_ms += donor.current_encoding_ms
            measurement_semantics += (
                "; shuffle total includes donor preprocessing/context/"
                "current-encoding needed to construct the control latent"
            )
    future_peak_allocated = (
        0.0 if future is None else float(future["peak_allocated_mib"])
    )
    future_peak_reserved = (
        0.0 if future is None else float(future["peak_reserved_mib"])
    )
    return {
        "action_context_cache_ms": float(
            condition["action_context_cache_ms"]
        ),
        "action_dit_ms": float(condition["action_dit_ms"]),
        "adapter_ms": float(condition["adapter_ms"]),
        "condition_total_ms": condition_ms,
        "context_construction_ms": context_construction_ms,
        "current_encoding_ms": current_encoding_ms,
        "future_video_dit_ms": future_ms,
        "measurement_semantics": measurement_semantics,
        "peak_allocated_mib": max(
            sample.preparation_peak_allocated_mib,
            (
                0.0
                if donor is None
                else donor.preparation_peak_allocated_mib
            ),
            float(condition["peak_allocated_mib"]),
            future_peak_allocated,
        ),
        "peak_reserved_mib": max(
            sample.preparation_peak_reserved_mib,
            (
                0.0
                if donor is None
                else donor.preparation_peak_reserved_mib
            ),
            float(condition["peak_reserved_mib"]),
            future_peak_reserved,
        ),
        "policy_total_ms": (
            preprocessing_ms
            + context_construction_ms
            + current_encoding_ms
            + condition_ms
        ),
        "preprocessing_ms": preprocessing_ms,
    }


def _replay_row(
    sample: PreparedOnlineSample,
    *,
    action_seed: int,
    actions: Sequence[Any],
    latency: Sequence[Mapping[str, Any]],
    cfg: K1OnlineCounterfactualConfig,
) -> dict[str, Any]:
    if len(actions) != cfg.replay_repeats or len(latency) != len(actions):
        raise Phase1OnlineCounterfactualError(
            "B0 replay count differs from the frozen protocol"
        )
    action_payloads = [
        {
            "dtype": "torch.float32",
            "finite": True,
            "sha256": action_sha256(action),
            "shape": list(action.shape),
            "tensor": action.tolist(),
        }
        for action in actions
    ]
    return {
        "schema_version": "thought3.k1_online_counterfactual.replay.v1",
        "action_seed_identity": {
            "experiment_seed": cfg.experiment_seed,
            "namespace": cfg.action_seed_namespace,
            "sample_id": sample.cohort.base_sample_id,
            "seed": action_seed,
        },
        "base_sample_id": sample.cohort.base_sample_id,
        "episode_id": sample.cohort.episode_id,
        "latency": [dict(value) for value in latency],
        "metrics": action_pair_metrics(actions[0], actions[1]),
        "repeats": action_payloads,
    }


def _validate_replay_rows(
    rows: Sequence[Mapping[str, Any]],
    cfg: K1OnlineCounterfactualConfig,
) -> None:
    expected = [sample.base_sample_id for sample in cfg.cohort]
    observed = [str(row.get("base_sample_id")) for row in rows]
    if observed != expected[: len(observed)] or len(set(observed)) != len(
        observed
    ):
        raise Phase1OnlineCounterfactualError(
            "replay rows are not an ordered prefix of the frozen cohort"
        )
    for row in rows:
        repeats = row.get("repeats", ())
        if (
            row.get("schema_version")
            != "thought3.k1_online_counterfactual.replay.v1"
            or len(repeats) != cfg.replay_repeats
        ):
            raise Phase1OnlineCounterfactualError(
                "stored B0 replay row schema/count mismatch"
            )
        actions = [_action_from_payload(value) for value in repeats]
        recomputed = action_pair_metrics(actions[0], actions[1])
        for key in ("l1", "l2", "linf", "action_cosine"):
            if float(recomputed[key]) != float(row["metrics"][key]):
                raise Phase1OnlineCounterfactualError(
                    "stored B0 replay metric mismatch"
                )


def _write_decision_report(
    output: Path,
    *,
    aggregate: Mapping[str, Any],
    cfg: K1OnlineCounterfactualConfig,
) -> Path:
    decision = aggregate["decision"]
    branch = str(decision["next_branch"])
    next_steps = {
        "A": (
            "进入 Phase 2：冻结唯一 A0/A1 完整 28/4 配方；本次运行不自动"
            "启动训练。"
        ),
        "B": (
            "最多允许一次单变量 Adapter 注入结构修复，然后用相同 cohort "
            "重复一次反事实；不得先进入完整训练。"
        ),
        "C": (
            "停止 Adapter-only 完整训练、A2/A4 与 OOD rollout；先登记机制"
            "负结果。"
        ),
    }
    values = decision["values"]
    text = (
        "# Thought3 Phase 1：K=1 在线动作反事实决策\n\n"
        "> 这是 engineering action-sensitivity smoke，不是 rollout，也不支持 "
        "success、ID/OOD 或 K=1 优于 K=2/4 的结论。\n\n"
        f"- 分类：`{decision['classification']}`\n"
        f"- 分支：`{branch}`\n"
        f"- 主 checkpoint：`{cfg.e6_checkpoint_dir}`\n"
        f"- cohort：`{cfg.cohort_fingerprint}`（8 条已消耗 train 样本）\n"
        f"- replay L2 floor："
        f"`{decision['thresholds']['material_l2_threshold']:.12g}`\n"
        f"- correct-null 超过 floor："
        f"`{values['correct_null_exceeds_replay_floor']}/8`\n"
        f"- correct-shuffle 超过 floor："
        f"`{values['correct_shuffle_exceeds_replay_floor']}/8`\n"
        f"- correct/shuffle action hash 改变："
        f"`{values['correct_vs_shuffle_action_hash_changes']}/8`\n"
        f"- B0/null hard parity："
        f"`{values['b0_null_hard_parity_passed']}`\n\n"
        "## 冻结的后续动作\n\n"
        f"{next_steps[branch]}\n\n"
        "本 runner 没有启动 Phase 2、rollout、OOD 或 success 评测。\n"
    )
    path = output / "phase1_decision_report.md"
    atomic_write_text(path, text)
    return path


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(output.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_manifest.json"
            and ".tmp" not in path.name
        ):
            relative = str(path.relative_to(output))
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "created_at": _utc_now(),
        "file_count": len(files),
        "files": files,
    }


def _verify_artifact_manifest(output: Path) -> dict[str, Any]:
    manifest = load_json(output / "artifact_manifest.json")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA
        or not isinstance(files, Mapping)
        or int(manifest.get("file_count", -1)) != len(files)
    ):
        raise Phase1OnlineCounterfactualError(
            "completed-run artifact manifest schema/count mismatch"
        )
    for relative, descriptor in files.items():
        path = output / str(relative)
        try:
            path.resolve().relative_to(output.resolve())
        except ValueError as exc:
            raise Phase1OnlineCounterfactualError(
                "artifact manifest path escaped output"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != int(descriptor["bytes"])
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise Phase1OnlineCounterfactualError(
                f"completed-run artifact checksum failed: {path}"
            )
    return manifest


def _integrity_after(
    model: Any,
    adapter: Any,
    *,
    cfg: K1OnlineCounterfactualConfig,
    before: Mapping[str, Any],
) -> dict[str, Any]:
    from fastwam_ood_eval.thought3.checkpointing import adapter_state_sha256
    from fastwam_ood_eval.thought3.model_wrapper import parameter_state_sha256

    frozen_after = parameter_state_sha256(iter(model.named_parameters()))
    adapter_after = adapter_state_sha256(adapter.state_dict())
    backbone_grads = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    adapter_grads = [
        name
        for name, parameter in adapter.named_parameters()
        if parameter.grad is not None
    ]
    checks = {
        "adapter_checkpoint_unchanged": (
            adapter_after == cfg.e6_adapter_state_sha256
        ),
        "adapter_has_no_gradients": not adapter_grads,
        "backbone_has_no_gradients": not backbone_grads,
        "frozen_fastwam_unchanged": (
            frozen_after
            == before["frozen_parameter_sha256_before"]
            == cfg.frozen_parameter_sha256
        ),
        "injector_context_inactive": True,
    }
    return {
        "adapter_state_sha256_after": adapter_after,
        "backbone_gradient_names": backbone_grads,
        "adapter_gradient_names": adapter_grads,
        "checks": checks,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": before[
            "frozen_parameter_sha256_before"
        ],
        "passed": all(checks.values()),
    }


def _protocol_lock(
    cfg: K1OnlineCounterfactualConfig,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "thought3.k1_online_counterfactual.lock.v1",
        "cohort_fingerprint": cfg.cohort_fingerprint,
        "config_fingerprint": cfg.fingerprint,
        "fastwam_commit": preflight["fastwam_commit"],
        "project_commit": preflight["project_commit"],
        "shuffle_mapping_sha256": preflight[
            "shuffle_mapping_sha256"
        ],
    }


def _validate_resume_lock(
    output: Path,
    expected: Mapping[str, Any],
) -> None:
    observed = load_json(output / "protocol_lock.json")
    if observed != dict(expected):
        raise Phase1OnlineCounterfactualError(
            "resume protocol lock differs from the current frozen run"
        )


def _sample_result(
    *,
    cfg: K1OnlineCounterfactualConfig,
    output: Path,
    sample: PreparedOnlineSample,
    donor: PreparedOnlineSample,
    replay: Mapping[str, Any],
    sampler: Any,
    velocity: _CountingVideoVelocity,
    model: Any,
    injector: Any,
    adapter_timer: _AdapterCudaTimer,
) -> dict[str, Any]:
    import torch
    from fastwam_ood_eval.thought3.future_sampler import tensor_sha256

    action_seed = int(replay["action_seed_identity"]["seed"])
    b0 = _action_from_payload(replay["repeats"][0])
    b0_payload = _action_payload(
        b0,
        output=output,
        base_sample_id=sample.cohort.base_sample_id,
        condition="B0",
    )

    null_action, null_runtime = _custom_action(
        model,
        injector,
        adapter_timer,
        sample,
        cfg=cfg,
        action_seed=action_seed,
        future=None,
        formal_null=True,
    )
    null_pair = action_pair_metrics(b0, null_action)
    if float(null_pair["linf"]) > cfg.replay_hard_max_linf:
        raise Phase1OnlineCounterfactualError(
            "formal null failed B0 hard parity; online sensitivity is invalid"
        )

    future_seed = stable_online_seed(
        cfg.future_seed_namespace,
        cfg.experiment_seed,
        sample.cohort.base_sample_id,
    )
    correct_future, correct_future_runtime = _sample_online_future(
        sampler,
        velocity,
        sample,
        cfg=cfg,
        future_seed=future_seed,
    )
    correct_action, correct_runtime = _custom_action(
        model,
        injector,
        adapter_timer,
        sample,
        cfg=cfg,
        action_seed=action_seed,
        future=correct_future,
        formal_null=False,
    )
    shuffle_future, shuffle_future_runtime = _sample_online_future(
        sampler,
        velocity,
        donor,
        cfg=cfg,
        future_seed=future_seed,
    )
    if (
        correct_future_runtime["initial_state_sha256"]
        != shuffle_future_runtime["initial_state_sha256"]
    ):
        raise Phase1OnlineCounterfactualError(
            "correct/shuffle did not reuse recipient future noise"
        )
    shuffle_action, shuffle_runtime = _custom_action(
        model,
        injector,
        adapter_timer,
        sample,
        cfg=cfg,
        action_seed=action_seed,
        future=shuffle_future,
        formal_null=False,
    )
    actions = {
        "B0": b0_payload,
        "correct": _action_payload(
            correct_action,
            output=output,
            base_sample_id=sample.cohort.base_sample_id,
            condition="correct",
        ),
        "null": _action_payload(
            null_action,
            output=output,
            base_sample_id=sample.cohort.base_sample_id,
            condition="null",
        ),
        "shuffle": _action_payload(
            shuffle_action,
            output=output,
            base_sample_id=sample.cohort.base_sample_id,
            condition="shuffle",
        ),
    }
    correct_future_artifact = _tensor_artifact(
        output,
        relative_path=(
            Path("tensors/futures")
            / f"{sample.cohort.base_sample_id}.correct.safetensors"
        ),
        key="future_latent",
        tensor=correct_future,
    )
    shuffle_future_artifact = _tensor_artifact(
        output,
        relative_path=(
            Path("tensors/futures")
            / f"{sample.cohort.base_sample_id}.shuffle.safetensors"
        ),
        key="future_latent",
        tensor=shuffle_future,
    )
    pairs = {
        "b0_null": null_pair,
        "correct_null": action_pair_metrics(
            correct_action,
            null_action,
        ),
        "correct_shuffle": action_pair_metrics(
            correct_action,
            shuffle_action,
        ),
        "null_shuffle": action_pair_metrics(
            null_action,
            shuffle_action,
        ),
    }
    b0_replay_latency = list(replay["latency"])
    b0_runtime = {
        key: (
            max(float(value[key]) for value in b0_replay_latency)
            if key in {"peak_allocated_mib", "peak_reserved_mib"}
            else sum(
                float(value[key]) for value in b0_replay_latency
            )
            / len(b0_replay_latency)
        )
        for key in (
            "action_context_cache_ms",
            "action_dit_ms",
            "adapter_ms",
            "condition_total_ms",
            "future_video_dit_ms",
            "peak_allocated_mib",
            "peak_reserved_mib",
        )
    }
    row = {
        "schema_version": ONLINE_CF_SAMPLE_SCHEMA,
        "action_seed_identity": dict(replay["action_seed_identity"]),
        "actions": actions,
        "base_sample_id": sample.cohort.base_sample_id,
        "correct_null_vs_correct_shuffle_delta_cosine": (
            delta_direction_cosine(
                correct=correct_action,
                null=null_action,
                shuffle=shuffle_action,
            )
        ),
        "episode_id": sample.cohort.episode_id,
        "future": {
            "correct": {
                **correct_future_runtime,
                "artifact": correct_future_artifact,
                "decoded_rgb": False,
                "source_base_sample_id": (
                    sample.cohort.base_sample_id
                ),
                "source_episode_id": sample.cohort.episode_id,
            },
            "null": {
                "adapter_call_count": null_runtime[
                    "adapter_call_count"
                ],
                "kind": cfg.null_kind,
                "tensor_artifact": None,
                "tensor_substitute": False,
                "video_dit_call_count": 0,
            },
            "shuffle": {
                **shuffle_future_runtime,
                "artifact": shuffle_future_artifact,
                "decoded_rgb": False,
                "source_base_sample_id": (
                    donor.cohort.base_sample_id
                ),
                "source_episode_id": donor.cohort.episode_id,
            },
        },
        "future_seed_identity": {
            "experiment_seed": cfg.experiment_seed,
            "namespace": cfg.future_seed_namespace,
            "recipient_base_sample_id": (
                sample.cohort.base_sample_id
            ),
            "seed": future_seed,
        },
        "latency": {
            "B0": _latency_row(
                sample,
                b0_runtime,
                future=None,
                b0_public_api=True,
            ),
            "correct": _latency_row(
                sample,
                correct_runtime,
                future=correct_future_runtime,
            ),
            "null": _latency_row(
                sample,
                null_runtime,
                future=None,
            ),
            "shuffle": _latency_row(
                sample,
                shuffle_runtime,
                future=shuffle_future_runtime,
                donor=donor,
            ),
        },
        "pairs": pairs,
        "shuffle": {
            "donor_base_sample_id": donor.cohort.base_sample_id,
            "donor_episode_id": donor.cohort.episode_id,
            "other_episode": (
                donor.cohort.episode_id != sample.cohort.episode_id
            ),
            "only_future_latent_replaced": True,
            "recipient_action_seed_reused": True,
            "recipient_current_context_reused": True,
            "recipient_future_noise_seed_reused": True,
        },
        "source": {
            **dict(sample.source),
            "action_target_read": False,
            "development_read": False,
            "future_rgb_read": False,
            "ood_read": False,
            "rollout_started": False,
            "success_read": False,
            "training_future_cache_read": False,
        },
        "target_input_identity": {
            "context_mask_sha256": tensor_sha256(
                sample.context_mask
            ),
            "context_sha256": tensor_sha256(sample.context),
            "current_latent_sha256": tensor_sha256(
                sample.current_latent
            ),
            "current_rgb_sha256": tensor_sha256(sample.image),
            "proprio_sha256": tensor_sha256(sample.proprio),
        },
        "shuffle_donor_input_identity": {
            "base_sample_id": donor.cohort.base_sample_id,
            "context_mask_sha256": tensor_sha256(
                donor.context_mask
            ),
            "context_sha256": tensor_sha256(donor.context),
            "current_latent_sha256": tensor_sha256(
                donor.current_latent
            ),
            "current_rgb_sha256": tensor_sha256(donor.image),
            "proprio_sha256": tensor_sha256(donor.proprio),
        },
    }
    validate_online_sample_result(row)
    del (
        b0,
        null_action,
        correct_action,
        shuffle_action,
        correct_future,
        shuffle_future,
    )
    torch.cuda.empty_cache()
    return row


def run_k1_online_counterfactual(
    cfg: K1OnlineCounterfactualConfig,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Run the confirmed one-GPU Phase 1 action-sensitivity experiment."""

    if os.environ.get("CONFIRM_THOUGHT3_K1_ONLINE_CF") != "YES":
        raise Phase1OnlineCounterfactualError(
            "set CONFIRM_THOUGHT3_K1_ONLINE_CF=YES for the real GPU smoke"
        )
    import numpy as np
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or cfg.device != "cuda:0"
    ):
        raise Phase1OnlineCounterfactualError(
            "Phase 1 requires exactly one CUDA-visible GPU as logical cuda:0"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise Phase1OnlineCounterfactualError(
            "Phase 1 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    np.random.seed(cfg.experiment_seed)
    base_cfg, preflight = _preflight(cfg)
    output = ensure_thought3_output_path(cfg.output_dir)
    lock = _protocol_lock(cfg, preflight=preflight)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(
                f"online counterfactual output is not empty: {output}"
            )
        _validate_resume_lock(output, lock)
        status_path = output / "run_status.json"
        if status_path.is_file():
            status = load_json(status_path)
            if status.get("status") == "completed":
                _verify_artifact_manifest(output)
                aggregate = load_json(output / "aggregate.json")
                return {
                    "aggregate": str(output / "aggregate.json"),
                    "classification": aggregate["decision"][
                        "classification"
                    ],
                    "decision_report": str(
                        output / "phase1_decision_report.md"
                    ),
                    "model_loaded": False,
                    "phase2_started": False,
                    "resume_validation_only": True,
                    "status": "already_completed",
                }
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "protocol_lock.json", lock)
        atomic_write_json(
            output / "config_snapshot.json",
            dict(cfg.raw),
        )
        atomic_write_json(output / "preflight.json", preflight)
        atomic_write_json(
            output / "cohort_manifest.json",
            _cohort_manifest(cfg),
        )
        atomic_write_json(
            output / "shuffle_manifest.json",
            build_episode_derangement(cfg),
        )
    started_at = _utc_now()
    atomic_write_json(
        output / "run_status.json",
        {
            "schema_version": RUN_STATUS_SCHEMA,
            "started_at": started_at,
            "status": "running",
        },
    )

    model = adapter = injector = adapter_timer = None
    loaded: Mapping[str, Any] | None = None
    try:
        model, adapter, prepared, loaded = _load_and_prepare(
            cfg,
            base_cfg,
        )
        velocity = _CountingVideoVelocity(model)
        from fastwam_ood_eval.thought3.future_sampler import (
            VideoOnlyFutureSampler,
        )

        sampler = VideoOnlyFutureSampler(
            velocity,
            shift=cfg.future_shift,
            num_train_timesteps=cfg.future_num_train_timesteps,
            rand_device=cfg.rand_device,
        )
        by_id = {
            sample.cohort.base_sample_id: sample for sample in prepared
        }
        shuffle = build_episode_derangement(cfg)
        donors = {
            row["target_base_sample_id"]: by_id[
                row["donor_base_sample_id"]
            ]
            for row in shuffle["mapping"]
        }

        # Warmups are deliberately discarded and never enter formal latency.
        warmup_sample = prepared[0]
        for index in range(cfg.warmup_b0_calls):
            warmup_seed = stable_online_seed(
                f"{cfg.action_seed_namespace}-warmup",
                cfg.experiment_seed + index,
                warmup_sample.cohort.base_sample_id,
            )
            warmup_action, _ = _official_b0_action(
                model,
                warmup_sample,
                cfg=cfg,
                action_seed=warmup_seed,
            )
            del warmup_action
        replay_path = output / "b0_replay.jsonl"
        replay_rows = (
            load_jsonl(replay_path)
            if resume and replay_path.is_file()
            else []
        )
        _validate_replay_rows(replay_rows, cfg)
        for sample in prepared[len(replay_rows) :]:
            action_seed = stable_online_seed(
                cfg.action_seed_namespace,
                cfg.experiment_seed,
                sample.cohort.base_sample_id,
            )
            replay_actions: list[Any] = []
            replay_latency: list[Mapping[str, Any]] = []
            for _ in range(cfg.replay_repeats):
                action, runtime = _official_b0_action(
                    model,
                    sample,
                    cfg=cfg,
                    action_seed=action_seed,
                )
                replay_actions.append(action)
                replay_latency.append(runtime)
            replay_rows.append(
                _replay_row(
                    sample,
                    action_seed=action_seed,
                    actions=replay_actions,
                    latency=replay_latency,
                    cfg=cfg,
                )
            )
            atomic_write_jsonl(replay_path, replay_rows)
            _progress(
                "b0_replay_sample_complete",
                completed=len(replay_rows),
                total=len(prepared),
            )
        replay_floor = compute_replay_floor(replay_rows, cfg)
        atomic_write_json(output / "replay_floor.json", replay_floor)
        if not replay_floor["hard_passed"]:
            raise Phase1OnlineCounterfactualError(
                "B0 replay is nondeterministic above the frozen hard bound; "
                "no sensitivity classification was produced"
            )
        _progress(
            "b0_replay_passed",
            material_l2_threshold=replay_floor[
                "material_l2_threshold"
            ],
        )

        from fastwam_ood_eval.thought3.injection import (
            ActionEncoderFutureInjector,
        )

        injector = ActionEncoderFutureInjector(
            model.action_expert.action_encoder,
            adapter,
        )
        adapter_timer = _AdapterCudaTimer(adapter)
        for index in range(cfg.warmup_future_action_calls):
            future_seed = stable_online_seed(
                f"{cfg.future_seed_namespace}-warmup",
                cfg.experiment_seed + index,
                warmup_sample.cohort.base_sample_id,
            )
            action_seed = stable_online_seed(
                f"{cfg.action_seed_namespace}-future-warmup",
                cfg.experiment_seed + index,
                warmup_sample.cohort.base_sample_id,
            )
            warmup_future, _ = _sample_online_future(
                sampler,
                velocity,
                warmup_sample,
                cfg=cfg,
                future_seed=future_seed,
            )
            warmup_action, _ = _custom_action(
                model,
                injector,
                adapter_timer,
                warmup_sample,
                cfg=cfg,
                action_seed=action_seed,
                future=warmup_future,
                formal_null=False,
            )
            del warmup_future, warmup_action
        atomic_write_json(
            output / "warmup.json",
            {
                "b0_calls": cfg.warmup_b0_calls,
                "current_encoding_calls": loaded[
                    "preparation_warmup"
                ]["current_encoding_calls"],
                "excluded_from_formal_latency": True,
                "future_action_calls": (
                    cfg.warmup_future_action_calls
                ),
            },
        )
        _progress("warmup_complete")

        sample_path = output / "sample_results.jsonl"
        sample_rows = (
            load_jsonl(sample_path)
            if resume and sample_path.is_file()
            else []
        )
        expected_prefix = [
            sample.cohort.base_sample_id
            for sample in prepared[: len(sample_rows)]
        ]
        if [
            str(row.get("base_sample_id")) for row in sample_rows
        ] != expected_prefix:
            raise Phase1OnlineCounterfactualError(
                "sample results are not an ordered cohort prefix"
            )
        for row in sample_rows:
            validate_online_sample_result(row)
            _verify_sample_artifacts(output, row)
        for index in range(len(sample_rows), len(prepared)):
            sample = prepared[index]
            row = _sample_result(
                cfg=cfg,
                output=output,
                sample=sample,
                donor=donors[sample.cohort.base_sample_id],
                replay=replay_rows[index],
                sampler=sampler,
                velocity=velocity,
                model=model,
                injector=injector,
                adapter_timer=adapter_timer,
            )
            sample_rows.append(row)
            atomic_write_jsonl(sample_path, sample_rows)
            _progress(
                "counterfactual_sample_complete",
                completed=len(sample_rows),
                total=len(prepared),
                base_sample_id=sample.cohort.base_sample_id,
            )
        aggregate = aggregate_online_counterfactual(
            sample_rows,
            replay_floor=replay_floor,
            cfg=cfg,
        )
        integrity = _integrity_after(
            model,
            adapter,
            cfg=cfg,
            before=loaded,
        )
        integrity["checks"]["injector_context_inactive"] = (
            not injector.has_active_context
        )
        integrity["passed"] = all(integrity["checks"].values())
        atomic_write_json(output / "execution_integrity.json", integrity)
        if not integrity["passed"]:
            raise Phase1OnlineCounterfactualError(
                "post-inference frozen-state integrity failed"
            )
        atomic_write_json(output / "aggregate.json", aggregate)
        atomic_write_json(
            output / "decision.json",
            aggregate["decision"],
        )
        decision_report = _write_decision_report(
            output,
            aggregate=aggregate,
            cfg=cfg,
        )
        completed_at = _utc_now()
        status = {
            "schema_version": RUN_STATUS_SCHEMA,
            "classification": aggregate["decision"]["classification"],
            "completed_at": completed_at,
            "phase2_started": False,
            "sample_count": len(sample_rows),
            "started_at": started_at,
            "status": "completed",
        }
        atomic_write_json(output / "run_status.json", status)
        manifest = _artifact_manifest(output)
        atomic_write_json(output / "artifact_manifest.json", manifest)
        _progress(
            "completed",
            classification=aggregate["decision"]["classification"],
            next_branch=aggregate["decision"]["next_branch"],
        )
        return {
            "aggregate": str(output / "aggregate.json"),
            "artifact_manifest": str(output / "artifact_manifest.json"),
            "classification": aggregate["decision"]["classification"],
            "decision_report": str(decision_report),
            "next_branch": aggregate["decision"]["next_branch"],
            "output_dir": str(output),
            "phase2_started": False,
            "status": "completed",
        }
    except BaseException as exc:
        failure: dict[str, Any] = {
            "schema_version": RUN_STATUS_SCHEMA,
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at": _utc_now(),
            "classification_valid": False,
            "phase2_started": False,
            "started_at": started_at,
            "status": "failed_closed",
            "traceback": traceback.format_exc(),
        }
        if (
            model is not None
            and adapter is not None
            and loaded is not None
        ):
            try:
                failure["integrity_after_failure"] = _integrity_after(
                    model,
                    adapter,
                    cfg=cfg,
                    before=loaded,
                )
            except BaseException as integrity_exc:
                failure["integrity_capture_error"] = (
                    f"{type(integrity_exc).__name__}: {integrity_exc}"
                )
        atomic_write_json(output / "run_status.json", failure)
        raise
    finally:
        if adapter_timer is not None:
            adapter_timer.close()
        if injector is not None:
            injector.close()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
