"""Single-GPU real Fast-WAM Adapter training for Thought3 Gate E.

This module deliberately keeps the Phase B mock trainer unchanged.  It joins
the committed Phase D cache to standard LIBERO demonstrations by stable
``base_sample_id``, reads only the current RGB/proprio plus the action
supervision, and trains only a fresh Future-to-Action Adapter.
"""

from __future__ import annotations

import gc
import hashlib
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.cache_validator import validate_cache
from fastwam_ood_eval.thought3.checkpointing import (
    adapter_state_sha256,
    find_latest_checkpoint,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.future_cache import FutureCacheReader
from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from fastwam_ood_eval.thought3.phase_c_smoke import (
    _action_from_video_cache,
    _prepare_video_cache,
    _sample_training_t_on_cpu,
    compute_upstream_action_loss,
)
from fastwam_ood_eval.thought3.real_cache_builder import (
    CurrentOnlyLiberoSource,
    RealCacheBuildError,
    _load_prompt_context,
)
from fastwam_ood_eval.thought3.safety import (
    ensure_thought3_output_path,
)
from fastwam_ood_eval.thought3.schemas import (
    AdapterCheckpointManifest,
    CachePlanEntry,
    NATIVE_FUTURE_SHAPE,
)
from fastwam_ood_eval.thought3.training_dataset import (
    validate_training_example,
)


class RealTrainingError(RuntimeError):
    """Raised when Gate E real training violates a hard invariant."""


ProgressCallback = Callable[[str, Mapping[str, Any]], None]
DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET = 10_000
DIVERSIFIED_HELDOUT_FLOW_STEPS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class CurrentActionObservation:
    """Current-only model inputs plus the permitted action supervision."""

    image: Tensor
    proprio: Tensor
    target_action: Tensor
    action_is_pad: Tensor
    source: Mapping[str, Any]


@dataclass(frozen=True)
class RealTrainingSample:
    """CPU-resident precomputed sample used by the real Adapter trainer."""

    base_sample_id: str
    split: str
    current_latent: Tensor
    context: Tensor
    context_mask: Tensor
    target_action: Tensor
    action_is_pad: Tensor
    future_latent: Tensor
    future_mask: Tensor
    source: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRealTrainingData:
    samples: tuple[RealTrainingSample, ...]
    split_fingerprint: str
    cache_fingerprint: str
    report: Mapping[str, Any]


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _flow_objective_identity(
    *,
    base_sample_id: str,
    train_seed: int,
    flow_step: int,
) -> dict[str, Any]:
    """Return the two deterministic RNG seeds for one action-flow objective."""

    noise_seed = _stable_seed(
        "thought3-real-action-noise-v1",
        train_seed,
        flow_step,
        base_sample_id,
    )
    timestep_seed = _stable_seed(
        "thought3-real-action-time-v1",
        train_seed,
        flow_step,
        base_sample_id,
    )
    digest = hashlib.sha256(
        (
            f"{base_sample_id}\0{train_seed}\0{flow_step}\0"
            f"{noise_seed}\0{timestep_seed}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "action_noise_seed": noise_seed,
        "action_timestep_seed": timestep_seed,
        "flow_objective_sha256": digest,
        "flow_step": flow_step,
    }


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "max": None, "mean": None, "min": None}
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "max": max(normalized),
        "mean": sum(normalized) / len(normalized),
        "min": min(normalized),
    }


def preprocess_current_action_target(
    raw_action: Tensor,
    raw_state: Tensor,
    action_is_pad: Tensor,
    *,
    processor: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match FastWAMProcessor for one current state and its 32-step action.

    The official dataset normally queries 33 image/state timestamps together.
    Gate E instead supplies one current state and the same 32 action indices.
    The pinned LIBERO processor uses global min/max normalization, so this
    current-only state path is exactly equivalent to slicing the official
    processed state at time zero.
    """

    action_meta = list(processor.shape_meta["action"])
    state_meta = list(processor.shape_meta["state"])
    if len(action_meta) != 1 or len(state_meta) != 1:
        raise RealTrainingError(
            "Gate E expects one merged action and one merged state"
        )
    action_key = str(action_meta[0]["key"])
    state_key = str(state_meta[0]["key"])
    action = raw_action.detach().clone().float()
    state = raw_state.detach().clone().float()
    padding = action_is_pad.detach().clone().bool()
    if tuple(action.shape) != (32, 7):
        raise RealTrainingError(
            f"raw action target must be [32,7], got {tuple(action.shape)}"
        )
    if tuple(state.shape) == (8,):
        state = state.unsqueeze(0)
    if tuple(state.shape) != (1, 8):
        raise RealTrainingError(
            f"raw current state must be [1,8], got {tuple(state.shape)}"
        )
    if tuple(padding.shape) != (32,):
        raise RealTrainingError(
            f"action padding must be [32], got {tuple(padding.shape)}"
        )

    batch: dict[str, Any] = {
        "action": {action_key: action},
        "state": {state_key: state},
        "action_is_pad": padding,
        "state_is_pad": torch.zeros(1, dtype=torch.bool),
        "idx": 0,
    }
    delta_masks = processor.delta_action_dim_mask
    if delta_masks is not None and bool(padding.any().item()):
        for key, dimension_mask in delta_masks.items():
            current = batch["action"][key]
            pad_delta_mask = (
                padding.to(device=current.device).unsqueeze(1)
                & dimension_mask.to(device=current.device).unsqueeze(0)
            )
            current[pad_delta_mask] = 0.0

    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    batch = processor.action_state_merger.forward(batch)
    target_action = batch["action"].contiguous()
    current_proprio = batch["state"].contiguous()
    processed_padding = batch["action_is_pad"].bool().contiguous()
    if (
        tuple(target_action.shape) != (32, 7)
        or tuple(current_proprio.shape) != (1, 8)
        or tuple(processed_padding.shape) != (32,)
    ):
        raise RealTrainingError(
            "official action/state preprocessing returned an unexpected shape"
        )
    if (
        not target_action.isfinite().all()
        or not current_proprio.isfinite().all()
    ):
        raise RealTrainingError(
            "official action/state preprocessing produced NaN/Inf"
        )
    return target_action, current_proprio, processed_padding


class CurrentActionLiberoSource(CurrentOnlyLiberoSource):
    """Read current RGB/proprio and action supervision without future RGB."""

    def __init__(self, cfg: Thought3Config, upstream_cfg: Any) -> None:
        super().__init__(cfg, upstream_cfg)
        self.telemetry.update(
            {
                "action_target_chunks_read": 0,
                "action_target_rows_read": 0,
                "state_rows_read": 0,
            }
        )

    def load_training(
        self,
        entry: CachePlanEntry,
    ) -> CurrentActionObservation:
        observation = super().load(entry)
        dataset_index = int(observation.source["dataset_index"])
        identity = entry.identity
        row = self.inner.hf_dataset[dataset_index]
        episodes = self.inner.episodes
        current_episode_index = (
            episodes.index(identity.episode_index)
            if episodes is not None
            else identity.episode_index
        )
        query_indices, padding = self.inner._get_query_indices(
            dataset_index,
            current_episode_index,
        )
        base = self.dataset.lerobot_dataset
        action_meta = list(base.action_meta)
        state_meta = list(base.state_meta)
        if len(action_meta) != 1 or len(state_meta) != 1:
            raise RealTrainingError(
                "Gate E expects one raw action/state field"
            )
        action_key = str(action_meta[0]["lerobot_key"])
        state_key = str(state_meta[0]["lerobot_key"])
        action_values = self.inner._query_hf_dataset_fast(
            {action_key: query_indices[action_key]}
        )
        raw_action = action_values[action_key]
        raw_state = row[state_key]
        action_padding = padding[f"{action_key}_is_pad"]
        target_action, current_proprio, processed_padding = (
            preprocess_current_action_target(
                raw_action,
                raw_state,
                action_padding,
                processor=self.processor,
            )
        )
        if not torch.equal(current_proprio, observation.proprio):
            difference = float(
                (current_proprio - observation.proprio).abs().max()
            )
            raise RealTrainingError(
                "current-only proprio differs from action-source preprocessing: "
                f"max_abs={difference}"
            )
        self.telemetry["action_target_read"] = True
        self.telemetry["action_target_chunks_read"] += 1
        self.telemetry["action_target_rows_read"] += int(
            target_action.shape[0]
        )
        self.telemetry["state_rows_read"] += 1
        return CurrentActionObservation(
            image=observation.image,
            proprio=current_proprio,
            target_action=target_action,
            action_is_pad=processed_padding,
            source={
                **dict(observation.source),
                "action_target_rows_read": int(target_action.shape[0]),
                "actual_future_read": False,
                "future_rgb_frames_decoded": 0,
                "state_rows_read": 1,
            },
        )


def _tensor_state_sha256(tensors: Mapping[str, Tensor]) -> str:
    return adapter_state_sha256(
        {
            str(name): tensor.detach().cpu().contiguous()
            for name, tensor in tensors.items()
        }
    )


def _training_order_key(
    base_sample_id: str,
    *,
    seed: int,
) -> str:
    return hashlib.sha256(
        f"thought3-real-train-order-v1\0{seed}\0"
        f"{base_sample_id}".encode("utf-8")
    ).hexdigest()


def prepare_real_training_data(
    cfg: Thought3Config,
    *,
    model: Any,
    upstream_cfg: Any,
    device: str,
    progress: ProgressCallback | None = None,
    train_only_limit: int | None = None,
    train_only_offset: int = 0,
) -> PreparedRealTrainingData:
    """Join Phase D K=1 identities to current-only LIBERO supervision.

    Gate E/E.1 use the complete frozen 28/4 subset. Gate E.2–E.5 pass
    ``train_only_limit=8`` with offset 0. Sequential replication Gates may
    select another deterministic eight-sample train window without decoding
    development action targets.
    """

    started = time.perf_counter()
    cache_report = validate_cache(cfg.cache.root)
    entries, plan = load_cache_plan(cfg.cache.root)
    all_reference_entries = sorted(
        (entry for entry in entries if entry.k == 1),
        key=lambda entry: entry.identity.base_sample_id,
    )
    if len(all_reference_entries) != 32:
        raise RealTrainingError(
            "Gate E requires exactly 32 Phase D base samples"
        )
    available_split_counts = {
        split: sum(
            entry.split == split for entry in all_reference_entries
        )
        for split in ("train", "development")
    }
    if available_split_counts != {"train": 28, "development": 4}:
        raise RealTrainingError(
            "Phase D reference entries must contain 28 train / "
            f"4 development, got {available_split_counts}"
        )
    if train_only_limit is None:
        if train_only_offset != 0:
            raise RealTrainingError(
                "train_only_offset requires train_only_limit=8"
            )
        reference_entries = all_reference_entries
        expected_split_counts = {"train": 28, "development": 4}
        selection_mode = "complete_phase_d_subset"
    else:
        if (
            train_only_limit != 8
            or isinstance(train_only_offset, bool)
            or not isinstance(train_only_offset, int)
            or train_only_offset < 0
            or train_only_offset + train_only_limit > 28
        ):
            raise RealTrainingError(
                "real train-only preparation requires an in-range "
                "eight-sample window"
            )
        ordered_train_entries = sorted(
            (
                entry
                for entry in all_reference_entries
                if entry.split == "train"
            ),
            key=lambda entry: _training_order_key(
                entry.identity.base_sample_id,
                seed=cfg.training.train_seed,
            ),
        )
        reference_entries = ordered_train_entries[
            train_only_offset:
            train_only_offset + train_only_limit
        ]
        expected_split_counts = {
            "train": train_only_limit,
            "development": 0,
        }
        selection_mode = (
            "ordered_train_only"
            if train_only_offset == 0
            else "ordered_train_window"
        )
    if plan["cache_fingerprint"] != cache_report["cache_fingerprint"]:
        raise RealTrainingError("cache plan/validation fingerprint mismatch")
    reader = FutureCacheReader(
        cfg.cache.root,
        expected_cache_fingerprint=str(cache_report["cache_fingerprint"]),
        validate=False,
    )
    source = CurrentActionLiberoSource(cfg, upstream_cfg)
    context_base, context_mask_base, prompt = _load_prompt_context(
        model,
        reference_entries,
        device=device,
    )
    if model.text_encoder is not None:
        model.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    samples: list[RealTrainingSample] = []
    decode_latencies: list[float] = []
    encode_latencies: list[float] = []
    peak_memory_mib: list[float] = []
    for sample_index, entry in enumerate(reference_entries):
        observation = source.load_training(entry)
        image = observation.image.to(
            device=device,
            dtype=model.torch_dtype,
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        encode_started = time.perf_counter()
        with torch.inference_mode():
            current_latent = model._encode_input_image_latents_tensor(
                image
            )
        torch.cuda.synchronize(device)
        encode_latency = (
            time.perf_counter() - encode_started
        ) * 1000.0
        encode_peak = int(torch.cuda.max_memory_allocated(device)) / 2**20
        if tuple(current_latent.shape) != (1, 48, 1, 14, 28):
            raise RealTrainingError(
                "current latent must be [1,48,1,14,28]"
            )
        proprio = observation.proprio.to(
            device=device,
            dtype=model.torch_dtype,
        )
        with torch.inference_mode():
            context, context_mask = model._append_proprio_to_context(
                context_base,
                context_mask_base,
                proprio,
            )
        if tuple(context.shape) != (1, 129, 4096):
            raise RealTrainingError(
                f"context with proprio must be [1,129,4096], got {tuple(context.shape)}"
            )
        if tuple(context_mask.shape) != (1, 129):
            raise RealTrainingError(
                f"context mask must be [1,129], got {tuple(context_mask.shape)}"
            )

        future, future_mask, metadata = reader.get(
            entry.identity.base_sample_id,
            1,
        )
        record = metadata["record"]
        if (
            str(record["base_sample_id"])
            != entry.identity.base_sample_id
            or int(record["k"]) != 1
            or str(record["split"]) != entry.split
            or bool(record["uses_ground_truth_future"])
        ):
            raise RealTrainingError(
                "cache metadata does not match the training identity"
            )
        if tuple(future.shape) != NATIVE_FUTURE_SHAPE:
            raise RealTrainingError("cached future latent shape mismatch")
        if tuple(future_mask.shape) != NATIVE_FUTURE_SHAPE[1:]:
            raise RealTrainingError("cached future mask shape mismatch")
        validate_training_example(
            {
                "action_is_pad": observation.action_is_pad,
                "base_sample_id": entry.identity.base_sample_id,
                "context": context[0],
                "context_mask": context_mask[0],
                "current_proprio": observation.proprio[0],
                "current_rgb": observation.image[0],
                "future_latent": future,
                "future_mask": future_mask,
                "metadata": {
                    "base_sample_id": entry.identity.base_sample_id,
                    "source_kind": str(record["source_kind"]),
                },
                "sample_id": str(record["cache_sample_id"]),
                "target_action": observation.target_action,
            }
        )
        samples.append(
            RealTrainingSample(
                base_sample_id=entry.identity.base_sample_id,
                split=entry.split,
                current_latent=current_latent[0]
                .detach()
                .to("cpu", dtype=model.torch_dtype)
                .contiguous(),
                context=context[0]
                .detach()
                .to("cpu", dtype=model.torch_dtype)
                .contiguous(),
                context_mask=context_mask[0]
                .detach()
                .to("cpu", dtype=torch.bool)
                .contiguous(),
                target_action=observation.target_action.detach()
                .to("cpu", dtype=torch.float32)
                .contiguous(),
                action_is_pad=observation.action_is_pad.detach()
                .to("cpu", dtype=torch.bool)
                .contiguous(),
                future_latent=future.detach().contiguous(),
                future_mask=future_mask.detach().bool().contiguous(),
                source=dict(observation.source),
            )
        )
        decode_latencies.append(
            float(observation.source["current_decode_latency_ms"])
        )
        encode_latencies.append(encode_latency)
        peak_memory_mib.append(encode_peak)
        del image, current_latent, proprio, context, context_mask
        torch.cuda.empty_cache()
        if progress is not None and (
            sample_index == 0
            or (sample_index + 1) % 8 == 0
            or sample_index + 1 == len(reference_entries)
        ):
            progress(
                "training_data_prepared",
                {
                    "prepared": sample_index + 1,
                    "total": len(reference_entries),
                },
            )

    split_counts = {
        split: sum(sample.split == split for sample in samples)
        for split in ("train", "development")
    }
    if split_counts != expected_split_counts:
        raise RealTrainingError(
            "real training selected split mismatch: "
            f"{split_counts} != {expected_split_counts}"
        )
    source_telemetry = dict(source.telemetry)
    selected_count = len(reference_entries)
    expected_source = {
        "action_target_chunks_read": selected_count,
        "action_target_read": True,
        "action_target_rows_read": selected_count * 32,
        "actual_future_read": False,
        "current_camera_frames_decoded": selected_count * 2,
        "future_rgb_frames_decoded": 0,
        "state_rows_read": selected_count,
    }
    for key, expected in expected_source.items():
        if source_telemetry.get(key) != expected:
            raise RealTrainingError(
                f"current/action source telemetry mismatch for {key}: "
                f"{source_telemetry.get(key)} != {expected}"
            )
    report = {
        "available_split_counts": available_split_counts,
        "cache_fingerprint": cache_report["cache_fingerprint"],
        "current_decode_latency_ms": _summary(decode_latencies),
        "current_encode_latency_ms": _summary(encode_latencies),
        "current_source": source_telemetry,
        "future_rgb_used_as_input": False,
        "peak_memory_mib": max(peak_memory_mib),
        "preparation_wall_s": time.perf_counter() - started,
        "prompt": prompt,
        "sample_count": len(samples),
        "sample_payload_sha256": _tensor_state_sha256(
            {
                f"{sample.base_sample_id}.action": sample.target_action
                for sample in samples
            }
            | {
                f"{sample.base_sample_id}.current": sample.current_latent
                for sample in samples
            }
        ),
        "selection_mode": selection_mode,
        "split_counts": split_counts,
        "split_fingerprint": plan["split_fingerprint"],
        "train_only_limit": train_only_limit,
    }
    if train_only_offset:
        report["train_only_offset"] = train_only_offset
    return PreparedRealTrainingData(
        samples=tuple(samples),
        split_fingerprint=str(plan["split_fingerprint"]),
        cache_fingerprint=str(cache_report["cache_fingerprint"]),
        report=report,
    )


def build_real_adapter(
    cfg: Thought3Config,
    *,
    device: str,
) -> FutureToActionAdapter:
    """Initialize A0/A1 identically; variant and K never enter the seed."""

    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)
    adapter = FutureToActionAdapter(
        FutureAdapterSpec(
            input_channels=cfg.adapter.input_channels,
            action_hidden_dim=cfg.adapter.action_hidden_dim,
            future_dim=cfg.adapter.future_dim,
            attention_dim=cfg.adapter.attention_dim,
            num_heads=cfg.adapter.num_heads,
            max_projected_grid=cfg.adapter.max_projected_grid,
            zero_init_gate=cfg.adapter.zero_init_gate,
        )
    ).to(device=device)
    adapter.capture_diagnostics = True
    return adapter


def _ordered_samples(
    samples: Iterable[RealTrainingSample],
    *,
    seed: int,
) -> list[RealTrainingSample]:
    return sorted(
        samples,
        key=lambda sample: _training_order_key(
            sample.base_sample_id,
            seed=seed,
        ),
    )


def _future_for_variant(
    sample: RealTrainingSample,
    variant: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if variant == "A0":
        future = torch.zeros(
            (1, *NATIVE_FUTURE_SHAPE),
            dtype=dtype,
            device=device,
        )
    elif variant == "A1":
        future = sample.future_latent.unsqueeze(0).to(
            device=device,
            dtype=dtype,
        )
    else:
        raise RealTrainingError(
            f"Gate E supports only A0/A1, got {variant}"
        )
    mask = sample.future_mask.unsqueeze(0).to(
        device=device,
        dtype=torch.bool,
    )
    return future, mask


def _flow_inputs(
    model: Any,
    sample: RealTrainingSample,
    *,
    train_seed: int,
    step: int,
    device: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    target_action = sample.target_action.unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    identity = _flow_objective_identity(
        base_sample_id=sample.base_sample_id,
        train_seed=train_seed,
        flow_step=step,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        int(identity["action_noise_seed"])
    )
    noise = torch.randn(
        tuple(target_action.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=model.torch_dtype)
    timestep = _sample_training_t_on_cpu(
        model.train_action_scheduler,
        int(identity["action_timestep_seed"]),
        device,
        model.torch_dtype,
    )
    noisy_action = model.train_action_scheduler.add_noise(
        target_action,
        noise,
        timestep,
    )
    velocity_target = model.train_action_scheduler.training_target(
        target_action,
        noise,
        timestep,
    )
    action_weight = model.train_action_scheduler.training_weight(timestep)
    return (
        target_action,
        noisy_action,
        velocity_target,
        action_weight,
        timestep,
    )


def _flow_timestep_and_weight_scalars(
    model: Any,
    sample: RealTrainingSample,
    *,
    train_seed: int,
    step: int,
    device: str,
) -> tuple[float, float]:
    """Recreate deterministic action-flow timestep and official loss weight."""

    timestep = _sample_training_t_on_cpu(
        model.train_action_scheduler,
        int(
            _flow_objective_identity(
                base_sample_id=sample.base_sample_id,
                train_seed=train_seed,
                flow_step=step,
            )["action_timestep_seed"]
        ),
        device,
        model.torch_dtype,
    )
    action_weight = model.train_action_scheduler.training_weight(
        timestep
    )
    timestep_value = float(
        timestep.detach().float().cpu().reshape(())
    )
    action_weight_value = float(
        action_weight.detach().float().cpu().reshape(())
    )
    del action_weight, timestep
    return timestep_value, action_weight_value


def _loss_for_real_sample(
    cfg: Thought3Config,
    model: Any,
    adapter: FutureToActionAdapter,
    injector: ActionEncoderFutureInjector,
    sample: RealTrainingSample,
    *,
    step: int,
    device: str,
) -> Tensor:
    (
        target_action,
        noisy_action,
        velocity_target,
        action_weight,
        timestep_action,
    ) = _flow_inputs(
        model,
        sample,
        train_seed=cfg.training.train_seed,
        step=step,
        device=device,
    )
    context = sample.context.unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    context_mask = sample.context_mask.unsqueeze(0).to(
        device=device,
        dtype=torch.bool,
    )
    current_latent = sample.current_latent.unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    future, future_mask = _future_for_variant(
        sample,
        cfg.variant,
        device=device,
        dtype=model.torch_dtype,
    )
    with torch.no_grad():
        shape_probe = model.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_cache, attention_mask, video_seq_len = _prepare_video_cache(
            model,
            current_latent,
            context,
            context_mask,
            action_seq_len=int(shape_probe["tokens"].shape[1]),
        )
    with injector.activate(future, future_mask, expected_calls=1):
        prediction = _action_from_video_cache(
            model,
            noisy_action,
            timestep_action,
            context,
            context_mask,
            video_cache,
            attention_mask,
            video_seq_len,
        )
    action_is_pad = sample.action_is_pad.unsqueeze(0).to(
        device=device,
        dtype=torch.bool,
    )
    loss = compute_upstream_action_loss(
        prediction,
        velocity_target,
        action_is_pad,
        action_weight,
        loss_lambda_action=model.loss_lambda_action,
    )
    if cfg.training.gate_l2:
        loss = loss + cfg.training.gate_l2 * adapter.gate.square()
    del (
        target_action,
        noisy_action,
        velocity_target,
        action_weight,
        context,
        context_mask,
        current_latent,
        future,
        future_mask,
        shape_probe,
        video_cache,
        attention_mask,
        timestep_action,
        prediction,
        action_is_pad,
    )
    return loss


def adapter_gradient_groups(
    adapter: FutureToActionAdapter,
) -> dict[str, dict[str, Any]]:
    """Return finite/nonzero telemetry for gate and major Adapter paths."""

    named = dict(adapter.named_parameters())
    groups: dict[str, tuple[str, ...]] = {
        "gate": ("gate",),
        "future_projector": tuple(
            name for name in named if name.startswith("future_projector.")
        ),
        "future_token_path": tuple(
            name
            for name in named
            if name.startswith("future_norm.")
            or name in {"time_position", "height_position", "width_position"}
        ),
        "attention": tuple(
            name
            for name in named
            if name.startswith(
                (
                    "query_norm.",
                    "query_projection.",
                    "key_projection.",
                    "value_projection.",
                    "output_projection.",
                )
            )
        ),
        "non_gate": tuple(name for name in named if name != "gate"),
        "all": tuple(named),
    }
    result: dict[str, dict[str, Any]] = {}
    for group, names in groups.items():
        squared = 0.0
        parameter_squared = 0.0
        nonzero = 0
        tensor_count = 0
        missing = 0
        finite = True
        for name in names:
            parameter_value = named[name].detach().float()
            parameter_squared += float(
                parameter_value.square().sum().cpu()
            )
            gradient = named[name].grad
            if gradient is None:
                missing += 1
                continue
            value = gradient.detach().float()
            tensor_count += 1
            finite = finite and bool(torch.isfinite(value).all().item())
            squared += float(value.square().sum().cpu())
            nonzero += int(torch.count_nonzero(value).item())
        gradient_l2 = math.sqrt(squared)
        parameter_l2 = math.sqrt(parameter_squared)
        result[group] = {
            "finite": finite,
            "gradient_to_parameter_l2_ratio": (
                gradient_l2 / parameter_l2
                if parameter_l2 > 0
                else None
            ),
            "l2": gradient_l2,
            "missing_tensor_count": missing,
            "nonzero_element_count": nonzero,
            "parameter_l2": parameter_l2,
            "parameter_tensor_count": len(names),
            "present_gradient_tensor_count": tensor_count,
        }
    return result


def probe_two_step_determinism(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    device: str,
) -> dict[str, Any]:
    """Run two in-memory optimizer steps for an exact CUDA replay preflight."""

    train_samples = _ordered_samples(
        (sample for sample in prepared.samples if sample.split == "train"),
        seed=cfg.training.train_seed,
    )
    if len(train_samples) != 28:
        raise RealTrainingError(
            "two-step determinism probe requires 28 training samples"
        )
    adapter = build_real_adapter(cfg, device=device)
    initial_sha256 = adapter_state_sha256(adapter.state_dict())
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    rows: list[dict[str, Any]] = []
    try:
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                train_samples[step],
                step=step,
                device=device,
            )
            loss.backward()
            groups = adapter_gradient_groups(adapter)
            if not all(bool(value["finite"]) for value in groups.values()):
                raise RealTrainingError(
                    "determinism probe produced a non-finite gradient"
                )
            if step == 0 and (
                float(groups["gate"]["l2"]) <= 0
                or int(groups["non_gate"]["nonzero_element_count"]) != 0
            ):
                raise RealTrainingError(
                    "determinism probe failed first-step zero-gate contract"
                )
            if step == 1 and (
                int(
                    groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                <= 0
                or int(
                    groups["attention"]["nonzero_element_count"]
                )
                <= 0
            ):
                raise RealTrainingError(
                    "determinism probe failed second-step gradient contract"
                )
            optimizer.step()
            torch.cuda.synchronize(device)
            rows.append(
                {
                    "gate": float(adapter.gate.detach().cpu()),
                    "gradient_groups": groups,
                    "loss": float(loss.detach().cpu()),
                    "step": step + 1,
                }
            )
            del loss
        return {
            "final_adapter_sha256": adapter_state_sha256(
                adapter.state_dict()
            ),
            "initial_adapter_sha256": initial_sha256,
            "optimizer_steps": 2,
            "rows": rows,
            "variant": cfg.variant,
        }
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_real_action_loss(
    cfg: Thought3Config,
    model: Any,
    adapter: FutureToActionAdapter,
    injector: ActionEncoderFutureInjector,
    samples: Sequence[RealTrainingSample],
    *,
    device: str,
    evaluation_step_base: int = 90_000,
) -> float:
    adapter.eval()
    values: list[float] = []
    for index, sample in enumerate(samples):
        loss = _loss_for_real_sample(
            cfg,
            model,
            adapter,
            injector,
            sample,
            step=evaluation_step_base + index,
            device=device,
        )
        values.append(float(loss.detach().cpu()))
    adapter.train()
    if not values or not all(math.isfinite(value) for value in values):
        raise RealTrainingError("development action loss is empty/non-finite")
    return sum(values) / len(values)


def _checkpoint_manifest(
    cfg: Thought3Config,
    adapter: FutureToActionAdapter,
    *,
    split_fingerprint: str,
    cache_fingerprint: str,
    frozen_parameter_sha256: str,
    global_step: int,
    sample_cursor: int,
    train_sample_count: int,
    extra: Mapping[str, Any] | None = None,
) -> AdapterCheckpointManifest:
    trainable_names = tuple(
        f"adapter.{name}"
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    )
    checkpoint_extra: dict[str, Any] = {
        "action_loss": "official_fastwam_flow_matching_velocity_mse",
        "backend": "fastwam",
        "contains_backbone": False,
        "future_source_kind": "model_sampled_from_current",
        "gate_e_smoke": True,
        "uses_ground_truth_future_input": False,
    }
    checkpoint_extra.update(dict(extra or {}))
    return AdapterCheckpointManifest(
        backbone_checkpoint_sha256=cfg.backbone.checkpoint_sha256,
        dataset_stats_sha256=cfg.backbone.dataset_stats_sha256,
        fastwam_commit=cfg.backbone.fastwam_commit,
        adapter_fingerprint=cfg.adapter_structural_fingerprint,
        config_fingerprint=cfg.fingerprint,
        split_fingerprint=split_fingerprint,
        cache_fingerprint=cache_fingerprint,
        variant=cfg.variant,
        k=cfg.sampler.active_k,
        train_seed=cfg.training.train_seed,
        global_step=global_step,
        epoch=sample_cursor // train_sample_count,
        sample_cursor=sample_cursor,
        trainable_parameter_count=adapter.trainable_parameter_count,
        trainable_parameter_names=trainable_names,
        frozen_parameter_sha256=frozen_parameter_sha256,
        world_size=1,
        extra=checkpoint_extra,
    )


def _checkpoint_expected(
    cfg: Thought3Config,
    prepared: PreparedRealTrainingData,
    *,
    frozen_parameter_sha256: str,
) -> dict[str, Any]:
    return {
        "adapter_fingerprint": cfg.adapter_structural_fingerprint,
        "backbone_checkpoint_sha256": cfg.backbone.checkpoint_sha256,
        "cache_fingerprint": prepared.cache_fingerprint,
        "config_fingerprint": cfg.fingerprint,
        "dataset_stats_sha256": cfg.backbone.dataset_stats_sha256,
        "fastwam_commit": cfg.backbone.fastwam_commit,
        "frozen_parameter_sha256": frozen_parameter_sha256,
        "k": cfg.sampler.active_k,
        "split_fingerprint": prepared.split_fingerprint,
        "variant": cfg.variant,
    }


def _metric_rows_for_resume(
    path: Path,
    *,
    start_step: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = load_jsonl(path)
    steps = [int(row["global_step"]) for row in rows]
    if steps != list(range(1, len(steps) + 1)):
        raise RealTrainingError("existing Gate E metrics are not contiguous")
    if len(rows) < start_step:
        raise RealTrainingError(
            "checkpoint step exceeds committed metric rows"
        )
    return [row for row in rows if int(row["global_step"]) <= start_step]


def _validation_rows_for_resume(
    path: Path,
    *,
    start_step: int,
    initial_development_loss: float,
    initial_training_probe_loss: float,
) -> list[dict[str, Any]]:
    if not path.is_file():
        if start_step:
            raise RealTrainingError(
                "checkpoint exists without development-loss history"
            )
        return []
    rows = load_jsonl(path)
    if not rows:
        raise RealTrainingError("development-loss history is empty")
    steps = [int(row["global_step"]) for row in rows]
    if steps[0] != 0 or steps != sorted(set(steps)):
        raise RealTrainingError(
            "development-loss history is not ordered and unique"
        )
    if (
        float(rows[0]["action_loss"]) != initial_development_loss
        or float(rows[0]["training_probe_action_loss"])
        != initial_training_probe_loss
    ):
        raise RealTrainingError(
            "loss history disagrees with initial training state"
        )
    return [row for row in rows if int(row["global_step"]) <= start_step]


def _checkpoint_roundtrip(
    cfg: Thought3Config,
    adapter: FutureToActionAdapter,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
    *,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    device: str,
) -> dict[str, Any]:
    live_hash = adapter_state_sha256(adapter.state_dict())
    restored = build_real_adapter(cfg, device=device)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    manifest = load_adapter_checkpoint(
        checkpoint,
        adapter=restored,
        optimizer=restored_optimizer,
        expected=_checkpoint_expected(
            cfg,
            prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
        ),
    )
    restored_hash = adapter_state_sha256(restored.state_dict())
    optimizer_state_entries = len(restored_optimizer.state)
    if live_hash != restored_hash or optimizer_state_entries == 0:
        raise RealTrainingError(
            "Adapter/optimizer checkpoint round-trip mismatch"
        )
    del restored_optimizer, restored
    torch.cuda.empty_cache()
    return {
        "adapter_state_sha256": live_hash,
        "checkpoint": str(checkpoint),
        "global_step": manifest.global_step,
        "optimizer_state_entries": optimizer_state_entries,
        "state_equal": True,
    }


def run_real_variant_training(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
    device: str,
    stop_after_steps: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train one A0/A1 variant, with restartable Adapter-only checkpoints."""

    if cfg.runtime.backend != "fastwam" or cfg.variant not in {"A0", "A1"}:
        raise RealTrainingError("real Gate E trainer supports fastwam A0/A1 only")
    if device != "cuda:0" or cfg.runtime.device != device:
        raise RealTrainingError("Gate E requires logical cuda:0")
    if cfg.training.microbatch_size != 1:
        raise RealTrainingError("Gate E is frozen to microbatch_size=1")
    if cfg.training.gradient_accumulation_steps != 1:
        raise RealTrainingError(
            "Gate E is frozen to gradient_accumulation_steps=1"
        )

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    status_path = output / "run_status.json"
    metrics_path = output / "train_metrics.jsonl"
    validation_path = output / "development_metrics.jsonl"
    state_path = output / "training_state.json"
    manifest_path = output / "training_manifest.json"
    checkpoints_root = output / "checkpoints"
    if manifest_path.is_file() and resume:
        existing = load_json(manifest_path)
        if (
            existing.get("status") == "complete"
            and int(existing.get("completed_steps", -1))
            == cfg.training.max_steps
        ):
            return existing
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(
            f"Gate E variant output exists; pass --resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "started_at_unix_s": time.time(),
            "status": "running",
            "variant": cfg.variant,
        },
    )

    train_samples = _ordered_samples(
        (sample for sample in prepared.samples if sample.split == "train"),
        seed=cfg.training.train_seed,
    )
    development_samples = _ordered_samples(
        (
            sample
            for sample in prepared.samples
            if sample.split == "development"
        ),
        seed=cfg.training.train_seed,
    )
    if len(train_samples) != 28 or len(development_samples) != 4:
        raise RealTrainingError("Gate E selected split must be 28 train / 4 development")
    training_probe_samples = train_samples[:4]

    adapter = build_real_adapter(cfg, device=device)
    initial_adapter_sha256 = adapter_state_sha256(adapter.state_dict())
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    adapter_ids = {id(parameter) for parameter in adapter.parameters()}
    if optimizer_ids != adapter_ids:
        raise RealTrainingError("optimizer contains non-Adapter parameters")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RealTrainingError("frozen Fast-WAM parameter became trainable")

    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    start_step = 0
    sample_cursor = 0
    latest = find_latest_checkpoint(checkpoints_root) if resume else None
    if latest is not None:
        loaded = load_adapter_checkpoint(
            latest,
            adapter=adapter,
            optimizer=optimizer,
            expected=_checkpoint_expected(
                cfg,
                prepared,
                frozen_parameter_sha256=frozen_parameter_sha256,
            ),
        )
        start_step = loaded.global_step
        sample_cursor = loaded.sample_cursor
    elif resume and checkpoints_root.exists() and any(checkpoints_root.iterdir()):
        raise RealTrainingError(
            "resume requested but no valid committed checkpoint exists"
        )

    existing_metrics = _metric_rows_for_resume(
        metrics_path,
        start_step=start_step,
    )
    if state_path.is_file():
        state = load_json(state_path)
        if (
            state["config_fingerprint"] != cfg.fingerprint
            or state["initial_adapter_sha256"] != initial_adapter_sha256
            or state["frozen_parameter_sha256"]
            != frozen_parameter_sha256
        ):
            raise RealTrainingError("Gate E training-state provenance mismatch")
        initial_validation_loss = float(
            state["initial_validation_action_loss"]
        )
        initial_training_probe_loss = float(
            state["initial_training_probe_action_loss"]
        )
    else:
        if start_step:
            raise RealTrainingError(
                "checkpoint exists without Gate E initial training state"
            )
        initial_validation_loss = evaluate_real_action_loss(
            cfg,
            model,
            adapter,
            injector,
            development_samples,
            device=device,
        )
        initial_training_probe_loss = evaluate_real_action_loss(
            cfg,
            model,
            adapter,
            injector,
            training_probe_samples,
            device=device,
            evaluation_step_base=80_000,
        )
        state = {
            "cache_fingerprint": prepared.cache_fingerprint,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "frozen_parameter_sha256": frozen_parameter_sha256,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_training_probe_action_loss": (
                initial_training_probe_loss
            ),
            "initial_validation_action_loss": initial_validation_loss,
            "split_fingerprint": prepared.split_fingerprint,
            "variant": cfg.variant,
        }
        atomic_write_json(state_path, state)

    validation_rows = _validation_rows_for_resume(
        validation_path,
        start_step=start_step,
        initial_development_loss=initial_validation_loss,
        initial_training_probe_loss=initial_training_probe_loss,
    )
    if not validation_rows:
        validation_rows = [
            {
                "action_loss": initial_validation_loss,
                "checkpoint": None,
                "global_step": 0,
                "selection_split": "development",
                "training_probe_action_loss": (
                    initial_training_probe_loss
                ),
                "variant": cfg.variant,
            }
        ]
        atomic_write_jsonl(validation_path, validation_rows)
    if (
        start_step
        and all(
            int(row["global_step"]) != start_step
            for row in validation_rows
        )
    ):
        resumed_validation_loss = evaluate_real_action_loss(
            cfg,
            model,
            adapter,
            injector,
            development_samples,
            device=device,
        )
        resumed_training_probe_loss = evaluate_real_action_loss(
            cfg,
            model,
            adapter,
            injector,
            training_probe_samples,
            device=device,
            evaluation_step_base=80_000,
        )
        latest_for_validation = find_latest_checkpoint(checkpoints_root)
        if latest_for_validation is None:
            raise RealTrainingError(
                "cannot backfill development loss without a checkpoint"
            )
        validation_rows.append(
            {
                "action_loss": resumed_validation_loss,
                "checkpoint": str(latest_for_validation),
                "global_step": start_step,
                "selection_split": "development",
                "training_probe_action_loss": (
                    resumed_training_probe_loss
                ),
                "variant": cfg.variant,
            }
        )
        atomic_write_jsonl(validation_path, validation_rows)

    execution_stop = cfg.training.max_steps
    if stop_after_steps is not None:
        if stop_after_steps <= 0:
            raise RealTrainingError("stop_after_steps must be positive")
        execution_stop = min(execution_stop, stop_after_steps)
    if execution_stop < start_step:
        raise RealTrainingError(
            "requested stop precedes the resumed checkpoint"
        )

    started = time.perf_counter()
    new_metrics: list[dict[str, Any]] = []
    first_non_gate_step: int | None = None
    first_projector_step: int | None = None
    first_attention_step: int | None = None
    for row in existing_metrics:
        step_value = int(row["global_step"])
        groups = row["gradient_groups"]
        if (
            first_non_gate_step is None
            and int(groups["non_gate"]["nonzero_element_count"]) > 0
        ):
            first_non_gate_step = step_value
        if (
            first_projector_step is None
            and int(groups["future_projector"]["nonzero_element_count"]) > 0
        ):
            first_projector_step = step_value
        if (
            first_attention_step is None
            and int(groups["attention"]["nonzero_element_count"]) > 0
        ):
            first_attention_step = step_value

    try:
        for step in range(start_step, execution_stop):
            sample = train_samples[sample_cursor % len(train_samples)]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            gate_before = float(adapter.gate.detach().cpu())
            loss = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=step,
                device=device,
            )
            if not bool(torch.isfinite(loss).item()):
                raise RealTrainingError("real action loss is NaN/Inf")
            loss.backward()
            gradient_groups = adapter_gradient_groups(adapter)
            if not all(
                bool(group["finite"])
                for group in gradient_groups.values()
            ):
                raise RealTrainingError("Adapter gradient contains NaN/Inf")
            backbone_grad_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if backbone_grad_names:
                raise RealTrainingError(
                    "frozen Fast-WAM received gradients: "
                    f"{backbone_grad_names[:5]}"
                )
            global_step = step + 1
            if global_step == 1:
                if (
                    float(gradient_groups["gate"]["l2"]) <= 0
                    or int(
                        gradient_groups["non_gate"][
                            "nonzero_element_count"
                        ]
                    )
                    != 0
                ):
                    raise RealTrainingError(
                        "zero-gate first-step gradient contract failed"
                    )
            if (
                first_non_gate_step is None
                and int(
                    gradient_groups["non_gate"]["nonzero_element_count"]
                )
                > 0
            ):
                first_non_gate_step = global_step
            if (
                first_projector_step is None
                and int(
                    gradient_groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                > 0
            ):
                first_projector_step = global_step
            if (
                first_attention_step is None
                and int(
                    gradient_groups["attention"][
                        "nonzero_element_count"
                    ]
                )
                > 0
            ):
                first_attention_step = global_step
            optimizer.step()
            sample_cursor += 1
            torch.cuda.synchronize(device)
            diagnostics = adapter.last_diagnostics
            gate_after = float(adapter.gate.detach().cpu())
            row = {
                "attention_residual_norm": (
                    diagnostics.attention_residual_norm
                    if diagnostics is not None
                    else None
                ),
                "base_sample_id": sample.base_sample_id,
                "device": device,
                "future_token_norm": (
                    diagnostics.future_token_norm
                    if diagnostics is not None
                    else None
                ),
                "gate_raw_after_step": gate_after,
                "gate_raw_before_step": gate_before,
                "gate_scale_after_step": math.tanh(gate_after),
                "global_step": global_step,
                "gradient_groups": gradient_groups,
                "loss": float(loss.detach().cpu()),
                "nan_or_inf": False,
                "peak_memory_mib": (
                    int(torch.cuda.max_memory_allocated(device)) / 2**20
                ),
                "sample_cursor": sample_cursor,
                "step_time_ms": (
                    time.perf_counter() - step_started
                )
                * 1000.0,
                "trainable_parameter_count": adapter.trainable_parameter_count,
                "variant": cfg.variant,
            }
            if not all(
                math.isfinite(float(row[key]))
                for key in (
                    "loss",
                    "gate_raw_after_step",
                    "gate_scale_after_step",
                    "step_time_ms",
                    "peak_memory_mib",
                )
            ):
                raise RealTrainingError(
                    "Gate E training metric contains NaN/Inf"
                )
            new_metrics.append(row)
            should_checkpoint = (
                global_step % cfg.training.checkpoint_interval == 0
                or global_step == execution_stop
            )
            if should_checkpoint:
                # Metrics may safely exist ahead of the last committed
                # checkpoint: resume trims them back to checkpoint step.
                # The reverse ordering could leave a checkpoint without its
                # diagnostic row after a process interruption.
                atomic_write_jsonl(
                    metrics_path,
                    [*existing_metrics, *new_metrics],
                )
                checkpoint = checkpoints_root / f"step_{global_step:08d}"
                save_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    optimizer=optimizer,
                    manifest=_checkpoint_manifest(
                        cfg,
                        adapter,
                        split_fingerprint=prepared.split_fingerprint,
                        cache_fingerprint=prepared.cache_fingerprint,
                        frozen_parameter_sha256=frozen_parameter_sha256,
                        global_step=global_step,
                        sample_cursor=sample_cursor,
                        train_sample_count=len(train_samples),
                    ),
                )
                del loss
                optimizer.zero_grad(set_to_none=True)
                checkpoint_validation_loss = evaluate_real_action_loss(
                    cfg,
                    model,
                    adapter,
                    injector,
                    development_samples,
                    device=device,
                )
                checkpoint_training_probe_loss = (
                    evaluate_real_action_loss(
                        cfg,
                        model,
                        adapter,
                        injector,
                        training_probe_samples,
                        device=device,
                        evaluation_step_base=80_000,
                    )
                )
                validation_rows.append(
                    {
                        "action_loss": checkpoint_validation_loss,
                        "checkpoint": str(checkpoint),
                        "global_step": global_step,
                        "selection_split": "development",
                        "training_probe_action_loss": (
                            checkpoint_training_probe_loss
                        ),
                        "variant": cfg.variant,
                    }
                )
                atomic_write_jsonl(validation_path, validation_rows)
                if progress is not None:
                    progress(
                        "training_checkpoint",
                        {
                            "development_action_loss": (
                                checkpoint_validation_loss
                            ),
                            "gate": gate_after,
                            "loss": row["loss"],
                            "step": global_step,
                            "training_probe_action_loss": (
                                checkpoint_training_probe_loss
                            ),
                            "variant": cfg.variant,
                        },
                    )

            else:
                del loss
            torch.cuda.empty_cache()

        atomic_write_jsonl(
            metrics_path,
            [*existing_metrics, *new_metrics],
        )
        final_validation_rows = [
            row
            for row in validation_rows
            if int(row["global_step"]) == execution_stop
        ]
        if len(final_validation_rows) != 1:
            raise RealTrainingError(
                "final checkpoint has no unique development loss"
            )
        final_validation_loss = float(
            final_validation_rows[0]["action_loss"]
        )
        final_training_probe_loss = float(
            final_validation_rows[0]["training_probe_action_loss"]
        )
        selectable_validation = [
            row
            for row in validation_rows
            if 0 < int(row["global_step"]) <= execution_stop
        ]
        if not selectable_validation:
            raise RealTrainingError(
                "no checkpoint is eligible for development-only selection"
            )
        selected_validation = min(
            selectable_validation,
            key=lambda row: (
                float(row["action_loss"]),
                int(row["global_step"]),
            ),
        )
        latest_checkpoint = find_latest_checkpoint(checkpoints_root)
        if latest_checkpoint is None:
            raise RealTrainingError("Gate E wrote no Adapter checkpoint")
        roundtrip = _checkpoint_roundtrip(
            cfg,
            adapter,
            optimizer,
            latest_checkpoint,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            device=device,
        )
        all_metrics = [*existing_metrics, *new_metrics]
        result = {
            "adapter_fingerprint": cfg.adapter_structural_fingerprint,
            "cache_fingerprint": prepared.cache_fingerprint,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_roundtrip": roundtrip,
            "completed_steps": execution_stop,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "device": device,
            "best_validation_action_loss": float(
                selected_validation["action_loss"]
            ),
            "development_metrics": str(validation_path),
            "development_loss_decreased": (
                final_validation_loss < initial_validation_loss
            ),
            "final_validation_action_loss": final_validation_loss,
            "final_training_probe_action_loss": (
                final_training_probe_loss
            ),
            "first_attention_nonzero_gradient_step": first_attention_step,
            "first_non_gate_nonzero_gradient_step": first_non_gate_step,
            "first_projector_nonzero_gradient_step": first_projector_step,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_training_probe_action_loss": (
                initial_training_probe_loss
            ),
            "initial_validation_action_loss": initial_validation_loss,
            "loss_decreased": (
                final_training_probe_loss
                < initial_training_probe_loss
            ),
            "max_peak_memory_mib": max(
                float(row["peak_memory_mib"]) for row in all_metrics
            ),
            "mean_step_time_ms": sum(
                float(row["step_time_ms"]) for row in all_metrics
            )
            / len(all_metrics),
            "metrics": str(metrics_path),
            "optimizer_parameter_scope": "adapter_only",
            "resumed_from_step": start_step,
            "runtime_seconds_this_invocation": (
                time.perf_counter() - started
            ),
            "sample_cursor": sample_cursor,
            "selected_checkpoint": str(
                selected_validation["checkpoint"]
            ),
            "selected_checkpoint_global_step": int(
                selected_validation["global_step"]
            ),
            "selection_metric": "development_action_loss",
            "selection_split": "development",
            "split_fingerprint": prepared.split_fingerprint,
            "status": (
                "complete"
                if execution_stop == cfg.training.max_steps
                else "intentional_gate_interruption"
            ),
            "train_sample_count": len(train_samples),
            "training_probe_loss_decreased": (
                final_training_probe_loss
                < initial_training_probe_loss
            ),
            "training_probe_sample_count": len(
                training_probe_samples
            ),
            "trainable_parameter_count": adapter.trainable_parameter_count,
            "uses_ground_truth_future_input": False,
            "validation_sample_count": len(development_samples),
            "variant": cfg.variant,
        }
        atomic_write_json(manifest_path, result)
        atomic_write_json(
            status_path,
            {
                "completed_steps": execution_stop,
                "finished_at_unix_s": time.time(),
                "status": result["status"],
                "variant": cfg.variant,
            },
        )
        return result
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_unix_s": time.time(),
                "status": "failed",
                "variant": cfg.variant,
            },
        )
        raise
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()


def run_fixed_sample_overfit(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
    device: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Overfit one real sample at one fixed action noise/timestep.

    This is a diagnostic, not a model-selection or generalization result.  It
    deliberately removes sample/noise variation to determine whether the
    Adapter injection and optimizer can reduce one exact official Fast-WAM
    flow-matching objective.
    """

    if (
        cfg.runtime.backend != "fastwam"
        or cfg.variant not in {"A0", "A1"}
        or cfg.training.max_steps != 200
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
        or device != "cuda:0"
        or cfg.runtime.device != device
    ):
        raise RealTrainingError(
            "fixed-sample diagnostic requires real A0/A1, cuda:0, 200 steps"
        )
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    status_path = output / "run_status.json"
    metrics_path = output / "overfit_metrics.jsonl"
    state_path = output / "overfit_state.json"
    manifest_path = output / "overfit_manifest.json"
    checkpoints_root = output / "checkpoints"
    if manifest_path.is_file() and resume:
        existing = load_json(manifest_path)
        if (
            existing.get("status") == "complete"
            and int(existing.get("completed_steps", -1))
            == cfg.training.max_steps
        ):
            return existing
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(
            f"overfit diagnostic output exists: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "started_at_unix_s": time.time(),
            "status": "running",
            "variant": cfg.variant,
        },
    )

    samples = _ordered_samples(
        (sample for sample in prepared.samples if sample.split == "train"),
        seed=cfg.training.train_seed,
    )
    if len(samples) != 28:
        raise RealTrainingError(
            "fixed-sample diagnostic requires 28 training samples"
        )
    sample = samples[0]
    fixed_flow_step = 0
    adapter = build_real_adapter(cfg, device=device)
    initial_adapter_sha256 = adapter_state_sha256(adapter.state_dict())
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    if {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } != {id(parameter) for parameter in adapter.parameters()}:
        raise RealTrainingError(
            "overfit optimizer contains non-Adapter parameters"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RealTrainingError(
            "frozen Fast-WAM parameter became trainable"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    start_step = 0
    latest = find_latest_checkpoint(checkpoints_root) if resume else None
    if latest is not None:
        loaded = load_adapter_checkpoint(
            latest,
            adapter=adapter,
            optimizer=optimizer,
            expected=_checkpoint_expected(
                cfg,
                prepared,
                frozen_parameter_sha256=frozen_parameter_sha256,
            ),
        )
        start_step = loaded.global_step
    elif resume and checkpoints_root.exists() and any(checkpoints_root.iterdir()):
        raise RealTrainingError(
            "overfit resume requested without a valid checkpoint"
        )
    existing_metrics = _metric_rows_for_resume(
        metrics_path,
        start_step=start_step,
    )
    if state_path.is_file():
        state = load_json(state_path)
        if (
            state["base_sample_id"] != sample.base_sample_id
            or state["config_fingerprint"] != cfg.fingerprint
            or state["frozen_parameter_sha256"]
            != frozen_parameter_sha256
            or state["initial_adapter_sha256"]
            != initial_adapter_sha256
        ):
            raise RealTrainingError(
                "overfit diagnostic state provenance mismatch"
            )
        initial_loss = float(state["initial_action_loss"])
    else:
        if start_step:
            raise RealTrainingError(
                "overfit checkpoint exists without initial state"
            )
        with torch.no_grad():
            initial_tensor = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=fixed_flow_step,
                device=device,
            )
        initial_loss = float(initial_tensor.detach().cpu())
        diagnostics = adapter.last_diagnostics
        if (
            diagnostics is None
            or diagnostics.gated_delta_norm != 0
            or diagnostics.gated_delta_nonzero_fraction != 0
        ):
            raise RealTrainingError(
                "zero-gate overfit initialization is not exact identity"
            )
        state = {
            "base_sample_id": sample.base_sample_id,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "fixed_action_flow_step": fixed_flow_step,
            "frozen_parameter_sha256": frozen_parameter_sha256,
            "initial_action_loss": initial_loss,
            "initial_adapter_sha256": initial_adapter_sha256,
            "uses_ground_truth_future_input": False,
            "variant": cfg.variant,
        }
        atomic_write_json(state_path, state)
        del initial_tensor

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    first_non_gate_step: int | None = None
    for row in existing_metrics:
        if (
            first_non_gate_step is None
            and int(
                row["gradient_groups"]["non_gate"][
                    "nonzero_element_count"
                ]
            )
            > 0
        ):
            first_non_gate_step = int(row["global_step"])
    try:
        for step in range(start_step, cfg.training.max_steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            gate_before = float(adapter.gate.detach().cpu())
            loss = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=fixed_flow_step,
                device=device,
            )
            loss.backward()
            groups = adapter_gradient_groups(adapter)
            if not all(bool(value["finite"]) for value in groups.values()):
                raise RealTrainingError(
                    "overfit diagnostic gradient is non-finite"
                )
            global_step = step + 1
            if global_step == 1 and (
                float(groups["gate"]["l2"]) <= 0
                or int(groups["non_gate"]["nonzero_element_count"]) != 0
            ):
                raise RealTrainingError(
                    "overfit first-step zero-gate contract failed"
                )
            if (
                first_non_gate_step is None
                and int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                > 0
            ):
                first_non_gate_step = global_step
            backbone_grads = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if backbone_grads:
                raise RealTrainingError(
                    f"overfit backbone received gradients: {backbone_grads[:5]}"
                )
            diagnostics = adapter.last_diagnostics
            if diagnostics is None:
                raise RealTrainingError(
                    "overfit Adapter diagnostics are missing"
                )
            optimizer.step()
            torch.cuda.synchronize(device)
            row = {
                "action_hidden_norm": diagnostics.action_hidden_norm,
                "attention_residual_norm": (
                    diagnostics.attention_residual_norm
                ),
                "base_sample_id": sample.base_sample_id,
                "fixed_action_flow_step": fixed_flow_step,
                "future_token_norm": diagnostics.future_token_norm,
                "gate_raw_after_step": float(
                    adapter.gate.detach().cpu()
                ),
                "gate_raw_before_step": gate_before,
                "gated_delta_nonzero_fraction": (
                    diagnostics.gated_delta_nonzero_fraction
                ),
                "gated_delta_norm": diagnostics.gated_delta_norm,
                "global_step": global_step,
                "gate_gradient": float(
                    adapter.gate.grad.detach().float().cpu()
                ),
                "gate_gradient_sign": (
                    1
                    if float(adapter.gate.grad.detach().float().cpu()) > 0
                    else -1
                    if float(adapter.gate.grad.detach().float().cpu()) < 0
                    else 0
                ),
                "gradient_groups": groups,
                "loss": float(loss.detach().cpu()),
                "nan_or_inf": False,
                "peak_memory_mib": (
                    int(torch.cuda.max_memory_allocated(device)) / 2**20
                ),
                "step_time_ms": (
                    time.perf_counter() - step_started
                )
                * 1000.0,
                "variant": cfg.variant,
            }
            rows.append(row)
            should_checkpoint = (
                global_step % cfg.training.checkpoint_interval == 0
                or global_step == cfg.training.max_steps
            )
            if should_checkpoint:
                atomic_write_jsonl(
                    metrics_path,
                    [*existing_metrics, *rows],
                )
                checkpoint = checkpoints_root / f"step_{global_step:08d}"
                save_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    optimizer=optimizer,
                    manifest=_checkpoint_manifest(
                        cfg,
                        adapter,
                        split_fingerprint=prepared.split_fingerprint,
                        cache_fingerprint=prepared.cache_fingerprint,
                        frozen_parameter_sha256=frozen_parameter_sha256,
                        global_step=global_step,
                        sample_cursor=global_step,
                        train_sample_count=1,
                    ),
                )
                if progress is not None:
                    progress(
                        "overfit_checkpoint",
                        {
                            "gate": row["gate_raw_after_step"],
                            "loss": row["loss"],
                            "step": global_step,
                            "variant": cfg.variant,
                        },
                    )
            del loss
            torch.cuda.empty_cache()

        atomic_write_jsonl(
            metrics_path,
            [*existing_metrics, *rows],
        )
        with torch.no_grad():
            final_tensor = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=fixed_flow_step,
                device=device,
            )
        final_loss = float(final_tensor.detach().cpu())
        final_diagnostics = adapter.last_diagnostics
        if final_diagnostics is None:
            raise RealTrainingError(
                "overfit final diagnostics are missing"
            )
        latest_checkpoint = find_latest_checkpoint(checkpoints_root)
        if latest_checkpoint is None:
            raise RealTrainingError(
                "overfit diagnostic wrote no checkpoint"
            )
        roundtrip = _checkpoint_roundtrip(
            cfg,
            adapter,
            optimizer,
            latest_checkpoint,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            device=device,
        )
        all_metrics = [*existing_metrics, *rows]
        result = {
            "base_sample_id": sample.base_sample_id,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_roundtrip": roundtrip,
            "completed_steps": cfg.training.max_steps,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "device": device,
            "final_action_loss": final_loss,
            "final_gated_delta_nonzero_fraction": (
                final_diagnostics.gated_delta_nonzero_fraction
            ),
            "final_gated_delta_norm": (
                final_diagnostics.gated_delta_norm
            ),
            "first_non_gate_nonzero_gradient_step": (
                first_non_gate_step
            ),
            "fixed_action_flow_step": fixed_flow_step,
            "future_input_kind": (
                "zero_null_latent"
                if cfg.variant == "A0"
                else "phase_d_k1_cached_latent"
            ),
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_action_loss": initial_loss,
            "loss_reduction_fraction": (
                (initial_loss - final_loss) / initial_loss
            ),
            "max_peak_memory_mib": max(
                float(row["peak_memory_mib"]) for row in all_metrics
            ),
            "mean_step_time_ms": sum(
                float(row["step_time_ms"]) for row in all_metrics
            )
            / len(all_metrics),
            "metrics": str(metrics_path),
            "optimizer_parameter_scope": "adapter_only",
            "resumed_from_step": start_step,
            "status": "complete",
            "final_gate_raw": float(adapter.gate.detach().float().cpu()),
            "trainable_parameter_count": adapter.trainable_parameter_count,
            "uses_ground_truth_future_input": False,
            "variant": cfg.variant,
            "wall_s_this_invocation": time.perf_counter() - started,
        }
        atomic_write_json(manifest_path, result)
        atomic_write_json(
            status_path,
            {
                "completed_steps": cfg.training.max_steps,
                "finished_at_unix_s": time.time(),
                "status": "complete",
                "variant": cfg.variant,
            },
        )
        del final_tensor
        return result
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_unix_s": time.time(),
                "status": "failed",
                "variant": cfg.variant,
            },
        )
        raise
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_fixed_subset_probe(
    cfg: Thought3Config,
    model: Any,
    adapter: FutureToActionAdapter,
    injector: ActionEncoderFutureInjector,
    samples: Sequence[RealTrainingSample],
    *,
    device: str,
) -> dict[str, Any]:
    """Evaluate the same fixed noise/timestep objective for each train sample."""

    if len(samples) != 8:
        raise RealTrainingError(
            "Gate E.2 fixed subset probe requires exactly 8 samples"
        )
    was_training = adapter.training
    adapter.eval()
    per_sample: list[dict[str, Any]] = []
    try:
        for sample in samples:
            loss = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=0,
                device=device,
            )
            diagnostics = adapter.last_diagnostics
            if diagnostics is None or diagnostics.action_hidden_norm <= 0:
                raise RealTrainingError(
                    "Gate E.2 fixed subset diagnostics are invalid"
                )
            value = float(loss.detach().cpu())
            ratio = (
                diagnostics.gated_delta_norm
                / diagnostics.action_hidden_norm
            )
            if not math.isfinite(value) or not math.isfinite(ratio):
                raise RealTrainingError(
                    "Gate E.2 fixed subset probe is non-finite"
                )
            per_sample.append(
                {
                    "action_hidden_norm": (
                        diagnostics.action_hidden_norm
                    ),
                    "action_loss": value,
                    "attention_residual_norm": (
                        diagnostics.attention_residual_norm
                    ),
                    "base_sample_id": sample.base_sample_id,
                    "gated_delta_nonzero_fraction": (
                        diagnostics.gated_delta_nonzero_fraction
                    ),
                    "gated_delta_norm": (
                        diagnostics.gated_delta_norm
                    ),
                    "gated_delta_to_action_hidden_ratio": ratio,
                }
            )
    finally:
        adapter.train(was_training)
    losses = [float(row["action_loss"]) for row in per_sample]
    ratios = [
        float(row["gated_delta_to_action_hidden_ratio"])
        for row in per_sample
    ]
    return {
        "fixed_action_flow_step": 0,
        "gate_raw": float(adapter.gate.detach().float().cpu()),
        "max_gated_delta_to_action_hidden_ratio": max(ratios),
        "mean_action_loss": statistics.fmean(losses),
        "median_gated_delta_to_action_hidden_ratio": (
            statistics.median(ratios)
        ),
        "per_sample": per_sample,
        "sample_count": len(per_sample),
        "sample_ids": [
            str(row["base_sample_id"]) for row in per_sample
        ],
        "uses_ground_truth_future_input": False,
        "variant": cfg.variant,
    }


def aggregate_multiflow_probe_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    flow_steps: Sequence[int],
    variant: str,
) -> dict[str, Any]:
    """Aggregate a complete sample × held-out-flow objective grid."""

    normalized_ids = tuple(str(value) for value in sample_ids)
    normalized_steps = tuple(int(value) for value in flow_steps)
    if (
        len(normalized_ids) != 8
        or len(set(normalized_ids)) != 8
        or normalized_steps != (1, 2, 3, 4, 5)
        or variant not in {"A0", "A1"}
    ):
        raise RealTrainingError(
            "Gate E.3 requires 8 unique samples and held-out "
            "flow steps 1..5"
        )
    expected_pairs = {
        (base_sample_id, flow_step)
        for base_sample_id in normalized_ids
        for flow_step in normalized_steps
    }
    keyed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["base_sample_id"]),
            int(row["flow_step"]),
        )
        if key in keyed:
            raise RealTrainingError(
                f"duplicate Gate E.3 multiflow objective: {key}"
            )
        values = (
            float(row["action_weight"]),
            float(row["action_loss"]),
            float(row["action_hidden_norm"]),
            float(row["attention_residual_norm"]),
            float(row["gated_delta_nonzero_fraction"]),
            float(row["gated_delta_norm"]),
            float(row["gated_delta_to_action_hidden_ratio"]),
            float(row["latency_ms"]),
            float(row["peak_memory_mib"]),
            float(row["timestep"]),
        )
        if (
            any(not math.isfinite(value) for value in values)
            or values[0] < 0
            or values[1] < 0
            or values[2] <= 0
            or values[3] < 0
            or values[4] < 0
            or values[4] > 1
            or values[5] < 0
            or values[6] < 0
            or values[7] < 0
            or values[8] < 0
            or values[9] < 0
            or (values[0] == 0 and values[1] != 0)
        ):
            raise RealTrainingError(
                f"non-finite/invalid Gate E.3 multiflow row: {key}"
            )
        keyed[key] = row
    if set(keyed) != expected_pairs:
        missing = sorted(expected_pairs - set(keyed))
        extra = sorted(set(keyed) - expected_pairs)
        raise RealTrainingError(
            "Gate E.3 multiflow objective grid mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    ordered_rows = [
        dict(keyed[(base_sample_id, flow_step)])
        for base_sample_id in normalized_ids
        for flow_step in normalized_steps
    ]
    per_sample: list[dict[str, Any]] = []
    for base_sample_id in normalized_ids:
        sample_rows = [
            keyed[(base_sample_id, flow_step)]
            for flow_step in normalized_steps
        ]

        def mean(field: str) -> float:
            return statistics.fmean(
                float(row[field]) for row in sample_rows
            )

        per_sample.append(
            {
                "action_hidden_norm": mean("action_hidden_norm"),
                "action_loss": mean("action_loss"),
                "action_weight": mean("action_weight"),
                "attention_residual_norm": mean(
                    "attention_residual_norm"
                ),
                "base_sample_id": base_sample_id,
                "flow_objective_count": len(normalized_steps),
                "flow_steps": list(normalized_steps),
                "gated_delta_nonzero_fraction": mean(
                    "gated_delta_nonzero_fraction"
                ),
                "gated_delta_norm": mean("gated_delta_norm"),
                "gated_delta_to_action_hidden_ratio": mean(
                    "gated_delta_to_action_hidden_ratio"
                ),
                "max_objective_gated_delta_to_action_hidden_ratio": max(
                    float(
                        row[
                            "gated_delta_to_action_hidden_ratio"
                        ]
                    )
                    for row in sample_rows
                ),
                "zero_action_loss_objective_count": sum(
                    float(row["action_loss"]) == 0
                    for row in sample_rows
                ),
                "zero_weight_objective_count": sum(
                    float(row["action_weight"]) == 0
                    for row in sample_rows
                ),
            }
        )
    sample_losses = [
        float(row["action_loss"]) for row in per_sample
    ]
    sample_ratios = [
        float(row["gated_delta_to_action_hidden_ratio"])
        for row in per_sample
    ]
    objective_ratios = [
        float(row["gated_delta_to_action_hidden_ratio"])
        for row in ordered_rows
    ]
    return {
        "flow_objective_count": len(ordered_rows),
        "flow_steps": list(normalized_steps),
        "max_gated_delta_to_action_hidden_ratio": max(sample_ratios),
        "max_objective_gated_delta_to_action_hidden_ratio": max(
            objective_ratios
        ),
        "mean_action_loss": statistics.fmean(sample_losses),
        "median_gated_delta_to_action_hidden_ratio": (
            statistics.median(sample_ratios)
        ),
        "per_objective": ordered_rows,
        "per_sample": per_sample,
        "sample_count": len(per_sample),
        "sample_ids": list(normalized_ids),
        "uses_ground_truth_future_input": False,
        "variant": variant,
        "zero_action_loss_objective_count": sum(
            float(row["action_loss"]) == 0
            for row in ordered_rows
        ),
        "zero_weight_objective_count": sum(
            float(row["action_weight"]) == 0
            for row in ordered_rows
        ),
    }


@torch.no_grad()
def evaluate_multiflow_subset_probe(
    cfg: Thought3Config,
    model: Any,
    adapter: FutureToActionAdapter,
    injector: ActionEncoderFutureInjector,
    samples: Sequence[RealTrainingSample],
    *,
    flow_steps: Sequence[int],
    device: str,
) -> dict[str, Any]:
    """Evaluate held-out deterministic action-flow draws without training."""

    sample_ids = [sample.base_sample_id for sample in samples]
    normalized_steps = tuple(int(value) for value in flow_steps)
    if (
        len(samples) != 8
        or len(set(sample_ids)) != 8
        or normalized_steps != (1, 2, 3, 4, 5)
    ):
        raise RealTrainingError(
            "Gate E.3 multiflow probe requires 8 samples and steps 1..5"
        )
    was_training = adapter.training
    adapter.eval()
    objective_rows: list[dict[str, Any]] = []
    try:
        for sample in samples:
            for flow_step in normalized_steps:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                loss = _loss_for_real_sample(
                    cfg,
                    model,
                    adapter,
                    injector,
                    sample,
                    step=flow_step,
                    device=device,
                )
                torch.cuda.synchronize(device)
                latency_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                diagnostics = adapter.last_diagnostics
                if (
                    diagnostics is None
                    or diagnostics.action_hidden_norm <= 0
                ):
                    raise RealTrainingError(
                        "Gate E.3 multiflow diagnostics are invalid"
                    )
                loss_value = float(loss.detach().float().cpu())
                ratio = (
                    diagnostics.gated_delta_norm
                    / diagnostics.action_hidden_norm
                )
                (
                    timestep_value,
                    action_weight_value,
                ) = _flow_timestep_and_weight_scalars(
                    model,
                    sample,
                    train_seed=cfg.training.train_seed,
                    step=flow_step,
                    device=device,
                )
                objective_rows.append(
                    {
                        "action_hidden_norm": (
                            diagnostics.action_hidden_norm
                        ),
                        "action_loss": loss_value,
                        "action_weight": action_weight_value,
                        "attention_residual_norm": (
                            diagnostics.attention_residual_norm
                        ),
                        "base_sample_id": sample.base_sample_id,
                        "flow_step": flow_step,
                        "gated_delta_nonzero_fraction": (
                            diagnostics.gated_delta_nonzero_fraction
                        ),
                        "gated_delta_norm": (
                            diagnostics.gated_delta_norm
                        ),
                        "gated_delta_to_action_hidden_ratio": ratio,
                        "latency_ms": latency_ms,
                        "peak_memory_mib": (
                            int(
                                torch.cuda.max_memory_allocated(
                                    device
                                )
                            )
                            / 2**20
                        ),
                        "timestep": timestep_value,
                    }
                )
                del loss
    finally:
        adapter.train(was_training)
    result = aggregate_multiflow_probe_rows(
        objective_rows,
        sample_ids=sample_ids,
        flow_steps=normalized_steps,
        variant=cfg.variant,
    )
    result.update(
        {
            "gate_raw": float(
                adapter.gate.detach().float().cpu()
            ),
            "max_objective_peak_memory_mib": max(
                float(row["peak_memory_mib"])
                for row in objective_rows
            ),
            "mean_objective_latency_ms": statistics.fmean(
                float(row["latency_ms"])
                for row in objective_rows
            ),
        }
    )
    return result


def multiflow_subset_outcome(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two complete Gate E.3 held-out objective grids."""

    if (
        list(initial_probe.get("flow_steps", []))
        != [1, 2, 3, 4, 5]
        or list(final_probe.get("flow_steps", []))
        != [1, 2, 3, 4, 5]
        or int(initial_probe.get("flow_objective_count", -1)) != 40
        or int(final_probe.get("flow_objective_count", -1)) != 40
    ):
        raise RealTrainingError(
            "Gate E.3 probe does not cover the frozen held-out grid"
        )
    initial_objectives = {
        (
            str(row["base_sample_id"]),
            int(row["flow_step"]),
        ): row
        for row in initial_probe["per_objective"]
    }
    final_objectives = {
        (
            str(row["base_sample_id"]),
            int(row["flow_step"]),
        ): row
        for row in final_probe["per_objective"]
    }
    if (
        len(initial_objectives) != 40
        or set(initial_objectives) != set(final_objectives)
    ):
        raise RealTrainingError(
            "Gate E.3 initial/final objective identities differ"
        )
    for probe, label in (
        (initial_probe, "initial"),
        (final_probe, "final"),
    ):
        recomputed = aggregate_multiflow_probe_rows(
            probe["per_objective"],
            sample_ids=probe["sample_ids"],
            flow_steps=probe["flow_steps"],
            variant=str(probe["variant"]),
        )
        checked_fields = (
            "flow_objective_count",
            "flow_steps",
            "max_gated_delta_to_action_hidden_ratio",
            "max_objective_gated_delta_to_action_hidden_ratio",
            "mean_action_loss",
            "median_gated_delta_to_action_hidden_ratio",
            "per_sample",
            "sample_count",
            "sample_ids",
            "uses_ground_truth_future_input",
            "variant",
            "zero_action_loss_objective_count",
            "zero_weight_objective_count",
        )
        if any(
            probe[field] != recomputed[field]
            for field in checked_fields
        ):
            raise RealTrainingError(
                f"Gate E.3 {label} summary differs from objective rows"
            )
    outcome = fixed_subset_outcome(initial_probe, final_probe)
    objective_loss_ratios: list[float] = []
    zero_initial_loss_objective_count = 0
    zero_initial_loss_with_positive_weight_count = 0
    positive_final_from_zero_initial_loss_count = 0
    max_final_loss_from_zero_initial_loss = 0.0
    for key, initial_row in initial_objectives.items():
        initial_loss = float(initial_row["action_loss"])
        final_loss = float(final_objectives[key]["action_loss"])
        initial_weight = float(initial_row["action_weight"])
        final_weight = float(final_objectives[key]["action_weight"])
        if (
            float(initial_row["timestep"])
            != float(final_objectives[key]["timestep"])
            or initial_weight != final_weight
        ):
            raise RealTrainingError(
                "Gate E.3 initial/final objective flow inputs differ"
            )
        if initial_loss == 0:
            zero_initial_loss_objective_count += 1
            if initial_weight > 0:
                zero_initial_loss_with_positive_weight_count += 1
            if final_loss > 0:
                positive_final_from_zero_initial_loss_count += 1
                max_final_loss_from_zero_initial_loss = max(
                    max_final_loss_from_zero_initial_loss,
                    final_loss,
                )
            continue
        objective_loss_ratios.append(final_loss / initial_loss)
    if not objective_loss_ratios:
        raise RealTrainingError(
            "Gate E.3 has no positive initial objective loss"
        )
    return {
        **outcome,
        "flow_objective_count": 40,
        "flow_steps": [1, 2, 3, 4, 5],
        "max_objective_gated_delta_to_action_hidden_ratio": float(
            final_probe[
                "max_objective_gated_delta_to_action_hidden_ratio"
            ]
        ),
        "max_objective_loss_ratio": max(objective_loss_ratios),
        "max_final_loss_from_zero_initial_loss": (
            max_final_loss_from_zero_initial_loss
        ),
        "objective_loss_ratio_count": len(objective_loss_ratios),
        "positive_final_from_zero_initial_loss_count": (
            positive_final_from_zero_initial_loss_count
        ),
        "zero_initial_loss_objective_count": (
            zero_initial_loss_objective_count
        ),
        "zero_initial_loss_with_positive_weight_count": (
            zero_initial_loss_with_positive_weight_count
        ),
        "zero_weight_objective_count": int(
            initial_probe["zero_weight_objective_count"]
        ),
    }


def fixed_subset_outcome(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two fixed-subset probes without consulting dev/OOD outcomes."""

    initial_rows = {
        str(row["base_sample_id"]): row
        for row in initial_probe["per_sample"]
    }
    final_rows = {
        str(row["base_sample_id"]): row
        for row in final_probe["per_sample"]
    }
    if (
        len(initial_rows) != 8
        or set(initial_rows) != set(final_rows)
        or list(initial_probe["sample_ids"])
        != list(final_probe["sample_ids"])
    ):
        raise RealTrainingError(
            "Gate E.2 initial/final fixed subset identities differ"
        )
    sample_rows: list[dict[str, Any]] = []
    improved_or_equal = 0
    catastrophic = 0
    for base_sample_id in initial_probe["sample_ids"]:
        initial_loss = float(
            initial_rows[str(base_sample_id)]["action_loss"]
        )
        final_loss = float(
            final_rows[str(base_sample_id)]["action_loss"]
        )
        if initial_loss <= 0:
            raise RealTrainingError(
                "Gate E.2 initial fixed action loss must be positive"
            )
        if final_loss <= initial_loss:
            improved_or_equal += 1
        if final_loss > initial_loss * 2.0:
            catastrophic += 1
        sample_rows.append(
            {
                "base_sample_id": str(base_sample_id),
                "final_action_loss": final_loss,
                "initial_action_loss": initial_loss,
                "loss_ratio": final_loss / initial_loss,
                "non_worsened": final_loss <= initial_loss,
            }
        )
    initial_mean = float(initial_probe["mean_action_loss"])
    final_mean = float(final_probe["mean_action_loss"])
    recomputed_initial_mean = statistics.fmean(
        float(row["action_loss"]) for row in initial_rows.values()
    )
    recomputed_final_mean = statistics.fmean(
        float(row["action_loss"]) for row in final_rows.values()
    )
    final_ratios = [
        float(row["gated_delta_to_action_hidden_ratio"])
        for row in final_rows.values()
    ]
    if (
        initial_mean != recomputed_initial_mean
        or final_mean != recomputed_final_mean
        or float(
            final_probe[
                "median_gated_delta_to_action_hidden_ratio"
            ]
        )
        != statistics.median(final_ratios)
        or float(
            final_probe["max_gated_delta_to_action_hidden_ratio"]
        )
        != max(final_ratios)
    ):
        raise RealTrainingError(
            "Gate E.2 fixed subset summary differs from per-sample rows"
        )
    if initial_mean <= 0:
        raise RealTrainingError(
            "Gate E.2 initial mean action loss must be positive"
        )
    return {
        "catastrophic_sample_count": catastrophic,
        "final_mean_action_loss": final_mean,
        "initial_mean_action_loss": initial_mean,
        "loss_reduction_fraction": (
            (initial_mean - final_mean) / initial_mean
        ),
        "max_gated_delta_to_action_hidden_ratio": float(
            final_probe["max_gated_delta_to_action_hidden_ratio"]
        ),
        "median_gated_delta_to_action_hidden_ratio": float(
            final_probe["median_gated_delta_to_action_hidden_ratio"]
        ),
        "non_worsened_sample_count": improved_or_equal,
        "per_sample": sample_rows,
        "sample_count": 8,
    }


def run_fixed_subset_training(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
    device: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train A0/A1 on eight train samples with fixed per-sample objectives.

    This Gate E.2 diagnostic deliberately avoids development and OOD outcomes.
    It is restartable at 50-step checkpoints and records sample-equal fixed
    loss plus the actual BF16 hidden-correction scale.
    """

    if (
        cfg.runtime.backend != "fastwam"
        or cfg.variant not in {"A0", "A1"}
        or cfg.training.max_steps != 200
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
        or cfg.training.checkpoint_interval != 50
        or device != "cuda:0"
        or cfg.runtime.device != device
    ):
        raise RealTrainingError(
            "fixed-subset diagnostic requires real A0/A1, cuda:0, "
            "200 steps, batch 1, checkpoint 50"
        )
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    status_path = output / "run_status.json"
    metrics_path = output / "train_metrics.jsonl"
    probe_path = output / "fixed_subset_metrics.jsonl"
    state_path = output / "training_state.json"
    manifest_path = output / "training_manifest.json"
    checkpoints_root = output / "checkpoints"
    if manifest_path.is_file() and resume:
        existing = load_json(manifest_path)
        if (
            existing.get("status") == "complete"
            and int(existing.get("completed_steps", -1))
            == cfg.training.max_steps
        ):
            return existing
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(
            f"Gate E.2 track output exists: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "learning_rate": cfg.training.learning_rate,
            "started_at_unix_s": time.time(),
            "status": "running",
            "variant": cfg.variant,
        },
    )

    all_train_samples = _ordered_samples(
        (sample for sample in prepared.samples if sample.split == "train"),
        seed=cfg.training.train_seed,
    )
    if (
        len(all_train_samples) != 8
        or len(prepared.samples) != 8
        or any(
            sample.split != "train" for sample in prepared.samples
        )
    ):
        raise RealTrainingError(
            "Gate E.2 requires exactly 8 source-filtered train samples"
        )
    samples = all_train_samples
    sample_ids = [sample.base_sample_id for sample in samples]
    adapter = build_real_adapter(cfg, device=device)
    initial_adapter_sha256 = adapter_state_sha256(
        adapter.state_dict()
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {
        id(parameter) for parameter in adapter.parameters()
    }:
        raise RealTrainingError(
            "Gate E.2 optimizer contains non-Adapter parameters"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RealTrainingError(
            "Gate E.2 frozen Fast-WAM parameter became trainable"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    start_step = 0
    sample_cursor = 0
    latest = find_latest_checkpoint(checkpoints_root) if resume else None
    if latest is not None:
        loaded = load_adapter_checkpoint(
            latest,
            adapter=adapter,
            optimizer=optimizer,
            expected=_checkpoint_expected(
                cfg,
                prepared,
                frozen_parameter_sha256=frozen_parameter_sha256,
            ),
        )
        start_step = loaded.global_step
        sample_cursor = loaded.sample_cursor
        if sample_cursor != start_step:
            raise RealTrainingError(
                "Gate E.2 checkpoint sample cursor differs from step"
            )
    elif (
        resume
        and checkpoints_root.exists()
        and any(checkpoints_root.iterdir())
    ):
        raise RealTrainingError(
            "Gate E.2 resume requested without a valid checkpoint"
        )
    existing_metrics = _metric_rows_for_resume(
        metrics_path,
        start_step=start_step,
    )

    if state_path.is_file():
        state = load_json(state_path)
        if (
            state["config_fingerprint"] != cfg.fingerprint
            or state["frozen_parameter_sha256"]
            != frozen_parameter_sha256
            or state["initial_adapter_sha256"]
            != initial_adapter_sha256
            or list(state["sample_ids"]) != sample_ids
        ):
            raise RealTrainingError(
                "Gate E.2 training-state provenance mismatch"
            )
        initial_probe = dict(state["initial_probe"])
    else:
        if start_step:
            raise RealTrainingError(
                "Gate E.2 checkpoint exists without initial state"
            )
        initial_probe = evaluate_fixed_subset_probe(
            cfg,
            model,
            adapter,
            injector,
            samples,
            device=device,
        )
        if (
            float(
                initial_probe[
                    "max_gated_delta_to_action_hidden_ratio"
                ]
            )
            != 0
            or any(
                float(row["gated_delta_nonzero_fraction"]) != 0
                for row in initial_probe["per_sample"]
            )
        ):
            raise RealTrainingError(
                "Gate E.2 zero-gate initialization is not exact identity"
            )
        state = {
            "cache_fingerprint": prepared.cache_fingerprint,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "fixed_action_flow_step": 0,
            "frozen_parameter_sha256": frozen_parameter_sha256,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_probe": initial_probe,
            "sample_ids": sample_ids,
            "split_fingerprint": prepared.split_fingerprint,
            "uses_ground_truth_future_input": False,
            "variant": cfg.variant,
        }
        atomic_write_json(state_path, state)

    if probe_path.is_file():
        probe_rows = load_jsonl(probe_path)
        probe_steps = [int(row["global_step"]) for row in probe_rows]
        if (
            not probe_rows
            or probe_steps[0] != 0
            or probe_steps != sorted(set(probe_steps))
            or list(probe_rows[0]["sample_ids"]) != sample_ids
            or float(probe_rows[0]["mean_action_loss"])
            != float(initial_probe["mean_action_loss"])
        ):
            raise RealTrainingError(
                "Gate E.2 fixed subset history is invalid"
            )
        probe_rows = [
            row
            for row in probe_rows
            if int(row["global_step"]) <= start_step
        ]
    else:
        if start_step:
            raise RealTrainingError(
                "Gate E.2 checkpoint exists without fixed subset history"
            )
        probe_rows = [
            {
                **initial_probe,
                "global_step": 0,
                "learning_rate": cfg.training.learning_rate,
            }
        ]
        atomic_write_jsonl(probe_path, probe_rows)
    if start_step and all(
        int(row["global_step"]) != start_step for row in probe_rows
    ):
        resumed_probe = evaluate_fixed_subset_probe(
            cfg,
            model,
            adapter,
            injector,
            samples,
            device=device,
        )
        probe_rows.append(
            {
                **resumed_probe,
                "global_step": start_step,
                "learning_rate": cfg.training.learning_rate,
                "outcome_from_initial": fixed_subset_outcome(
                    initial_probe,
                    resumed_probe,
                ),
            }
        )
        atomic_write_jsonl(probe_path, probe_rows)

    new_metrics: list[dict[str, Any]] = []
    first_non_gate_step: int | None = None
    first_projector_step: int | None = None
    first_attention_step: int | None = None
    for row in existing_metrics:
        global_step = int(row["global_step"])
        groups = row["gradient_groups"]
        if (
            first_non_gate_step is None
            and int(
                groups["non_gate"]["nonzero_element_count"]
            )
            > 0
        ):
            first_non_gate_step = global_step
        if (
            first_projector_step is None
            and int(
                groups["future_projector"]["nonzero_element_count"]
            )
            > 0
        ):
            first_projector_step = global_step
        if (
            first_attention_step is None
            and int(
                groups["attention"]["nonzero_element_count"]
            )
            > 0
        ):
            first_attention_step = global_step
    started = time.perf_counter()
    try:
        for step in range(start_step, cfg.training.max_steps):
            sample = samples[sample_cursor % len(samples)]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            gate_before = float(adapter.gate.detach().float().cpu())
            loss = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=0,
                device=device,
            )
            if not bool(torch.isfinite(loss).item()):
                raise RealTrainingError(
                    "Gate E.2 action loss is NaN/Inf"
                )
            loss.backward()
            groups = adapter_gradient_groups(adapter)
            if not all(bool(value["finite"]) for value in groups.values()):
                raise RealTrainingError(
                    "Gate E.2 Adapter gradient is non-finite"
                )
            global_step = step + 1
            if global_step == 1 and (
                float(groups["gate"]["l2"]) <= 0
                or int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                != 0
            ):
                raise RealTrainingError(
                    "Gate E.2 first-step zero-gate contract failed"
                )
            if (
                first_non_gate_step is None
                and int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                > 0
            ):
                first_non_gate_step = global_step
            if (
                first_projector_step is None
                and int(
                    groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                > 0
            ):
                first_projector_step = global_step
            if (
                first_attention_step is None
                and int(
                    groups["attention"]["nonzero_element_count"]
                )
                > 0
            ):
                first_attention_step = global_step
            backbone_grads = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if backbone_grads:
                raise RealTrainingError(
                    "Gate E.2 frozen Fast-WAM received gradients: "
                    f"{backbone_grads[:5]}"
                )
            diagnostics = adapter.last_diagnostics
            if diagnostics is None or diagnostics.action_hidden_norm <= 0:
                raise RealTrainingError(
                    "Gate E.2 Adapter diagnostics are missing"
                )
            gate_gradient = float(
                adapter.gate.grad.detach().float().cpu()
            )
            optimizer.step()
            sample_cursor += 1
            torch.cuda.synchronize(device)
            gate_after = float(adapter.gate.detach().float().cpu())
            row = {
                "action_hidden_norm": diagnostics.action_hidden_norm,
                "attention_residual_norm": (
                    diagnostics.attention_residual_norm
                ),
                "base_sample_id": sample.base_sample_id,
                "fixed_action_flow_step": 0,
                "future_token_norm": diagnostics.future_token_norm,
                "gate_gradient": gate_gradient,
                "gate_gradient_sign": (
                    1
                    if gate_gradient > 0
                    else -1
                    if gate_gradient < 0
                    else 0
                ),
                "gate_raw_after_step": gate_after,
                "gate_raw_before_step": gate_before,
                "gated_delta_nonzero_fraction": (
                    diagnostics.gated_delta_nonzero_fraction
                ),
                "gated_delta_norm": diagnostics.gated_delta_norm,
                "gated_delta_to_action_hidden_ratio": (
                    diagnostics.gated_delta_norm
                    / diagnostics.action_hidden_norm
                ),
                "global_step": global_step,
                "gradient_groups": groups,
                "learning_rate": cfg.training.learning_rate,
                "loss": float(loss.detach().cpu()),
                "nan_or_inf": False,
                "peak_memory_mib": (
                    int(torch.cuda.max_memory_allocated(device)) / 2**20
                ),
                "sample_cursor": sample_cursor,
                "step_time_ms": (
                    time.perf_counter() - step_started
                )
                * 1000.0,
                "variant": cfg.variant,
            }
            new_metrics.append(row)
            should_checkpoint = (
                global_step % cfg.training.checkpoint_interval == 0
                or global_step == cfg.training.max_steps
            )
            if should_checkpoint:
                fixed_probe = evaluate_fixed_subset_probe(
                    cfg,
                    model,
                    adapter,
                    injector,
                    samples,
                    device=device,
                )
                outcome = fixed_subset_outcome(
                    initial_probe,
                    fixed_probe,
                )
                probe_rows.append(
                    {
                        **fixed_probe,
                        "global_step": global_step,
                        "learning_rate": cfg.training.learning_rate,
                        "outcome_from_initial": outcome,
                    }
                )
                atomic_write_jsonl(
                    metrics_path,
                    [*existing_metrics, *new_metrics],
                )
                atomic_write_jsonl(probe_path, probe_rows)
                checkpoint = (
                    checkpoints_root / f"step_{global_step:08d}"
                )
                save_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    optimizer=optimizer,
                    manifest=_checkpoint_manifest(
                        cfg,
                        adapter,
                        split_fingerprint=prepared.split_fingerprint,
                        cache_fingerprint=prepared.cache_fingerprint,
                        frozen_parameter_sha256=(
                            frozen_parameter_sha256
                        ),
                        global_step=global_step,
                        sample_cursor=sample_cursor,
                        train_sample_count=len(samples),
                        extra={
                            "fixed_action_flow_step": 0,
                            "gate_e2_eight_sample": True,
                            "subset_sample_count": len(samples),
                        },
                    ),
                )
                if progress is not None:
                    progress(
                        "fixed_subset_checkpoint",
                        {
                            "loss_reduction_fraction": outcome[
                                "loss_reduction_fraction"
                            ],
                            "max_delta_hidden_ratio": outcome[
                                "max_gated_delta_to_action_hidden_ratio"
                            ],
                            "learning_rate": (
                                cfg.training.learning_rate
                            ),
                            "mean_action_loss": fixed_probe[
                                "mean_action_loss"
                            ],
                            "step": global_step,
                            "variant": cfg.variant,
                        },
                    )
            del loss
            torch.cuda.empty_cache()

        atomic_write_jsonl(
            metrics_path,
            [*existing_metrics, *new_metrics],
        )
        final_probe_rows = [
            row
            for row in probe_rows
            if int(row["global_step"]) == cfg.training.max_steps
        ]
        if len(final_probe_rows) != 1:
            raise RealTrainingError(
                "Gate E.2 final fixed subset probe is missing/duplicated"
            )
        final_probe = final_probe_rows[0]
        final_outcome = fixed_subset_outcome(
            initial_probe,
            final_probe,
        )
        latest_checkpoint = find_latest_checkpoint(checkpoints_root)
        if latest_checkpoint is None:
            raise RealTrainingError(
                "Gate E.2 wrote no checkpoint"
            )
        roundtrip = _checkpoint_roundtrip(
            cfg,
            adapter,
            optimizer,
            latest_checkpoint,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            device=device,
        )
        all_metrics = [*existing_metrics, *new_metrics]
        result = {
            "adapter_fingerprint": cfg.adapter_structural_fingerprint,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_roundtrip": roundtrip,
            "completed_steps": cfg.training.max_steps,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "device": device,
            "final_gate_raw": float(
                adapter.gate.detach().float().cpu()
            ),
            "final_probe": final_probe,
            "first_attention_nonzero_gradient_step": (
                first_attention_step
            ),
            "first_non_gate_nonzero_gradient_step": (
                first_non_gate_step
            ),
            "first_projector_nonzero_gradient_step": (
                first_projector_step
            ),
            "fixed_action_flow_step": 0,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_probe": initial_probe,
            "learning_rate": cfg.training.learning_rate,
            "max_peak_memory_mib": max(
                float(row["peak_memory_mib"]) for row in all_metrics
            ),
            "mean_step_time_ms": statistics.fmean(
                float(row["step_time_ms"]) for row in all_metrics
            ),
            "metrics": str(metrics_path),
            "optimizer_parameter_scope": "adapter_only",
            "outcome": final_outcome,
            "probe_metrics": str(probe_path),
            "resumed_from_step": start_step,
            "sample_count": len(samples),
            "sample_ids": sample_ids,
            "status": "complete",
            "trainable_parameter_count": (
                adapter.trainable_parameter_count
            ),
            "uses_development_outcomes": False,
            "uses_ground_truth_future_input": False,
            "uses_ood_or_success_outcomes": False,
            "variant": cfg.variant,
            "wall_s_this_invocation": time.perf_counter() - started,
        }
        atomic_write_json(manifest_path, result)
        atomic_write_json(
            status_path,
            {
                "completed_steps": cfg.training.max_steps,
                "finished_at_unix_s": time.time(),
                "learning_rate": cfg.training.learning_rate,
                "status": "complete",
                "variant": cfg.variant,
            },
        )
        return result
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_unix_s": time.time(),
                "learning_rate": cfg.training.learning_rate,
                "status": "failed",
                "variant": cfg.variant,
            },
        )
        raise
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()


def diversified_training_flow_slot(global_step: int) -> int:
    """Map Gate E.4 optimizer steps to unique slots outside probes 0..5."""

    if global_step < 1 or global_step > 200:
        raise RealTrainingError(
            "Gate E.4 global step must be in the frozen range 1..200"
        )
    return DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET + global_step


def diversified_flow_schedule_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the ordered sample/noise/timestep schedule used by Gate E.4."""

    payload: list[str] = []
    for expected_step, row in enumerate(rows, start=1):
        global_step = int(row["global_step"])
        flow_slot = int(row["training_flow_slot"])
        if (
            global_step != expected_step
            or flow_slot
            != diversified_training_flow_slot(global_step)
        ):
            raise RealTrainingError(
                "Gate E.4 diversified flow schedule is not contiguous"
            )
        payload.append(
            "\0".join(
                (
                    str(global_step),
                    str(row["base_sample_id"]),
                    str(flow_slot),
                    str(int(row["action_noise_seed"])),
                    str(int(row["action_timestep_seed"])),
                    str(row["flow_objective_sha256"]),
                    repr(float(row["timestep"])),
                    repr(float(row["action_weight"])),
                )
            )
        )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def run_diversified_flow_training(
    cfg: Thought3Config,
    *,
    model: Any,
    prepared: PreparedRealTrainingData,
    frozen_parameter_sha256: str,
    resume: bool,
    device: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train Gate E.4 with one deterministic, unique flow slot per visit."""

    if (
        cfg.runtime.backend != "fastwam"
        or cfg.variant not in {"A0", "A1"}
        or cfg.training.max_steps != 200
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
        or cfg.training.checkpoint_interval != 50
        or device != "cuda:0"
        or cfg.runtime.device != device
    ):
        raise RealTrainingError(
            "Gate E.4 requires real A0/A1, cuda:0, 200 steps, "
            "batch 1, checkpoint 50"
        )
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    status_path = output / "run_status.json"
    metrics_path = output / "train_metrics.jsonl"
    probe_path = output / "heldout_multiflow_metrics.jsonl"
    state_path = output / "training_state.json"
    manifest_path = output / "training_manifest.json"
    checkpoints_root = output / "checkpoints"
    if manifest_path.is_file() and resume:
        existing = load_json(manifest_path)
        if (
            existing.get("status") == "complete"
            and int(existing.get("completed_steps", -1)) == 200
        ):
            return existing
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(
            f"Gate E.4 track output exists: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "learning_rate": cfg.training.learning_rate,
            "started_at_unix_s": time.time(),
            "status": "running",
            "variant": cfg.variant,
        },
    )

    samples = _ordered_samples(
        (sample for sample in prepared.samples if sample.split == "train"),
        seed=cfg.training.train_seed,
    )
    if (
        len(samples) != 8
        or len(prepared.samples) != 8
        or any(sample.split != "train" for sample in prepared.samples)
    ):
        raise RealTrainingError(
            "Gate E.4 requires exactly 8 source-filtered train samples"
        )
    sample_ids = [sample.base_sample_id for sample in samples]
    adapter = build_real_adapter(cfg, device=device)
    initial_adapter_sha256 = adapter_state_sha256(
        adapter.state_dict()
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {
        id(parameter) for parameter in adapter.parameters()
    }:
        raise RealTrainingError(
            "Gate E.4 optimizer contains non-Adapter parameters"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RealTrainingError(
            "Gate E.4 frozen Fast-WAM parameter became trainable"
        )
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    start_step = 0
    sample_cursor = 0
    latest = find_latest_checkpoint(checkpoints_root) if resume else None
    if latest is not None:
        loaded = load_adapter_checkpoint(
            latest,
            adapter=adapter,
            optimizer=optimizer,
            expected=_checkpoint_expected(
                cfg,
                prepared,
                frozen_parameter_sha256=frozen_parameter_sha256,
            ),
        )
        start_step = loaded.global_step
        sample_cursor = loaded.sample_cursor
        if sample_cursor != start_step:
            raise RealTrainingError(
                "Gate E.4 checkpoint sample cursor differs from step"
            )
    elif (
        resume
        and checkpoints_root.exists()
        and any(checkpoints_root.iterdir())
    ):
        raise RealTrainingError(
            "Gate E.4 resume requested without a valid checkpoint"
        )
    existing_metrics = _metric_rows_for_resume(
        metrics_path,
        start_step=start_step,
    )

    if state_path.is_file():
        state = load_json(state_path)
        if (
            state["config_fingerprint"] != cfg.fingerprint
            or state["frozen_parameter_sha256"]
            != frozen_parameter_sha256
            or state["initial_adapter_sha256"]
            != initial_adapter_sha256
            or list(state["sample_ids"]) != sample_ids
            or int(state["training_flow_slot_offset"])
            != DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET
            or list(state["heldout_flow_steps"])
            != list(DIVERSIFIED_HELDOUT_FLOW_STEPS)
        ):
            raise RealTrainingError(
                "Gate E.4 training-state provenance mismatch"
            )
        initial_probe = dict(state["initial_probe"])
    else:
        if start_step:
            raise RealTrainingError(
                "Gate E.4 checkpoint exists without initial state"
            )
        initial_probe = evaluate_multiflow_subset_probe(
            cfg,
            model,
            adapter,
            injector,
            samples,
            flow_steps=DIVERSIFIED_HELDOUT_FLOW_STEPS,
            device=device,
        )
        if (
            float(
                initial_probe[
                    "max_gated_delta_to_action_hidden_ratio"
                ]
            )
            != 0
            or any(
                float(row["gated_delta_nonzero_fraction"]) != 0
                for row in initial_probe["per_objective"]
            )
        ):
            raise RealTrainingError(
                "Gate E.4 zero-gate initialization is not exact identity"
            )
        state = {
            "cache_fingerprint": prepared.cache_fingerprint,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "frozen_parameter_sha256": frozen_parameter_sha256,
            "heldout_flow_steps": list(
                DIVERSIFIED_HELDOUT_FLOW_STEPS
            ),
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_probe": initial_probe,
            "sample_ids": sample_ids,
            "split_fingerprint": prepared.split_fingerprint,
            "training_flow_slot_offset": (
                DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET
            ),
            "uses_ground_truth_future_input": False,
            "variant": cfg.variant,
        }
        atomic_write_json(state_path, state)

    if probe_path.is_file():
        probe_rows = load_jsonl(probe_path)
        if (
            not probe_rows
            or int(probe_rows[0]["global_step"]) != 0
            or list(probe_rows[0]["sample_ids"]) != sample_ids
            or float(probe_rows[0]["mean_action_loss"])
            != float(initial_probe["mean_action_loss"])
            or [int(row["global_step"]) for row in probe_rows]
            not in ([0], [0, 200])
        ):
            raise RealTrainingError(
                "Gate E.4 held-out probe history is invalid"
            )
        if start_step < 200 and len(probe_rows) != 1:
            raise RealTrainingError(
                "Gate E.4 final probe precedes final checkpoint"
            )
    else:
        if start_step:
            raise RealTrainingError(
                "Gate E.4 checkpoint exists without initial probe"
            )
        probe_rows = [
            {
                **initial_probe,
                "global_step": 0,
                "learning_rate": cfg.training.learning_rate,
            }
        ]
        atomic_write_jsonl(probe_path, probe_rows)

    new_metrics: list[dict[str, Any]] = []
    first_non_gate_step: int | None = None
    first_projector_step: int | None = None
    first_attention_step: int | None = None
    for row in existing_metrics:
        global_step = int(row["global_step"])
        groups = row["gradient_groups"]
        if (
            first_non_gate_step is None
            and int(groups["non_gate"]["nonzero_element_count"]) > 0
        ):
            first_non_gate_step = global_step
        if (
            first_projector_step is None
            and int(
                groups["future_projector"]["nonzero_element_count"]
            )
            > 0
        ):
            first_projector_step = global_step
        if (
            first_attention_step is None
            and int(groups["attention"]["nonzero_element_count"]) > 0
        ):
            first_attention_step = global_step

    started = time.perf_counter()
    try:
        for step in range(start_step, cfg.training.max_steps):
            global_step = step + 1
            sample = samples[sample_cursor % len(samples)]
            flow_slot = diversified_training_flow_slot(global_step)
            identity = _flow_objective_identity(
                base_sample_id=sample.base_sample_id,
                train_seed=cfg.training.train_seed,
                flow_step=flow_slot,
            )
            timestep, action_weight = (
                _flow_timestep_and_weight_scalars(
                    model,
                    sample,
                    train_seed=cfg.training.train_seed,
                    step=flow_slot,
                    device=device,
                )
            )
            if global_step <= 2 and action_weight <= 0:
                raise RealTrainingError(
                    "Gate E.4 first two frozen flow slots must "
                    "have positive official weight"
                )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            gate_before = float(
                adapter.gate.detach().float().cpu()
            )
            loss = _loss_for_real_sample(
                cfg,
                model,
                adapter,
                injector,
                sample,
                step=flow_slot,
                device=device,
            )
            if not bool(torch.isfinite(loss).item()):
                raise RealTrainingError(
                    "Gate E.4 action loss is NaN/Inf"
                )
            loss.backward()
            groups = adapter_gradient_groups(adapter)
            if not all(bool(value["finite"]) for value in groups.values()):
                raise RealTrainingError(
                    "Gate E.4 Adapter gradient is non-finite"
                )
            if global_step == 1 and (
                float(groups["gate"]["l2"]) <= 0
                or int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                != 0
            ):
                raise RealTrainingError(
                    "Gate E.4 first-step zero-gate contract failed"
                )
            if (
                first_non_gate_step is None
                and int(
                    groups["non_gate"]["nonzero_element_count"]
                )
                > 0
            ):
                first_non_gate_step = global_step
            if (
                first_projector_step is None
                and int(
                    groups["future_projector"][
                        "nonzero_element_count"
                    ]
                )
                > 0
            ):
                first_projector_step = global_step
            if (
                first_attention_step is None
                and int(
                    groups["attention"]["nonzero_element_count"]
                )
                > 0
            ):
                first_attention_step = global_step
            backbone_grads = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if backbone_grads:
                raise RealTrainingError(
                    "Gate E.4 frozen Fast-WAM received gradients: "
                    f"{backbone_grads[:5]}"
                )
            diagnostics = adapter.last_diagnostics
            if diagnostics is None or diagnostics.action_hidden_norm <= 0:
                raise RealTrainingError(
                    "Gate E.4 Adapter diagnostics are missing"
                )
            gate_gradient = float(
                adapter.gate.grad.detach().float().cpu()
            )
            optimizer.step()
            sample_cursor += 1
            torch.cuda.synchronize(device)
            gate_after = float(
                adapter.gate.detach().float().cpu()
            )
            row = {
                **identity,
                "action_hidden_norm": diagnostics.action_hidden_norm,
                "action_weight": action_weight,
                "attention_residual_norm": (
                    diagnostics.attention_residual_norm
                ),
                "base_sample_id": sample.base_sample_id,
                "future_token_norm": diagnostics.future_token_norm,
                "gate_gradient": gate_gradient,
                "gate_gradient_sign": (
                    1
                    if gate_gradient > 0
                    else -1
                    if gate_gradient < 0
                    else 0
                ),
                "gate_raw_after_step": gate_after,
                "gate_raw_before_step": gate_before,
                "gated_delta_nonzero_fraction": (
                    diagnostics.gated_delta_nonzero_fraction
                ),
                "gated_delta_norm": diagnostics.gated_delta_norm,
                "gated_delta_to_action_hidden_ratio": (
                    diagnostics.gated_delta_norm
                    / diagnostics.action_hidden_norm
                ),
                "global_step": global_step,
                "gradient_groups": groups,
                "learning_rate": cfg.training.learning_rate,
                "loss": float(loss.detach().float().cpu()),
                "nan_or_inf": False,
                "peak_memory_mib": (
                    int(torch.cuda.max_memory_allocated(device))
                    / 2**20
                ),
                "sample_cursor": sample_cursor,
                "step_time_ms": (
                    time.perf_counter() - step_started
                )
                * 1000.0,
                "timestep": timestep,
                "training_flow_slot": flow_slot,
                "variant": cfg.variant,
                "zero_weight_objective": action_weight == 0,
            }
            if action_weight == 0 and float(row["loss"]) != 0:
                raise RealTrainingError(
                    "Gate E.4 zero-weight objective has nonzero loss"
                )
            new_metrics.append(row)
            should_checkpoint = (
                global_step % cfg.training.checkpoint_interval == 0
                or global_step == cfg.training.max_steps
            )
            if should_checkpoint:
                committed_metrics = [
                    *existing_metrics,
                    *new_metrics,
                ]
                atomic_write_jsonl(metrics_path, committed_metrics)
                checkpoint = (
                    checkpoints_root / f"step_{global_step:08d}"
                )
                save_adapter_checkpoint(
                    checkpoint,
                    adapter=adapter,
                    optimizer=optimizer,
                    manifest=_checkpoint_manifest(
                        cfg,
                        adapter,
                        split_fingerprint=prepared.split_fingerprint,
                        cache_fingerprint=prepared.cache_fingerprint,
                        frozen_parameter_sha256=(
                            frozen_parameter_sha256
                        ),
                        global_step=global_step,
                        sample_cursor=sample_cursor,
                        train_sample_count=len(samples),
                        extra={
                            "gate_e4_diversified_flow": True,
                            "heldout_flow_steps": list(
                                DIVERSIFIED_HELDOUT_FLOW_STEPS
                            ),
                            "subset_sample_count": len(samples),
                            "train_flow_schedule_sha256": (
                                diversified_flow_schedule_sha256(
                                    committed_metrics
                                )
                            ),
                            "training_flow_slot_offset": (
                                DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET
                            ),
                        },
                    ),
                )
                if progress is not None:
                    progress(
                        "diversified_flow_checkpoint",
                        {
                            "learning_rate": (
                                cfg.training.learning_rate
                            ),
                            "step": global_step,
                            "training_flow_slot": flow_slot,
                            "variant": cfg.variant,
                            "zero_weight_steps": sum(
                                bool(value["zero_weight_objective"])
                                for value in committed_metrics
                            ),
                        },
                    )
            del loss
            torch.cuda.empty_cache()

        atomic_write_jsonl(
            metrics_path,
            [*existing_metrics, *new_metrics],
        )
        all_metrics = [*existing_metrics, *new_metrics]
        if len(all_metrics) != 200:
            raise RealTrainingError(
                "Gate E.4 did not commit exactly 200 metrics"
            )
        final_probe_rows = [
            row
            for row in probe_rows
            if int(row["global_step"]) == 200
        ]
        if final_probe_rows:
            if len(final_probe_rows) != 1:
                raise RealTrainingError(
                    "Gate E.4 final probe is duplicated"
                )
            final_probe = final_probe_rows[0]
        else:
            final_probe = evaluate_multiflow_subset_probe(
                cfg,
                model,
                adapter,
                injector,
                samples,
                flow_steps=DIVERSIFIED_HELDOUT_FLOW_STEPS,
                device=device,
            )
            final_probe = {
                **final_probe,
                "global_step": 200,
                "learning_rate": cfg.training.learning_rate,
            }
            probe_rows.append(final_probe)
            atomic_write_jsonl(probe_path, probe_rows)
        final_outcome = multiflow_subset_outcome(
            initial_probe,
            final_probe,
        )
        latest_checkpoint = find_latest_checkpoint(checkpoints_root)
        if latest_checkpoint is None:
            raise RealTrainingError(
                "Gate E.4 wrote no checkpoint"
            )
        roundtrip = _checkpoint_roundtrip(
            cfg,
            adapter,
            optimizer,
            latest_checkpoint,
            prepared=prepared,
            frozen_parameter_sha256=frozen_parameter_sha256,
            device=device,
        )
        schedule_sha256 = diversified_flow_schedule_sha256(
            all_metrics
        )
        result = {
            "adapter_fingerprint": cfg.adapter_structural_fingerprint,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_roundtrip": roundtrip,
            "completed_steps": 200,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "device": device,
            "final_gate_raw": float(
                adapter.gate.detach().float().cpu()
            ),
            "final_probe": final_probe,
            "first_attention_nonzero_gradient_step": (
                first_attention_step
            ),
            "first_non_gate_nonzero_gradient_step": (
                first_non_gate_step
            ),
            "first_projector_nonzero_gradient_step": (
                first_projector_step
            ),
            "heldout_flow_steps": list(
                DIVERSIFIED_HELDOUT_FLOW_STEPS
            ),
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_probe": initial_probe,
            "learning_rate": cfg.training.learning_rate,
            "max_peak_memory_mib": max(
                float(row["peak_memory_mib"])
                for row in all_metrics
            ),
            "mean_step_time_ms": statistics.fmean(
                float(row["step_time_ms"]) for row in all_metrics
            ),
            "metrics": str(metrics_path),
            "optimizer_parameter_scope": "adapter_only",
            "outcome": final_outcome,
            "probe_metrics": str(probe_path),
            "resumed_from_step": start_step,
            "sample_count": len(samples),
            "sample_ids": sample_ids,
            "status": "complete",
            "train_flow_schedule_sha256": schedule_sha256,
            "train_flow_slot_end": diversified_training_flow_slot(200),
            "train_flow_slot_start": diversified_training_flow_slot(1),
            "trainable_parameter_count": (
                adapter.trainable_parameter_count
            ),
            "training_flow_slot_offset": (
                DIVERSIFIED_TRAIN_FLOW_SLOT_OFFSET
            ),
            "uses_development_outcomes": False,
            "uses_ground_truth_future_input": False,
            "uses_ood_or_success_outcomes": False,
            "variant": cfg.variant,
            "wall_s_this_invocation": time.perf_counter() - started,
            "zero_weight_step_count": sum(
                bool(row["zero_weight_objective"])
                for row in all_metrics
            ),
        }
        atomic_write_json(manifest_path, result)
        atomic_write_json(
            status_path,
            {
                "completed_steps": 200,
                "finished_at_unix_s": time.time(),
                "learning_rate": cfg.training.learning_rate,
                "status": "complete",
                "variant": cfg.variant,
            },
        )
        return result
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_unix_s": time.time(),
                "learning_rate": cfg.training.learning_rate,
                "status": "failed",
                "variant": cfg.variant,
            },
        )
        raise
    finally:
        injector.close()
        del optimizer, adapter
        torch.cuda.empty_cache()
