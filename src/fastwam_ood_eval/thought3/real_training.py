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


def prepare_real_training_data(
    cfg: Thought3Config,
    *,
    model: Any,
    upstream_cfg: Any,
    device: str,
    progress: ProgressCallback | None = None,
) -> PreparedRealTrainingData:
    """Join all Phase D K=1 identities to current-only LIBERO supervision."""

    started = time.perf_counter()
    cache_report = validate_cache(cfg.cache.root)
    entries, plan = load_cache_plan(cfg.cache.root)
    reference_entries = sorted(
        (entry for entry in entries if entry.k == 1),
        key=lambda entry: entry.identity.base_sample_id,
    )
    if len(reference_entries) != 32:
        raise RealTrainingError(
            "Gate E requires exactly 32 Phase D base samples"
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
    if split_counts != {"train": 28, "development": 4}:
        raise RealTrainingError(
            f"Gate E expected selected split 28/4, got {split_counts}"
        )
    source_telemetry = dict(source.telemetry)
    expected_source = {
        "action_target_chunks_read": 32,
        "action_target_read": True,
        "action_target_rows_read": 1024,
        "actual_future_read": False,
        "current_camera_frames_decoded": 64,
        "future_rgb_frames_decoded": 0,
        "state_rows_read": 32,
    }
    for key, expected in expected_source.items():
        if source_telemetry.get(key) != expected:
            raise RealTrainingError(
                f"current/action source telemetry mismatch for {key}: "
                f"{source_telemetry.get(key)} != {expected}"
            )
    report = {
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
        "split_counts": split_counts,
        "split_fingerprint": plan["split_fingerprint"],
    }
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
        key=lambda sample: hashlib.sha256(
            f"thought3-real-train-order-v1\0{seed}\0"
            f"{sample.base_sample_id}".encode("utf-8")
        ).hexdigest(),
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
    generator = torch.Generator(device="cpu").manual_seed(
        _stable_seed(
            "thought3-real-action-noise-v1",
            train_seed,
            step,
            sample.base_sample_id,
        )
    )
    noise = torch.randn(
        tuple(target_action.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=model.torch_dtype)
    timestep = _sample_training_t_on_cpu(
        model.train_action_scheduler,
        _stable_seed(
            "thought3-real-action-time-v1",
            train_seed,
            step,
            sample.base_sample_id,
        ),
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
        nonzero = 0
        tensor_count = 0
        missing = 0
        finite = True
        for name in names:
            gradient = named[name].grad
            if gradient is None:
                missing += 1
                continue
            value = gradient.detach().float()
            tensor_count += 1
            finite = finite and bool(torch.isfinite(value).all().item())
            squared += float(value.square().sum().cpu())
            nonzero += int(torch.count_nonzero(value).item())
        result[group] = {
            "finite": finite,
            "l2": math.sqrt(squared),
            "missing_tensor_count": missing,
            "nonzero_element_count": nonzero,
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
            step=90_000 + index,
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
) -> AdapterCheckpointManifest:
    trainable_names = tuple(
        f"adapter.{name}"
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    )
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
        extra={
            "action_loss": "official_fastwam_flow_matching_velocity_mse",
            "backend": "fastwam",
            "contains_backbone": False,
            "future_source_kind": "model_sampled_from_current",
            "gate_e_smoke": True,
            "uses_ground_truth_future_input": False,
        },
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
    initial_loss: float,
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
    if float(rows[0]["action_loss"]) != initial_loss:
        raise RealTrainingError(
            "development-loss history disagrees with initial state"
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
        state = {
            "cache_fingerprint": prepared.cache_fingerprint,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint,
            "frozen_parameter_sha256": frozen_parameter_sha256,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_validation_action_loss": initial_validation_loss,
            "split_fingerprint": prepared.split_fingerprint,
            "variant": cfg.variant,
        }
        atomic_write_json(state_path, state)

    validation_rows = _validation_rows_for_resume(
        validation_path,
        start_step=start_step,
        initial_loss=initial_validation_loss,
    )
    if not validation_rows:
        validation_rows = [
            {
                "action_loss": initial_validation_loss,
                "checkpoint": None,
                "global_step": 0,
                "selection_split": "development",
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
                validation_rows.append(
                    {
                        "action_loss": checkpoint_validation_loss,
                        "checkpoint": str(checkpoint),
                        "global_step": global_step,
                        "selection_split": "development",
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
            "final_validation_action_loss": final_validation_loss,
            "first_attention_nonzero_gradient_step": first_attention_step,
            "first_non_gate_nonzero_gradient_step": first_non_gate_step,
            "first_projector_nonzero_gradient_step": first_projector_step,
            "initial_adapter_sha256": initial_adapter_sha256,
            "initial_validation_action_loss": initial_validation_loss,
            "loss_decreased": (
                final_validation_loss < initial_validation_loss
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
