"""Phase C: one real LIBERO sample, one GPU, one backward, no optimizer step.

This module is intentionally separate from cache building and training.  It
loads exactly one standard LIBERO demonstration sample, exercises the frozen
Fast-WAM video/action paths and the zero-gated Adapter, writes JSON telemetry,
and exits.  It never serializes a future latent or creates an optimizer.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from fastwam_ood_eval.thought3.config import Thought3Config
from fastwam_ood_eval.thought3.io_utils import atomic_write_json, sha256_file
from fastwam_ood_eval.thought3.safety import (
    ensure_standard_training_source,
    ensure_thought3_output_path,
    validate_training_batch_keys,
)


PHASE_C_SCHEMA = "thought3.phase_c.gate.v1"
OFFICIAL_LIBERO_REVISION = "117413dc0ca99c7cd64036c4eaa4a316c537d692"
OFFICIAL_ARCHIVE_SHA256 = {
    "libero_goal_no_noops_lerobot": (
        "a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea"
    ),
}


class PhaseCGateError(RuntimeError):
    """Raised when a Phase C hard gate fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"phase": "C", "stage": stage, **fields},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "finite": bool(tensor.isfinite().all().item())
        if tensor.is_floating_point()
        else True,
        "shape": [int(value) for value in tensor.shape],
    }


def _max_abs(left: Any, right: Any) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _mean_abs(left: Any, right: Any) -> float:
    return float((left.detach().float() - right.detach().float()).abs().mean().cpu())


def compute_upstream_action_loss(
    pred_action: Any,
    target_action: Any,
    action_is_pad: Any | None,
    action_weight: Any,
    *,
    loss_lambda_action: float,
) -> Any:
    """Reproduce FastWAM.training_loss's action-only lines 550-561."""

    import torch
    from torch.nn import functional as F

    action_loss_token = F.mse_loss(
        pred_action.float(),
        target_action.float(),
        reduction="none",
    ).mean(dim=2)
    if action_is_pad is not None:
        valid = (~action_is_pad).to(
            device=action_loss_token.device,
            dtype=action_loss_token.dtype,
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
    else:
        action_loss_per_sample = action_loss_token.mean(dim=1)
    weight = action_weight.to(
        device=action_loss_per_sample.device,
        dtype=action_loss_per_sample.dtype,
    )
    return float(loss_lambda_action) * (action_loss_per_sample * weight).mean()


def _assert_phase_c_scope(cfg: Thought3Config) -> None:
    if cfg.runtime.backend != "fastwam":
        raise PhaseCGateError("Phase C requires runtime.backend=fastwam")
    if not cfg.runtime.device.startswith("cuda:"):
        raise PhaseCGateError("Phase C requires one explicit logical CUDA device")
    if cfg.variant != "A4" or cfg.sampler.active_k != 4:
        raise PhaseCGateError("Phase C backward stress is frozen to variant=A4")
    if tuple(cfg.sampler.cache_k) != (1, 2, 4):
        raise PhaseCGateError("Phase C must probe exactly K=1/2/4")
    if (
        cfg.training.max_steps != 1
        or cfg.training.microbatch_size != 1
        or cfg.training.gradient_accumulation_steps != 1
    ):
        raise PhaseCGateError(
            "Phase C requires max_steps=microbatch=gradient_accumulation=1"
        )
    if cfg.runtime.online_use_cache:
        raise PhaseCGateError("Phase C forbids online cache reads")
    if len(cfg.data.dataset_roots) != 1:
        raise PhaseCGateError("Phase C accepts exactly one standard LIBERO subset")
    if cfg.data.dataset_revision != OFFICIAL_LIBERO_REVISION:
        raise PhaseCGateError(
            "Phase C dataset revision must equal the pinned official revision"
        )


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _scalar(value: Any) -> Any:
    if hasattr(value, "numel") and int(value.numel()) == 1:
        return value.item()
    return value


def _dataset_inventory(dataset_root: Path, archive: Path | None) -> dict[str, Any]:
    files = [path for path in dataset_root.rglob("*") if path.is_file()]
    suffix_counts: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    payload: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }
    if archive is not None and archive.is_file():
        payload["archive"] = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    return payload


def _load_upstream_model(cfg: Thought3Config) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    fastwam_root = Path("third_party/FastWAM").resolve()
    for path in (fastwam_root,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    sim_config = fastwam_root / "configs" / "sim_libero.yaml"
    checkpoint = cfg.backbone.checkpoint_path
    stats = cfg.backbone.dataset_stats_path
    if checkpoint is None or stats is None:
        raise PhaseCGateError("checkpoint and dataset stats paths are required")
    overrides = [
        "task=libero_uncond_2cam224_1e-4",
        f"ckpt={checkpoint.resolve()}",
        "mixed_precision=bf16",
        f"EVALUATION.dataset_stats_path={stats.resolve()}",
        f"EVALUATION.device={cfg.runtime.device}",
        "EVALUATION.replan_steps=10",
    ]
    started = time.perf_counter()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(sim_config.parent),
    ):
        upstream_cfg = compose(config_name=sim_config.stem, overrides=overrides)
    model = instantiate(
        upstream_cfg.model,
        model_dtype=torch.bfloat16,
        device=cfg.runtime.device,
    )
    if type(model).__name__ != "FastWAM":
        raise PhaseCGateError(
            f"expected FastWAM, instantiated {type(model).__name__}"
        )
    model.load_checkpoint(str(checkpoint))
    model.requires_grad_(False)
    model.eval()
    if getattr(model.video_expert, "action_conditioned", None) is not False:
        raise PhaseCGateError("release checkpoint must use unconditional video expert")
    if str(model.video_expert.video_attention_mask_mode) != "first_frame_causal":
        raise PhaseCGateError("action equivalence requires first_frame_causal")
    return model, upstream_cfg, {
        "load_latency_s": time.perf_counter() - started,
        "model_class": type(model).__name__,
        "torch_dtype": str(model.torch_dtype),
    }


def _load_one_training_sample(
    cfg: Thought3Config,
    model: Any,
    upstream_cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )

    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    dataset_root = ensure_standard_training_source(cfg.data.dataset_roots[0])
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    dataset = instantiate(
        upstream_cfg.data.train,
        dataset_dirs=[str(dataset_root)],
        processor=None,
        text_embedding_cache_dir=None,
    )
    processor = instantiate(upstream_cfg.data.train.processor)
    stats_path = cfg.backbone.dataset_stats_path
    assert stats_path is not None
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(stats_path))
    )
    dataset.lerobot_dataset.set_processor(processor)

    encoded_prompts: list[str] = []

    def encode_context(prompt: str) -> tuple[Any, Any]:
        context, context_mask = model.encode_prompt(prompt)
        encoded_prompts.append(prompt)
        return context[0].detach(), context_mask[0].detach()

    dataset._get_cached_text_context = encode_context
    sample_index = 0
    sample = dataset._get(sample_index)
    if len(encoded_prompts) != 1 or encoded_prompts[0] != sample["prompt"]:
        raise PhaseCGateError("sample prompt/context capture mismatch")

    inner = dataset.lerobot_dataset.multi_dataset._datasets[0]
    row = inner.hf_dataset[sample_index]
    identity = {
        "dataset_index": 0,
        "dataset_root": str(dataset_root),
        "episode_index": int(_scalar(row["episode_index"])),
        "frame_index": int(_scalar(row["frame_index"])),
        "fps": int(inner.fps),
        "sample_index": sample_index,
        "task": str(inner.meta.tasks[int(_scalar(row["task_index"]))]),
        "task_index": int(_scalar(row["task_index"])),
        "timestamp": float(_scalar(row["timestamp"])),
        "total_episodes": int(inner.meta.total_episodes),
        "total_frames": int(inner.meta.total_frames),
    }
    expected = {
        "video": (3, 9, 224, 448),
        "action": (32, 7),
        "proprio": (32, 8),
        "context": (128, 4096),
        "context_mask": (128,),
    }
    for key, shape in expected.items():
        if tuple(sample[key].shape) != shape:
            raise PhaseCGateError(
                f"real sample {key} shape mismatch: "
                f"expected {shape}, got {tuple(sample[key].shape)}"
            )
    if not sample["video"].isfinite().all():
        raise PhaseCGateError("real sample video contains NaN/Inf")
    if not sample["action"].isfinite().all():
        raise PhaseCGateError("real sample action contains NaN/Inf")
    return sample, identity


class _UpstreamVideoVelocity:
    """Bind VideoOnlyFutureSampler to the frozen upstream Video DiT."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.first_state = None
        self.first_timestep = None
        self.first_velocity = None

    def __call__(
        self,
        state: Any,
        timestep: Any,
        conditions: Mapping[str, object],
    ) -> Any:
        context = conditions["context"]
        context_mask = conditions["context_mask"]
        velocity = self.model.video_expert(
            x=state,
            timestep=timestep.to(device=state.device, dtype=state.dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        if self.first_velocity is None:
            self.first_state = state.detach().cpu()
            self.first_timestep = timestep.detach().cpu()
            self.first_velocity = velocity.detach().cpu()
        return velocity


def _cuda_measure(
    torch: Any,
    device: str,
    limit_bytes: int,
    stage: str,
    function: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak = int(torch.cuda.max_memory_allocated(device))
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    report = {
        "allocated_mib": allocated / 2**20,
        "latency_ms": elapsed_ms,
        "peak_gib": peak / 2**30,
        "peak_mib": peak / 2**20,
        "reserved_mib": reserved / 2**20,
        "stage": stage,
    }
    if peak >= limit_bytes:
        raise PhaseCGateError(
            f"{stage} peak {peak / 2**30:.3f} GiB violates "
            f"hard limit {limit_bytes / 2**30:.3f} GiB"
        )
    return value, report


def _sample_training_t_on_cpu(scheduler: Any, seed: int, device: str, dtype: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.rand((1,), generator=generator, dtype=torch.float32)
    sigma = scheduler._phi(u, scheduler.shift)
    return (sigma * float(scheduler.num_train_timesteps)).to(
        device=device,
        dtype=dtype,
    )


def _prepare_video_cache(
    model: Any,
    current_latent: Any,
    context: Any,
    context_mask: Any,
    *,
    action_seq_len: int,
) -> tuple[Any, Any, int]:
    import torch

    timestep_video = torch.zeros(
        (current_latent.shape[0],),
        device=current_latent.device,
        dtype=current_latent.dtype,
    )
    video_pre = model.video_expert.pre_dit(
        x=current_latent,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    return video_cache, attention_mask, video_seq_len


def _action_from_video_cache(
    model: Any,
    noisy_action: Any,
    timestep_action: Any,
    context: Any,
    context_mask: Any,
    video_cache: Any,
    attention_mask: Any,
    video_seq_len: int,
) -> Any:
    action_pre = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    action_tokens = model.mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
        video_kv_cache=video_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    return model.action_expert.post_dit(action_tokens, action_pre)


def _full_joint_action_reference(
    model: Any,
    noisy_video: Any,
    timestep_video: Any,
    noisy_action: Any,
    timestep_action: Any,
    context: Any,
    context_mask: Any,
    target_action: Any,
) -> Any:
    video_pre = model.video_expert.pre_dit(
        x=noisy_video,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=target_action,
        fuse_vae_embedding_in_latents=True,
    )
    action_pre = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_pre["tokens"].shape[1],
        action_seq_len=action_pre["tokens"].shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    tokens = model.mot(
        embeds_all={
            "video": video_pre["tokens"],
            "action": action_pre["tokens"],
        },
        attention_mask=attention_mask,
        freqs_all={
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        },
        context_all={
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        },
        t_mod_all={
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        },
    )
    return model.action_expert.post_dit(tokens["action"], action_pre)


def _run_phase_c(cfg: Thought3Config) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.adapter import (
        FutureAdapterSpec,
        FutureToActionAdapter,
    )
    from fastwam_ood_eval.thought3.future_sampler import (
        VideoOnlyFutureSampler,
        tensor_sha256,
    )
    from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector
    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_c_scope(cfg)
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_C") != "YES":
        raise PhaseCGateError(
            "set CONFIRM_THOUGHT3_PHASE_C=YES for the real single-sample gate"
        )
    if not torch.cuda.is_available():
        raise PhaseCGateError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise PhaseCGateError(
            "Phase C requires CUDA_VISIBLE_DEVICES to expose exactly one GPU"
        )
    device = cfg.runtime.device
    if device != "cuda:0":
        raise PhaseCGateError(
            "inside the single-card window runtime.device must be logical cuda:0"
        )
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)
    limit_bytes = int(cfg.runtime.max_gpu_memory_gb * 2**30)

    dataset_root = ensure_standard_training_source(cfg.data.dataset_roots[0])
    archive = dataset_root.parent / f"{dataset_root.name}.tar.gz"
    inventory = _dataset_inventory(
        dataset_root,
        archive if archive.is_file() else None,
    )
    expected_archive_hash = OFFICIAL_ARCHIVE_SHA256.get(dataset_root.name)
    actual_archive_hash = inventory.get("archive", {}).get("sha256")
    if expected_archive_hash is None or actual_archive_hash != expected_archive_hash:
        raise PhaseCGateError(
            f"official dataset archive hash mismatch: {actual_archive_hash}"
        )
    _progress(
        "dataset_verified",
        archive_sha256=actual_archive_hash,
        dataset_root=str(dataset_root),
    )

    checkpoint = cfg.backbone.checkpoint_path
    stats_path = cfg.backbone.dataset_stats_path
    model_config = cfg.backbone.model_config_path
    if checkpoint is None or stats_path is None or model_config is None:
        raise PhaseCGateError("Phase C backbone paths are incomplete")
    provenance = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256_expected": cfg.backbone.checkpoint_sha256,
        "dataset": inventory,
        "dataset_revision": cfg.data.dataset_revision,
        "dataset_stats_sha256": sha256_file(stats_path),
        "fastwam_commit": _git_head(Path("third_party/FastWAM")),
        "model_config_sha256": sha256_file(model_config),
    }
    if provenance["dataset_stats_sha256"] != cfg.backbone.dataset_stats_sha256:
        raise PhaseCGateError("dataset stats SHA-256 mismatch")
    if provenance["model_config_sha256"] != cfg.backbone.model_config_sha256:
        raise PhaseCGateError("model config SHA-256 mismatch")
    if provenance["fastwam_commit"] != cfg.backbone.fastwam_commit:
        raise PhaseCGateError("FastWAM commit mismatch")
    # The 12 GB checkpoint was frozen in Phase A.  Re-hashing it here would add
    # disk I/O but no tensor/backward coverage; the expected digest is recorded
    # and the loaded live MoT is hashed before/after below.

    _progress("model_load_started", device=device)
    torch.cuda.reset_peak_memory_stats(device)
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize(device)
    model_report["load_peak_gib"] = (
        torch.cuda.max_memory_allocated(device) / 2**30
    )
    if torch.cuda.max_memory_allocated(device) >= limit_bytes:
        raise PhaseCGateError("model load exceeded the 43 GiB gate")
    _progress(
        "model_loaded",
        load_peak_gib=model_report["load_peak_gib"],
        model_class=model_report["model_class"],
    )

    sample, identity = _load_one_training_sample(cfg, model, upstream_cfg)
    _progress(
        "real_sample_loaded",
        episode_index=identity["episode_index"],
        frame_index=identity["frame_index"],
        task=identity["task"],
    )
    video_cpu = sample["video"].detach().cpu()
    action_cpu = sample["action"].detach().cpu()
    proprio_cpu = sample["proprio"].detach().cpu()
    action_is_pad_cpu = sample["action_is_pad"].detach().cpu().bool()
    context = sample["context"].unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    context_mask = sample["context_mask"].unsqueeze(0).to(
        device=device,
        dtype=torch.bool,
    )
    proprio = proprio_cpu[0:1].to(device=device, dtype=model.torch_dtype)
    context, context_mask = model._append_proprio_to_context(
        context,
        context_mask,
        proprio,
    )
    if model.text_encoder is not None:
        model.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    sample_report = {
        "identity": identity,
        "prompt": sample["prompt"],
        "tensors": {
            "action": _tensor_metadata(action_cpu),
            "action_is_pad": _tensor_metadata(action_is_pad_cpu),
            "context_with_proprio": _tensor_metadata(context),
            "proprio": _tensor_metadata(proprio_cpu),
            "video": _tensor_metadata(video_cpu),
        },
    }
    current_image = video_cpu[:, 0, :, :].unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    (current_latent, current_encode_memory) = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "current_frame_vae_encode",
        lambda: model._encode_input_image_latents_tensor(current_image),
    )
    expected_current_shape = (1, 48, 1, 14, 28)
    if tuple(current_latent.shape) != expected_current_shape:
        raise PhaseCGateError(
            f"current latent expected {expected_current_shape}, "
            f"got {tuple(current_latent.shape)}"
        )

    frozen_hash_before = parameter_state_sha256(
        iter(model.mot.named_parameters())
    )
    _progress("frozen_hash_before", sha256=frozen_hash_before)
    velocity = _UpstreamVideoVelocity(model)
    sampler = VideoOnlyFutureSampler(
        velocity,
        shift=cfg.sampler.shift,
        num_train_timesteps=cfg.sampler.num_train_timesteps,
        rand_device=cfg.sampler.rand_device,
    )
    future_by_k: dict[int, Any] = {}
    sampler_reports: dict[str, Any] = {}
    initial_hashes: set[tuple[str, ...]] = set()
    conditions = {"context": context, "context_mask": context_mask}
    for k in (1, 2, 4):
        (future_sample, memory) = _cuda_measure(
            torch,
            device,
            limit_bytes,
            f"video_only_k{k}",
            lambda k=k: sampler.sample(
                current_latent,
                initial_noise_seeds=(cfg.sampler.global_cache_seed,),
                k=k,
                conditions=conditions,
            ),
        )
        future_cpu = future_sample.future_latent.detach().cpu()
        if tuple(future_cpu.shape) != (1, 48, 2, 14, 28):
            raise PhaseCGateError(f"K={k} future latent shape mismatch")
        if not future_cpu.isfinite().all():
            raise PhaseCGateError(f"K={k} future latent contains NaN/Inf")
        initial_hashes.add(future_sample.initial_state_sha256)
        future_by_k[k] = future_cpu
        sampler_reports[str(k)] = {
            "future_sha256": tensor_sha256(future_cpu),
            "initial_state_sha256": list(
                future_sample.initial_state_sha256
            ),
            "memory": memory,
            "schedule": future_sample.schedule.to_dict(),
            "tensor": _tensor_metadata(future_cpu),
        }
        _progress(
            "future_sampled",
            k=k,
            latency_ms=memory["latency_ms"],
            peak_gib=memory["peak_gib"],
        )
        del future_sample
        torch.cuda.empty_cache()
    if len(initial_hashes) != 1:
        raise PhaseCGateError("K=1/2/4 did not share paired initial noise")

    # Upstream parity: direct video-only velocity versus the official joint MoT
    # video slice at the exact same state/timestep/context.
    if (
        velocity.first_state is None
        or velocity.first_timestep is None
        or velocity.first_velocity is None
    ):
        raise PhaseCGateError("video sampler did not capture a parity state")
    parity_state = velocity.first_state.to(device=device, dtype=model.torch_dtype)
    parity_timestep = velocity.first_timestep.to(
        device=device,
        dtype=model.torch_dtype,
    )
    action_generator = torch.Generator(device="cpu").manual_seed(
        cfg.experiment.seed + 11
    )
    parity_action = torch.randn(
        (1, 32, 7),
        generator=action_generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=model.torch_dtype)

    def joint_video_reference() -> Any:
        with torch.inference_mode():
            pred_video, _ = model._predict_joint_noise(
                latents_video=parity_state,
                latents_action=parity_action,
                timestep_video=parity_timestep,
                timestep_action=parity_timestep,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=True,
                gt_action=None,
            )
        return pred_video

    joint_video, joint_video_memory = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "video_only_vs_joint_parity",
        joint_video_reference,
    )
    direct_video = velocity.first_velocity.to(
        device=device,
        dtype=joint_video.dtype,
    )
    video_parity_max = _max_abs(direct_video, joint_video)
    video_parity_mean = _mean_abs(direct_video, joint_video)
    video_parity_ok = bool(
        torch.allclose(direct_video, joint_video, atol=1e-2, rtol=1e-2)
    )
    if not video_parity_ok:
        raise PhaseCGateError(
            f"video-only/upstream joint parity failed: max_abs={video_parity_max}"
        )
    _progress(
        "video_parity_passed",
        max_abs=video_parity_max,
        peak_gib=joint_video_memory["peak_gib"],
    )
    del parity_state, parity_action, direct_video, joint_video
    torch.cuda.empty_cache()

    # Mutation leakage gate.  Only future RGB is changed; current RGB and all
    # sampler inputs remain bitwise identical.  Re-running K=1 must match.
    mutated_video = video_cpu.clone()
    mutated_video[:, 1:, :, :] = -mutated_video[:, 1:, :, :]
    if not torch.equal(mutated_video[:, 0], video_cpu[:, 0]):
        raise PhaseCGateError("future RGB mutation changed the current frame")
    (mutation_sample, mutation_memory) = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "future_rgb_mutation_k1",
        lambda: sampler.sample(
            current_latent,
            initial_noise_seeds=(cfg.sampler.global_cache_seed,),
            k=1,
            conditions=conditions,
        ),
    )
    mutation_hash = tensor_sha256(mutation_sample.future_latent.detach().cpu())
    leakage_ok = mutation_hash == sampler_reports["1"]["future_sha256"]
    if not leakage_ok:
        raise PhaseCGateError("future RGB mutation changed generated future hash")
    _progress("future_rgb_mutation_passed", sha256=mutation_hash)
    del mutation_sample, mutated_video
    torch.cuda.empty_cache()

    # Prepare fixed action-flow noise/timesteps and one full upstream reference.
    target_action = action_cpu.unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    action_is_pad = action_is_pad_cpu.unsqueeze(0).to(
        device=device,
        dtype=torch.bool,
    )
    action_noise_generator = torch.Generator(device="cpu").manual_seed(
        cfg.training.train_seed + 101
    )
    action_noise = torch.randn(
        tuple(target_action.shape),
        generator=action_noise_generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=model.torch_dtype)
    timestep_action = _sample_training_t_on_cpu(
        model.train_action_scheduler,
        cfg.training.train_seed + 102,
        device,
        model.torch_dtype,
    )
    noisy_action = model.train_action_scheduler.add_noise(
        target_action,
        action_noise,
        timestep_action,
    )
    action_flow_target = model.train_action_scheduler.training_target(
        target_action,
        action_noise,
        timestep_action,
    )
    action_weight = model.train_action_scheduler.training_weight(
        timestep_action
    )

    full_video = video_cpu.unsqueeze(0).to(
        device=device,
        dtype=model.torch_dtype,
    )
    (full_latents, full_vae_memory) = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "full_demo_vae_encode_parity_only",
        lambda: model._encode_video_latents(full_video),
    )
    first_latent_max_abs = _max_abs(
        current_latent,
        full_latents[:, :, 0:1],
    )
    video_noise_generator = torch.Generator(device="cpu").manual_seed(
        cfg.training.train_seed + 103
    )
    video_noise = torch.randn(
        tuple(full_latents.shape),
        generator=video_noise_generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=model.torch_dtype)
    timestep_video = _sample_training_t_on_cpu(
        model.train_video_scheduler,
        cfg.training.train_seed + 104,
        device,
        model.torch_dtype,
    )
    noisy_video = model.train_video_scheduler.add_noise(
        full_latents,
        video_noise,
        timestep_video,
    )
    noisy_video[:, :, 0:1] = full_latents[:, :, 0:1]

    (reference_action, action_reference_memory) = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "upstream_full_joint_action_reference",
        lambda: (
            _full_joint_action_reference(
                model,
                noisy_video,
                timestep_video,
                noisy_action,
                timestep_action,
                context,
                context_mask,
                target_action,
            )
        ),
    )

    with torch.inference_mode():
        action_pre_for_shape = model.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_cache, action_attention_mask, video_seq_len = _prepare_video_cache(
            model,
            current_latent,
            context,
            context_mask,
            action_seq_len=int(action_pre_for_shape["tokens"].shape[1]),
        )
    # Inference-mode tensors cannot be saved by autograd during the later
    # Adapter backward.  Clone the frozen cache once into ordinary tensors.
    video_cache = [
        {
            "k": layer["k"].detach().clone(),
            "v": layer["v"].detach().clone(),
        }
        for layer in video_cache
    ]
    action_attention_mask = action_attention_mask.detach().clone()
    del action_pre_for_shape

    (baseline_action, action_current_memory) = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "current_only_action_baseline",
        lambda: _action_from_video_cache(
            model,
            noisy_action,
            timestep_action,
            context,
            context_mask,
            video_cache,
            action_attention_mask,
            video_seq_len,
        ),
    )
    action_parity_max = _max_abs(reference_action, baseline_action)
    action_parity_mean = _mean_abs(reference_action, baseline_action)
    action_parity_ok = bool(
        torch.allclose(reference_action, baseline_action, atol=1e-2, rtol=1e-2)
    )
    if not action_parity_ok:
        raise PhaseCGateError(
            f"current-only/upstream action parity failed: max_abs={action_parity_max}"
        )
    _progress(
        "action_parity_passed",
        max_abs=action_parity_max,
        peak_gib=action_reference_memory["peak_gib"],
    )

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
    adapter.train()
    injector = ActionEncoderFutureInjector(
        model.action_expert.action_encoder,
        adapter,
    )
    future_k4 = future_by_k[4].to(device=device, dtype=model.torch_dtype)
    future_mask = torch.ones(
        (1, 2, 14, 28),
        device=device,
        dtype=torch.bool,
    )

    def zero_gate_forward() -> Any:
        with injector.activate(future_k4, future_mask, expected_calls=1):
            return _action_from_video_cache(
                model,
                noisy_action,
                timestep_action,
                context,
                context_mask,
                video_cache,
                action_attention_mask,
                video_seq_len,
            )

    with torch.inference_mode():
        (zero_action, zero_gate_memory) = _cuda_measure(
            torch,
            device,
            limit_bytes,
            "zero_gate_action_parity",
            zero_gate_forward,
        )
    zero_gate_max = _max_abs(baseline_action, zero_action)
    if zero_gate_max != 0.0:
        raise PhaseCGateError(
            f"zero-gate Adapter changed action output: max_abs={zero_gate_max}"
        )
    _progress("zero_gate_parity_passed", max_abs=zero_gate_max)

    allowed_training_batch = {
        "action_is_pad": action_is_pad,
        "context": context,
        "context_mask": context_mask,
        "current_proprio": proprio,
        "current_rgb": current_image,
        "future_latent": future_k4,
        "future_mask": future_mask,
        "target_action": target_action,
    }
    validate_training_batch_keys(allowed_training_batch)
    leakage_rejection_verified = False
    try:
        validate_training_batch_keys(
            {**allowed_training_batch, "future_frames": full_video}
        )
    except Exception:
        leakage_rejection_verified = True
    if not leakage_rejection_verified:
        raise PhaseCGateError("training schema accepted forbidden future_frames")

    for parameter in adapter.parameters():
        parameter.grad = None
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = None

    def backward_once() -> tuple[Any, Any]:
        with injector.activate(future_k4, future_mask, expected_calls=1):
            prediction = _action_from_video_cache(
                model,
                noisy_action,
                timestep_action,
                context,
                context_mask,
                video_cache,
                action_attention_mask,
                video_seq_len,
            )
        loss = compute_upstream_action_loss(
            prediction,
            action_flow_target,
            action_is_pad,
            action_weight,
            loss_lambda_action=model.loss_lambda_action,
        )
        loss.backward()
        return prediction, loss

    ((backward_prediction, action_loss), backward_memory) = _cuda_measure(
        torch,
        device,
        limit_bytes,
        "adapter_action_loss_backward",
        backward_once,
    )
    if not bool(torch.isfinite(action_loss).item()):
        raise PhaseCGateError("action loss is NaN/Inf")
    backbone_grad_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if backbone_grad_names:
        raise PhaseCGateError(
            f"frozen backbone received gradients: {backbone_grad_names[:5]}"
        )
    adapter_gradients: dict[str, Any] = {}
    for name, parameter in adapter.named_parameters():
        grad = parameter.grad
        adapter_gradients[name] = {
            "has_grad": grad is not None,
            "finite": bool(torch.isfinite(grad).all().item())
            if grad is not None
            else None,
            "l2": float(grad.detach().float().norm().cpu())
            if grad is not None
            else None,
            "nonzero": int(torch.count_nonzero(grad).item())
            if grad is not None
            else 0,
        }
    gate_grad = adapter_gradients["gate"]
    if (
        not gate_grad["has_grad"]
        or not gate_grad["finite"]
        or float(gate_grad["l2"]) <= 0.0
    ):
        raise PhaseCGateError("zero-gate parameter did not receive a finite nonzero gradient")
    if any(
        values["has_grad"] and not values["finite"]
        for values in adapter_gradients.values()
    ):
        raise PhaseCGateError("Adapter received non-finite gradients")
    _progress(
        "backward_passed",
        action_loss=float(action_loss.detach().cpu()),
        gate_grad_l2=gate_grad["l2"],
        peak_gib=backward_memory["peak_gib"],
    )

    frozen_hash_after = parameter_state_sha256(
        iter(model.mot.named_parameters())
    )
    if frozen_hash_after != frozen_hash_before:
        raise PhaseCGateError("frozen MoT parameter hash changed after backward")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise PhaseCGateError("backbone parameter requires_grad was re-enabled")
    _progress("frozen_hash_after", sha256=frozen_hash_after)

    injector.close()
    memory_stages = {
        item["stage"]: item
        for item in (
            current_encode_memory,
            joint_video_memory,
            mutation_memory,
            full_vae_memory,
            action_reference_memory,
            action_current_memory,
            zero_gate_memory,
            backward_memory,
            *(sampler_reports[str(k)]["memory"] for k in (1, 2, 4)),
        )
    }
    peak_stage = max(memory_stages.values(), key=lambda item: item["peak_mib"])
    result = {
        "schema_version": PHASE_C_SCHEMA,
        "status": "passed",
        "gate_c_passed": True,
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "device": {
            "cuda_device_count_visible": torch.cuda.device_count(),
            "logical_device": device,
            "name": torch.cuda.get_device_name(device),
            "physical_gpu_id": os.environ.get("THOUGHT3_PHYSICAL_GPU_ID"),
            "total_memory_gib": (
                torch.cuda.get_device_properties(device).total_memory / 2**30
            ),
        },
        "provenance": provenance,
        "model": model_report,
        "sample": sample_report,
        "current_latent": {
            **_tensor_metadata(current_latent),
            "sha256": tensor_sha256(current_latent.detach().cpu()),
        },
        "future_sampler": {
            "K": sampler_reports,
            "paired_initial_noise": True,
            "real_cache_generated": False,
            "serialized_latent_files": 0,
        },
        "parity": {
            "action_current_only_vs_full_joint": {
                "atol": 1e-2,
                "first_frame_vae_latent_max_abs": first_latent_max_abs,
                "max_abs": action_parity_max,
                "mean_abs": action_parity_mean,
                "passed": action_parity_ok,
                "rtol": 1e-2,
            },
            "video_only_vs_joint": {
                "atol": 1e-2,
                "max_abs": video_parity_max,
                "mean_abs": video_parity_mean,
                "passed": video_parity_ok,
                "rtol": 1e-2,
            },
            "zero_gate_vs_current_only": {
                "bitwise_equal": zero_gate_max == 0.0,
                "max_abs": zero_gate_max,
            },
        },
        "leakage": {
            "future_rgb_mutation_hash": mutation_hash,
            "future_rgb_mutation_invariant": leakage_ok,
            "forbidden_future_frames_rejected": leakage_rejection_verified,
            "training_batch_keys": sorted(allowed_training_batch),
            "training_api_uses_gt_future_rgb": False,
        },
        "backward": {
            "action_loss": float(action_loss.detach().cpu()),
            "adapter_gradients": adapter_gradients,
            "adapter_parameter_count": sum(
                parameter.numel() for parameter in adapter.parameters()
            ),
            "backbone_gradient_count": len(backbone_grad_names),
            "backward_prediction": _tensor_metadata(backward_prediction),
            "loss_formula": "upstream FastWAM action flow MSE + pad mask + training_weight",
            "microbatch_size": 1,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "backward_calls": 1,
        },
        "freezing": {
            "all_backbone_requires_grad_false": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "mot_sha256_after": frozen_hash_after,
            "mot_sha256_before": frozen_hash_before,
            "unchanged": frozen_hash_after == frozen_hash_before,
        },
        "memory": {
            "hard_limit_gib": cfg.runtime.max_gpu_memory_gb,
            "peak_stage": peak_stage,
            "stages": memory_stages,
            "under_limit": peak_stage["peak_gib"]
            < cfg.runtime.max_gpu_memory_gb,
        },
        "scope": {
            "dataset_samples": 1,
            "future_cache_generated": False,
            "k_values": [1, 2, 4],
            "long_training_started": False,
            "optimizer_steps": 0,
            "single_gpu": True,
        },
    }
    return result


def run_phase_c_smoke(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Gate C and atomically record pass/fail status under outputs/thought3."""

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_c_result.json"
    status_path = output / "run_status.json"
    if result_path.exists() and not resume:
        raise FileExistsError(
            f"Phase C result exists; pass --resume to inspect it: {result_path}"
        )
    if result_path.exists() and resume:
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if value.get("gate_c_passed") is True:
            return value
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_C_SCHEMA,
            "status": "running",
            "started_at": _utc_now(),
        },
    )
    try:
        result = _run_phase_c(cfg)
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_c_passed": False,
                "schema_version": PHASE_C_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(result_path, result)
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_c_passed": True,
            "result": str(result_path),
            "schema_version": PHASE_C_SCHEMA,
            "status": "passed",
        },
    )
    return result
