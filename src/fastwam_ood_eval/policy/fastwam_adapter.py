"""Thin adapter over Fast-WAM's official LIBERO evaluator helpers.

No model or preprocessing implementation is copied here. The adapter composes the
official Hydra config and calls the checkpoint loader and action prediction helper
from the pinned upstream checkout.
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any

from fastwam_ood_eval.config import EvalConfig
from fastwam_ood_eval.policy.base import BasePolicy, PolicyOutput

LOGGER = logging.getLogger(__name__)


class FastWAMAdapter(BasePolicy):
    def __init__(self, cfg: EvalConfig, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.task_description = ""
        self._first = True
        self._load_upstream()

    def _load_upstream(self) -> None:
        from fastwam_ood_eval.envs.libero_adapter import configure_libero_package

        if self.cfg.benchmark.backend == "libero":
            configure_libero_package(
                Path("third_party/LIBERO"),
                Path("outputs/runtime/libero"),
            )
        elif self.cfg.benchmark.backend == "libero_plus":
            configure_libero_package(
                Path("third_party/LIBERO-plus"),
                Path("outputs/runtime/libero_plus"),
            )
        fastwam_root = Path("third_party/FastWAM").resolve()
        experiment_root = fastwam_root / "experiments" / "libero"
        for path in (experiment_root, fastwam_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            import torch
            from hydra import compose, initialize_config_dir
            from hydra.utils import instantiate
            from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
            from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
            from experiments.libero import eval_libero_single as official
        except ImportError as exc:
            raise RuntimeError(
                "Fast-WAM dependencies are unavailable. Follow docs/environment_setup.md and install "
                "third_party/FastWAM before real evaluation."
            ) from exc

        if self.cfg.checkpoint.config_path is None:
            raise ValueError("checkpoint.config_path is required")
        config_path = self.cfg.checkpoint.config_path.resolve()
        overrides = [
            f"task={self.cfg.checkpoint.model_name}",
            f"ckpt={self.cfg.checkpoint.path}",
            f"mixed_precision={official._normalize_mixed_precision(self.cfg.hardware.precision.replace('fp32', 'no'))}",
            f"EVALUATION.dataset_stats_path={self.cfg.checkpoint.dataset_stats_path}",
            f"EVALUATION.device={self.device}",
            f"EVALUATION.replan_steps={self.cfg.benchmark.control_horizon}",
        ]
        with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
            upstream_cfg = compose(config_name=config_path.stem, overrides=overrides)
        dtype = official._mixed_precision_to_model_dtype(upstream_cfg.get("mixed_precision", "bf16"))
        model = instantiate(upstream_cfg.model, model_dtype=dtype, device=self.device)
        expected_classes = {
            "fastwam": "FastWAM",
            "joint_wam": "FastWAMJoint",
            "idm": "FastWAMIDM",
        }
        expected_class = expected_classes[self.cfg.policy.variant]
        actual_class = type(model).__name__
        if actual_class != expected_class:
            raise RuntimeError(
                f"policy.variant={self.cfg.policy.variant} expected upstream class {expected_class}, "
                f"but Hydra instantiated {actual_class}"
            )
        if self.cfg.policy.test_time_future_imagination:
            infer_parameters = inspect.signature(model.infer_action).parameters
            if "num_video_frames" not in infer_parameters:
                raise RuntimeError(
                    f"{actual_class}.infer_action has no num_video_frames argument; "
                    "future-imagination semantics cannot be verified"
                )
        official._load_model_checkpoint(model, str(self.cfg.checkpoint.path))
        self.model = model.to(self.device).eval()
        stats = load_dataset_stats_from_json(str(self.cfg.checkpoint.dataset_stats_path))
        processor: FastWAMProcessor = instantiate(upstream_cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(stats)
        self.processor = processor
        self.upstream_cfg = upstream_cfg
        self.official = official
        self.torch = torch
        video_size = upstream_cfg.data.train.get("video_size", [224, 448])
        self.input_h, self.input_w = int(video_size[0]), int(video_size[1])
        configured_horizon = upstream_cfg.EVALUATION.get("action_horizon")
        self.action_horizon = int(configured_horizon or (int(upstream_cfg.data.train.num_frames) - 1))
        if self.cfg.hardware.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def reset(self, task_description: str, *, seed: int | None = None) -> None:
        self.task_description = task_description
        self.upstream_cfg.seed = seed
        self._first = True
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def act(self, observation: dict[str, Any]) -> PolicyOutput:
        started = time.perf_counter()
        with self.torch.inference_mode():
            actions, images, _ = self.official._predict_action_chunk(
                obs=observation,
                task_description=self.task_description,
                model=self.model,
                processor=self.processor,
                cfg=self.upstream_cfg,
                action_horizon=self.action_horizon,
                input_w=self.input_w,
                input_h=self.input_h,
                model_device=self.device,
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)
        latency = (time.perf_counter() - started) * 1000.0
        warmup = latency if self._first else None
        self._first = False
        allocated = reserved = 0.0
        if self.torch.cuda.is_available():
            allocated = self.torch.cuda.memory_allocated(self.device) / 2**20
            reserved = self.torch.cuda.memory_reserved(self.device) / 2**20
        primary = images.get("image")
        return PolicyOutput(
            actions=actions,
            latency_ms=latency,
            warmup_latency_ms=warmup,
            action_chunk_shape=list(actions.shape),
            observation_image_shape=list(primary.shape) if hasattr(primary, "shape") else None,
            gpu_memory_allocated_mb=allocated,
            gpu_memory_reserved_mb=reserved,
        )

    def peak_memory_mb(self) -> float:
        if self.torch.cuda.is_available():
            return float(self.torch.cuda.max_memory_allocated(self.device) / 2**20)
        return 0.0

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
