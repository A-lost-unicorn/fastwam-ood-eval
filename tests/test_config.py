from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import write_config
from fastwam_ood_eval.config import ConfigError, load_config, validate_hardware_inventory


def test_config_load_and_override(tmp_path):
    path = write_config(tmp_path)
    cfg = load_config(path, ["benchmark.episodes_per_task=9", "experiment.save_video=true"])
    assert cfg.benchmark.episodes_per_task == 9
    assert cfg.experiment.save_video is True


def test_illegal_perturbation_rejected(tmp_path):
    path = write_config(tmp_path, perturbation=True)
    with pytest.raises(ConfigError, match="illegal perturbation categories"):
        load_config(path, ["perturbation.category=[invented_noise]"])


def test_invalid_worker_count_rejected(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="workers_per_gpu"):
        load_config(path, ["hardware.workers_per_gpu=2"])


def test_future_imagination_requires_matching_upstream_variant(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="inconsistent with policy.variant"):
        load_config(
            path,
            [
                "policy.variant=joint_wam",
                "policy.test_time_future_imagination=false",
                "checkpoint.model_name=libero_joint_2cam224_1e-4",
            ],
        )


def test_joint_variant_cannot_reuse_uncond_checkpoint_name(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="checkpoint filename"):
        load_config(
            path,
            [
                "benchmark.backend=libero",
                "policy.variant=joint_wam",
                "policy.test_time_future_imagination=true",
                "checkpoint.model_name=libero_joint_2cam224_1e-4",
                "checkpoint.path=checkpoints/libero_uncond_2cam224.pt",
            ],
        )


def test_hardware_inventory_accepts_three_sufficient_gpus(tmp_path):
    cfg = load_config(write_config(tmp_path))
    cfg = replace(cfg, benchmark=replace(cfg.benchmark, backend="libero"))
    validate_hardware_inventory(
        cfg,
        cuda_available=True,
        device_memory_gb=[47.4, 47.4, 47.4],
    )


def test_hardware_inventory_rejects_missing_or_small_gpus(tmp_path):
    cfg = load_config(write_config(tmp_path))
    cfg = replace(cfg, benchmark=replace(cfg.benchmark, backend="libero"))
    with pytest.raises(ConfigError, match="only 2 CUDA devices"):
        validate_hardware_inventory(
            cfg,
            cuda_available=True,
            device_memory_gb=[47.4, 47.4],
        )
    with pytest.raises(ConfigError, match="below configured memory budget"):
        validate_hardware_inventory(
            cfg,
            cuda_available=True,
            device_memory_gb=[16.0, 16.0, 16.0],
        )
