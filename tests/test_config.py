from __future__ import annotations

import pytest

from conftest import write_config
from fastwam_ood_eval.config import ConfigError, load_config


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

