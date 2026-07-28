from __future__ import annotations

from pathlib import Path

import pytest

from fastwam_ood_eval.thought3.config import (
    Thought3ConfigError,
    load_thought3_config,
)
from thought3_test_utils import write_thought3_config


def test_all_committed_thought3_configs_validate():
    paths = sorted(Path("configs/thought3").glob("*.yaml"))
    assert len(paths) == 19
    assert Path("configs/thought3/phase_c_single_sample.yaml") in paths
    assert Path("configs/thought3/phase_d_cache_smoke.yaml") in paths
    assert Path("configs/thought3/phase_e_training_smoke.yaml") in paths
    assert (
        Path("configs/thought3/phase_e1_overfit_diagnostic.yaml")
        in paths
    )
    assert (
        Path("configs/thought3/phase_e2_eight_sample_diagnostic.yaml")
        in paths
    )
    assert (
        Path("configs/thought3/phase_e3_multiflow_diagnostic.yaml")
        in paths
    )
    assert (
        Path("configs/thought3/phase_e3_multiflow_diagnostic_v2.yaml")
        in paths
    )
    for path in paths:
        cfg = load_thought3_config(path)
        assert cfg.schema_version == "thought3.config.v1"
        assert "thought3" in cfg.experiment.output_dir.parts
        assert "thought3" in cfg.cache.root.parts


def test_variant_k_and_adapter_invariants_are_fail_closed(tmp_path):
    path = write_thought3_config(tmp_path)
    with pytest.raises(Thought3ConfigError, match="requires sampler.active_k=1"):
        load_thought3_config(path, ["sampler.active_k=4"])
    with pytest.raises(Thought3ConfigError, match="requires adapter.enabled=true"):
        load_thought3_config(path, ["adapter.enabled=false"])
    with pytest.raises(Thought3ConfigError, match="action_denoise_steps"):
        load_thought3_config(path, ["runtime.action_denoise_steps=10"])


def test_online_cache_and_lora_are_disabled_in_first_protocol(tmp_path):
    path = write_thought3_config(tmp_path)
    with pytest.raises(Thought3ConfigError, match="online_use_cache"):
        load_thought3_config(path, ["runtime.online_use_cache=true"])
    with pytest.raises(Thought3ConfigError, match="lora_enabled=false"):
        load_thought3_config(path, ["adapter.lora_enabled=true"])


def test_unknown_config_key_is_not_silently_ignored(tmp_path):
    path = write_thought3_config(tmp_path)
    with pytest.raises(Thought3ConfigError, match="Unknown Thought3 training"):
        load_thought3_config(path, ["training.learnng_rate=0.1"])
