from __future__ import annotations

from fastwam_ood_eval.thought3.cache_builder import build_cache
from fastwam_ood_eval.thought3.cache_planner import write_cache_plan
from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.io_utils import load_jsonl
from fastwam_ood_eval.thought3.trainer import run_mock_training
from fastwam_ood_eval.thought3.checkpointing import adapter_state_sha256
from safetensors.torch import load_file
from thought3_test_utils import write_thought3_config


def test_mock_action_flow_training_decreases_validation_loss_and_resumes(tmp_path):
    cfg = load_thought3_config(
        write_thought3_config(
            tmp_path,
            sample_count=8,
            max_steps=60,
            overrides={
                "training": {
                    "checkpoint_interval": 100,
                    "learning_rate": 0.01,
                }
            },
        )
    )
    write_cache_plan(cfg)
    build_cache(cfg, resume=False)
    result = run_mock_training(cfg, resume=False)
    assert (
        result["final_validation_action_loss"]
        < result["initial_validation_action_loss"]
    )
    assert result["trainable_parameter_count"] > 0
    assert result["uses_ground_truth_future_input"] is False
    rows = load_jsonl(cfg.experiment.output_dir / "train_metrics.jsonl")
    assert len(rows) == 60
    assert not any(row["nan_or_inf"] for row in rows)
    resumed = run_mock_training(cfg, resume=True)
    assert resumed["resumed_from_step"] == 60
    assert len(
        load_jsonl(cfg.experiment.output_dir / "train_metrics.jsonl")
    ) == 60


def test_interrupted_and_resumed_training_matches_uninterrupted_weights(tmp_path):
    shared = {
        "training": {
            "checkpoint_interval": 6,
            "learning_rate": 0.01,
        }
    }
    interrupted_cfg = load_thought3_config(
        write_thought3_config(
            tmp_path / "interrupted",
            sample_count=8,
            max_steps=12,
            overrides=shared,
        )
    )
    write_cache_plan(interrupted_cfg)
    build_cache(interrupted_cfg, resume=False)
    partial = run_mock_training(
        interrupted_cfg,
        resume=False,
        stop_after_steps=6,
    )
    assert partial["status"] == "intentional_test_interruption"
    resumed = run_mock_training(interrupted_cfg, resume=True)
    assert resumed["resumed_from_step"] == 6

    uninterrupted_path = write_thought3_config(
        tmp_path / "uninterrupted",
        sample_count=8,
        max_steps=12,
        overrides={
            **shared,
            "cache": {
                "root": str(interrupted_cfg.cache.root),
                "shard_size": interrupted_cfg.cache.shard_size,
                "required_free_space_fraction": 0.0,
            },
        },
    )
    uninterrupted_cfg = load_thought3_config(uninterrupted_path)
    uninterrupted = run_mock_training(uninterrupted_cfg, resume=False)
    resumed_adapter = (
        interrupted_cfg.experiment.output_dir
        / "checkpoints"
        / "step_00000012"
        / "adapter.safetensors"
    )
    uninterrupted_adapter = (
        uninterrupted_cfg.experiment.output_dir
        / "checkpoints"
        / "step_00000012"
        / "adapter.safetensors"
    )
    assert resumed["status"] == uninterrupted["status"] == "complete"
    assert adapter_state_sha256(
        load_file(str(resumed_adapter))
    ) == adapter_state_sha256(load_file(str(uninterrupted_adapter)))
