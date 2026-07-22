from __future__ import annotations

import pytest

from conftest import write_config
from fastwam_ood_eval.config import ConfigError, load_config
from fastwam_ood_eval.evaluation.jobs import plan_jobs


def test_diagnostics_are_disabled_and_omitted_by_default(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.diagnostics.enabled is False
    assert "diagnostics" not in cfg.to_dict()


def test_diagnostics_do_not_change_existing_job_ids(tmp_path):
    path = write_config(tmp_path, episodes=3)
    baseline = load_config(path)
    diagnostic = load_config(
        path,
        [
            "diagnostics.enabled=true",
            "diagnostics.source_experiment_id=thought1-source",
            "diagnostics.num_inference_steps=7",
            "diagnostics.max_probes_per_episode=1",
            "diagnostics.probe_strategy=first",
        ],
    )
    assert [job.job_id for job in plan_jobs(baseline)] == [
        job.job_id for job in plan_jobs(diagnostic)
    ]


def test_invalid_diagnostic_video_length_is_rejected(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match=r"T % 4 == 1"):
        load_config(path, ["diagnostics.num_video_frames=8"])


def test_explicit_probe_strategy_requires_bounded_indices(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="requires explicit indices"):
        load_config(path, ["diagnostics.probe_strategy=explicit_replan_indices"])
    with pytest.raises(ConfigError, match="cannot exceed max_probes_per_episode"):
        load_config(
            path,
            [
                "diagnostics.probe_strategy=explicit_replan_indices",
                "diagnostics.explicit_replan_indices=[0,2]",
                "diagnostics.max_probes_per_episode=1",
            ],
        )

