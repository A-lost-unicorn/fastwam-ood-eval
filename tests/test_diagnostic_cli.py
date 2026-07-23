from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from conftest import write_config
from fastwam_ood_eval import cli
from fastwam_ood_eval.config import load_config
from fastwam_ood_eval.diagnostics.diagnostic_runner import load_source_jobs
from fastwam_ood_eval.evaluation.jobs import plan_jobs, shard_jobs, write_jobs


def _diagnostic_fixture(tmp_path: Path, *, episodes: int = 2):
    config_path = write_config(tmp_path / "config", episodes=episodes)
    source_dir = tmp_path / "source"
    source_cfg = load_config(
        config_path,
        [
            "experiment.name=source",
            f"experiment.output_dir={source_dir}",
        ],
    )
    write_jobs(source_dir / "job_manifest.jsonl", plan_jobs(source_cfg))
    (source_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "source",
                "job_count": len(plan_jobs(source_cfg)),
                "config": source_cfg.to_dict(),
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "read_only_sentinel.txt").write_text("unchanged", encoding="utf-8")
    output_dir = tmp_path / "diagnostic"
    overrides = [
        "experiment.name=diagnostic",
        f"experiment.output_dir={output_dir}",
        "diagnostics.enabled=true",
        "diagnostics.source_experiment_id=source",
        f"diagnostics.source_output_dir={source_dir}",
        "diagnostics.num_video_frames=5",
        "diagnostics.max_probes_per_episode=1",
        "diagnostics.probe_strategy=first",
    ]
    cfg = load_config(config_path, overrides)
    return config_path, overrides, cfg, source_dir, output_dir


def _cli_config_args(config_path: Path, overrides: list[str]) -> list[str]:
    args = ["--config", str(config_path)]
    for override in overrides:
        args.extend(("--set", override))
    return args


def test_parser_exposes_diagnostics_without_removing_existing_commands():
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    commands = set(subparsers.choices)
    assert {
        "doctor",
        "fetch-upstreams",
        "plan",
        "evaluate",
        "distributed-evaluate",
        "aggregate",
        "report",
        "review-failures",
    } <= commands
    assert {
        "diagnose-future",
        "distributed-diagnose-future",
        "aggregate-diagnostics",
        "report-diagnostics",
    } <= commands

    parsed = parser.parse_args(
        [
            "diagnose-future",
            "--config",
            "diagnostic.yaml",
            "--device",
            "cuda:0",
            "--dry-run",
            "--rerun",
            "failed",
            "--overwrite",
            "--set",
            "diagnostics.max_probes_per_episode=1",
        ]
    )
    assert parsed.command == "diagnose-future"
    assert parsed.device == "cuda:0"
    assert parsed.dry_run is True
    assert parsed.rerun == "failed"
    assert parsed.overwrite is True


def test_original_mock_evaluate_core_output_contract_is_unchanged(tmp_path, capsys):
    config_path = write_config(tmp_path / "thought1", episodes=1)
    common = [
        "--config",
        str(config_path),
        "--set",
        "benchmark.tasks=[0]",
    ]
    assert cli.main(["plan", *common]) == 0
    capsys.readouterr()
    assert cli.main(["evaluate", *common]) == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result == {"assigned": 1, "completed": 1, "pending": 1}

    cfg = load_config(config_path, ["benchmark.tasks=[0]"])
    row = json.loads(
        (cfg.experiment.output_dir / "workers/rank_0/episode_results.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    job = plan_jobs(cfg)[0]
    assert {
        "experiment_id": row["experiment_id"],
        "job_id": row["job_id"],
        "suite": row["suite"],
        "task_id": row["task_id"],
        "episode_index": row["episode_index"],
        "episode_seed": row["episode_seed"],
        "condition": row["condition"],
        "status": row["status"],
        "policy_variant": row["policy_variant"],
        "test_time_future_imagination": row["test_time_future_imagination"],
        "worker_rank": row["worker_rank"],
        "action_chunk_shape": row["action_chunk_shape"],
    } == {
        "experiment_id": job.experiment_id,
        "job_id": job.job_id,
        "suite": job.suite,
        "task_id": 0,
        "episode_index": 0,
        "episode_seed": job.episode_seed,
        "condition": "clean",
        "status": "completed",
        "policy_variant": "mock",
        "test_time_future_imagination": False,
        "worker_rank": 0,
        "action_chunk_shape": [2, 7],
    }


def test_diagnostic_dry_run_is_read_only_and_loads_no_runtime(
    tmp_path, monkeypatch, capsys
):
    config_path, overrides, _, source_dir, output_dir = _diagnostic_fixture(tmp_path)
    before = {
        path.relative_to(source_dir): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic dry-run loaded a model, probe, or environment")

    monkeypatch.setattr(cli, "_make_policy", forbidden)
    monkeypatch.setattr(cli, "_make_environment", forbidden)
    monkeypatch.setattr(cli, "FastWAMFutureProbe", forbidden)
    status = cli.main(
        ["diagnose-future", *_cli_config_args(config_path, overrides), "--dry-run"]
    )

    assert status == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "assigned": 4,
        "pending": 4,
        "completed": 0,
        "probes": 0,
        "skipped_by_resume": 0,
    }
    assert not output_dir.exists()
    after = {
        path.relative_to(source_dir): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_distributed_diagnostic_dry_run_shards_source_jobs(
    tmp_path, monkeypatch, capsys
):
    config_path, overrides, cfg, _, output_dir = _diagnostic_fixture(tmp_path, episodes=3)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    status = cli.main(
        [
            "distributed-diagnose-future",
            *_cli_config_args(config_path, overrides),
            "--dry-run",
        ]
    )

    assert status == 0
    result = json.loads(capsys.readouterr().out)
    expected = len(shard_jobs(load_source_jobs(cfg), rank=1, world_size=3))
    assert result["assigned"] == expected
    assert result["pending"] == expected
    assert result["completed"] == 0
    assert not output_dir.exists()
    assert cli.os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"


def test_diagnostic_command_requires_enabled_config(tmp_path, capsys):
    config_path = write_config(tmp_path, episodes=1)
    status = cli.main(["diagnose-future", "--config", str(config_path), "--dry-run"])

    assert status == 2
    assert "diagnostics are disabled" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("plan", []),
        ("evaluate", ["--dry-run"]),
        ("distributed-evaluate", ["--dry-run"]),
    ],
)
def test_thought1_commands_reject_enabled_diagnostics_without_writing(
    tmp_path, capsys, command, extra
):
    config_path, overrides, _, _, output_dir = _diagnostic_fixture(tmp_path)
    status = cli.main(
        [command, *_cli_config_args(config_path, overrides), *extra]
    )

    assert status == 2
    error = capsys.readouterr().err
    assert "diagnostics.enabled=true" in error
    assert "diagnose-future" in error
    assert not output_dir.exists()


def test_probe_capability_failure_precedes_environment_construction(
    tmp_path, monkeypatch
):
    _, _, cfg, _, _ = _diagnostic_fixture(tmp_path, episodes=1)
    calls: list[str] = []

    class Policy:
        def close(self):
            calls.append("policy.close")

    def make_policy(config, device):
        calls.append("policy")
        return Policy()

    def make_probe(policy, *, mode):
        calls.append("probe")
        assert mode == cfg.diagnostics.mode
        raise RuntimeError("checkpoint is not action-conditioned")

    def make_environment(config):
        calls.append("environment")
        raise AssertionError("environment must not be constructed")

    monkeypatch.setattr(cli, "validate_runtime_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_select_diagnostic_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(cli, "_make_policy", make_policy)
    monkeypatch.setattr(cli, "FastWAMFutureProbe", make_probe)
    monkeypatch.setattr(cli, "_make_environment", make_environment)

    with pytest.raises(RuntimeError, match="not action-conditioned"):
        cli._diagnose_future_worker(cfg)
    assert calls == ["policy", "probe", "policy.close"]


def test_thought1_output_is_rejected_before_model_load(tmp_path, monkeypatch):
    _, _, cfg, _, output_dir = _diagnostic_fixture(tmp_path, episodes=1)
    output_dir.mkdir(parents=True)
    manifest = output_dir / "experiment_manifest.json"
    manifest.write_text('{"experiment_id":"thought1"}\n', encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("model must not load into a Thought 1 output namespace")

    monkeypatch.setattr(cli, "_make_policy", forbidden)
    with pytest.raises(RuntimeError, match="Thought 1 evaluation output"):
        cli._diagnose_future_worker(cfg)
    assert manifest.read_text(encoding="utf-8") == '{"experiment_id":"thought1"}\n'
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "experiment_manifest.json"
    ]


def test_diagnostic_aggregate_and_report_commands(tmp_path, capsys):
    experiment_dir = tmp_path / "empty_diagnostics"
    status = cli.main(
        ["aggregate-diagnostics", "--experiment-dir", str(experiment_dir)]
    )
    assert status == 0
    aggregate_result = json.loads(capsys.readouterr().out)
    assert aggregate_result["clips"] == 0
    assert aggregate_result["episodes"] == 0
    assert (experiment_dir / "summary/all_diagnostics.csv").is_file()

    status = cli.main(
        ["report-diagnostics", "--experiment-dir", str(experiment_dir)]
    )
    assert status == 0
    report_path = Path(capsys.readouterr().out.strip())
    assert report_path == experiment_dir / "summary/thought2_report.md"
    assert report_path.is_file()
