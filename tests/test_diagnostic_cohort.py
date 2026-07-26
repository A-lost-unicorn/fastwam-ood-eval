from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import write_config
from fastwam_ood_eval import cli
from fastwam_ood_eval.config import load_config
from fastwam_ood_eval.diagnostics import diagnostic_cohort
from fastwam_ood_eval.diagnostics.diagnostic_cohort import (
    plan_diagnostic_cohort,
    ratify_diagnostic_cohort,
    validate_diagnostic_cohort,
)
from fastwam_ood_eval.diagnostics.diagnostic_runner import load_source_jobs
from fastwam_ood_eval.evaluation.jobs import plan_jobs, write_jobs


def _source(tmp_path: Path, *, episodes: int = 3):
    config_path = write_config(
        tmp_path / "config",
        episodes=episodes,
        output_name="source_output",
    )
    cfg = load_config(config_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "source",
                "config": cfg.to_dict(),
                "provenance": {
                    "checkpoint_hash": "mock",
                    "fastwam_commit": "mock",
                },
                "status": "planned_not_run",
            }
        ),
        encoding="utf-8",
    )
    write_jobs(source / "job_manifest.jsonl", plan_jobs(cfg))
    return cfg, source


def test_outcome_blind_cohort_is_deterministic_and_runner_uses_exact_ids(
    tmp_path,
):
    cfg, source = _source(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    planned = plan_diagnostic_cohort(
        source_dir=source,
        output_path=cohort_path,
        seed=99,
        per_stratum=2,
        stratum_fields=["task_id"],
        anchor_episode_indices=[0],
    )
    assert planned["status"] == "draft_not_frozen"
    assert planned["selected_jobs"] == 4
    assert planned["runnable_strata"] == 2
    payload = json.loads(cohort_path.read_text())
    assert payload["selection"]["outcome_fields_read"] is False
    assert payload["selection"]["episode_result_files_read"] is False
    assert payload["selection"]["selected_job_count"] == 4

    diagnostic_cfg = replace(
        cfg,
        experiment=replace(
            cfg.experiment,
            name="diagnostic",
            output_dir=tmp_path / "diagnostic",
        ),
        diagnostics=replace(
            cfg.diagnostics,
            enabled=True,
            mode="unconditional_future",
            source_experiment_id="source",
            source_output_dir=source,
            cohort_manifest_path=cohort_path,
        ),
    )
    selected = load_source_jobs(diagnostic_cfg)
    assert [job.job_id for job in selected] == planned["selected_job_ids"]
    assert {
        job.task_id
        for job in selected
        if job.episode_index == 0
    } == {0, 1}
    assert "cohort_manifest_path" in diagnostic_cfg.to_dict()["diagnostics"]
    assert "require_frozen_cohort" not in diagnostic_cfg.to_dict()["diagnostics"]

    repeated_path = tmp_path / "cohort_repeat.json"
    repeated = plan_diagnostic_cohort(
        source_dir=source,
        output_path=repeated_path,
        seed=99,
        per_stratum=2,
        stratum_fields=["task_id"],
        anchor_episode_indices=[0],
    )
    assert repeated["cohort_id"] == planned["cohort_id"]
    assert repeated["selected_job_ids"] == planned["selected_job_ids"]


def test_formal_runner_rejects_draft_cohort(tmp_path):
    cfg, source = _source(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    plan_diagnostic_cohort(
        source_dir=source,
        output_path=cohort_path,
        seed=99,
        per_stratum=1,
        stratum_fields=["task_id"],
    )
    diagnostic_cfg = replace(
        cfg,
        experiment=replace(
            cfg.experiment,
            name="diagnostic",
            output_dir=tmp_path / "diagnostic",
        ),
        diagnostics=replace(
            cfg.diagnostics,
            enabled=True,
            mode="unconditional_future",
            source_experiment_id="source",
            source_output_dir=source,
            cohort_manifest_path=cohort_path,
            require_frozen_cohort=True,
        ),
    )
    assert diagnostic_cfg.to_dict()["diagnostics"][
        "require_frozen_cohort"
    ] is True
    with pytest.raises(RuntimeError, match="require a certified frozen cohort"):
        load_source_jobs(diagnostic_cfg)


def test_cohort_freeze_requires_clean_tree_and_no_source_outcomes(
    tmp_path,
    monkeypatch,
):
    cfg, source = _source(tmp_path)
    monkeypatch.setattr(diagnostic_cohort, "git_dirty", lambda path: False)
    monkeypatch.setattr(
        diagnostic_cohort, "git_commit", lambda path: "clean-commit"
    )
    frozen_path = tmp_path / "frozen.json"
    frozen = plan_diagnostic_cohort(
        source_dir=source,
        output_path=frozen_path,
        seed=7,
        per_stratum=1,
        stratum_fields=["task_id"],
        freeze=True,
    )
    assert frozen["frozen"] is True
    assert frozen["status"] == "frozen_before_source_outcomes"
    formal_cfg = replace(
        cfg,
        experiment=replace(
            cfg.experiment,
            name="formal_diagnostic",
            output_dir=tmp_path / "formal_diagnostic",
        ),
        diagnostics=replace(
            cfg.diagnostics,
            enabled=True,
            mode="unconditional_future",
            source_experiment_id="source",
            source_output_dir=source,
            cohort_manifest_path=frozen_path,
            require_frozen_cohort=True,
        ),
    )
    assert [job.job_id for job in load_source_jobs(formal_cfg)] == frozen[
        "selected_job_ids"
    ]

    outcome = source / "workers/rank_0/episode_results.jsonl"
    outcome.parent.mkdir(parents=True)
    outcome.write_text('{"success": true}\n')
    with pytest.raises(RuntimeError, match="outcome JSONL"):
        plan_diagnostic_cohort(
            source_dir=source,
            output_path=tmp_path / "too_late.json",
            seed=7,
            per_stratum=1,
            stratum_fields=["task_id"],
            freeze=True,
        )


def test_cohort_validation_detects_source_job_manifest_change(tmp_path):
    _, source = _source(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    plan_diagnostic_cohort(
        source_dir=source,
        output_path=cohort_path,
        seed=1,
        per_stratum=1,
        stratum_fields=["task_id"],
    )
    with (source / "job_manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="no longer matches"):
        validate_diagnostic_cohort(cohort_path, source)


def test_pre_outcome_draft_can_be_ratified_before_diagnostic_metrics(
    tmp_path,
    monkeypatch,
):
    cfg, source = _source(tmp_path)
    monkeypatch.setattr(diagnostic_cohort, "git_dirty", lambda path: True)
    monkeypatch.setattr(
        diagnostic_cohort, "git_commit", lambda path: "draft-commit"
    )
    draft_path = tmp_path / "draft.json"
    draft = plan_diagnostic_cohort(
        source_dir=source,
        output_path=draft_path,
        seed=13,
        per_stratum=1,
        stratum_fields=["task_id"],
    )
    assert draft["frozen"] is False
    outcome = source / "workers/rank_0/episode_results.jsonl"
    outcome.parent.mkdir(parents=True)
    outcome.write_text('{"success": true}\n', encoding="utf-8")

    monkeypatch.setattr(diagnostic_cohort, "git_dirty", lambda path: False)
    monkeypatch.setattr(
        diagnostic_cohort, "git_commit", lambda path: "ratify-commit"
    )
    ratified_path = tmp_path / "ratified.json"
    ratified = ratify_diagnostic_cohort(
        draft_manifest_path=draft_path,
        source_dir=source,
        output_path=ratified_path,
    )
    assert ratified["frozen"] is True
    assert ratified["status"] == "ratified_before_diagnostic_outcomes"
    assert ratified["selected_job_ids"] == draft["selected_job_ids"]

    payload = json.loads(ratified_path.read_text(encoding="utf-8"))
    assert payload["freeze_scope"] == "phase2_future_consistency_diagnostics"
    assert payload["ratification"]["exact_selected_job_ids_preserved"] is True
    assert (
        payload["ratification"][
            "source_outcome_contents_read_for_ratification"
        ]
        is False
    )
    assert payload["ratification_provenance"]["git_dirty"] is False

    formal_cfg = replace(
        cfg,
        experiment=replace(
            cfg.experiment,
            name="ratified_diagnostic",
            output_dir=tmp_path / "ratified_diagnostic",
        ),
        diagnostics=replace(
            cfg.diagnostics,
            enabled=True,
            mode="unconditional_future",
            source_experiment_id="source",
            source_output_dir=source,
            cohort_manifest_path=ratified_path,
            require_frozen_cohort=True,
        ),
    )
    assert [job.job_id for job in load_source_jobs(formal_cfg)] == ratified[
        "selected_job_ids"
    ]


def test_ratification_refuses_dirty_tree_and_post_outcome_draft(
    tmp_path,
    monkeypatch,
):
    _, source = _source(tmp_path)
    draft_path = tmp_path / "draft.json"
    plan_diagnostic_cohort(
        source_dir=source,
        output_path=draft_path,
        seed=5,
        per_stratum=1,
        stratum_fields=["task_id"],
    )
    outcome = source / "workers/rank_0/episode_results.jsonl"
    outcome.parent.mkdir(parents=True)
    outcome.write_text('{"success": true}\n', encoding="utf-8")

    monkeypatch.setattr(diagnostic_cohort, "git_dirty", lambda path: True)
    with pytest.raises(RuntimeError, match="clean project tree"):
        ratify_diagnostic_cohort(
            draft_manifest_path=draft_path,
            source_dir=source,
            output_path=tmp_path / "dirty.json",
        )

    monkeypatch.setattr(diagnostic_cohort, "git_dirty", lambda path: False)
    post_outcome_draft = json.loads(draft_path.read_text(encoding="utf-8"))
    post_outcome_draft["source"]["outcome_files_present_at_selection"] = [
        str(outcome)
    ]
    bad_draft = tmp_path / "bad_draft.json"
    bad_draft.write_text(
        json.dumps(post_outcome_draft),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="before source outcomes"):
        ratify_diagnostic_cohort(
            draft_manifest_path=bad_draft,
            source_dir=source,
            output_path=tmp_path / "bad_ratified.json",
        )


def test_cohort_cli_plans_and_validates(tmp_path, capsys):
    _, source = _source(tmp_path)
    cohort_path = tmp_path / "cli_cohort.json"
    assert cli.main(
        [
            "plan-diagnostic-cohort",
            "--source-dir",
            str(source),
            "--output",
            str(cohort_path),
            "--seed",
            "17",
            "--per-stratum",
            "1",
            "--stratum-field",
            "task_id",
            "--anchor-episode-index",
            "0",
        ]
    ) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["selected_jobs"] == 2
    assert "selected_job_ids" not in planned
    assert cli.main(
        [
            "validate-diagnostic-cohort",
            "--manifest",
            str(cohort_path),
            "--source-dir",
            str(source),
        ]
    ) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["cohort_id"] == planned["cohort_id"]


def test_cohort_cli_ratifies_exact_draft(tmp_path, capsys, monkeypatch):
    _, source = _source(tmp_path)
    draft_path = tmp_path / "cli_draft.json"
    plan_diagnostic_cohort(
        source_dir=source,
        output_path=draft_path,
        seed=19,
        per_stratum=1,
        stratum_fields=["task_id"],
    )
    outcome = source / "workers/rank_0/episode_results.jsonl"
    outcome.parent.mkdir(parents=True)
    outcome.write_text('{"success": false}\n', encoding="utf-8")
    monkeypatch.setattr(diagnostic_cohort, "git_dirty", lambda path: False)
    monkeypatch.setattr(
        diagnostic_cohort, "git_commit", lambda path: "ratify-commit"
    )

    ratified_path = tmp_path / "cli_ratified.json"
    assert cli.main(
        [
            "ratify-diagnostic-cohort",
            "--draft-manifest",
            str(draft_path),
            "--source-dir",
            str(source),
            "--output",
            str(ratified_path),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frozen"] is True
    assert result["status"] == "ratified_before_diagnostic_outcomes"
