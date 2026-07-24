from __future__ import annotations

import csv
import hashlib
import json
import stat
from pathlib import Path

import pytest

from fastwam_ood_eval import cli
from fastwam_ood_eval.diagnostics.blind_review import (
    BLIND_ANNOTATION_FIELDS,
    PRIVATE_KEY_NAME,
    PUBLIC_MANIFEST_NAME,
    prepare_blind_review,
    validate_blind_review_packet,
)
from fastwam_ood_eval.diagnostics.blind_review_analysis import (
    analyze_blind_review,
    validate_blind_review_analysis,
)


def _write_diagnostic_root(
    root: Path,
    *,
    experiment_id: str,
    fingerprint: str,
    diagnostic_id: str,
    job_id: str,
    condition: str,
    success: bool,
) -> None:
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "kind": "future_shadow_diagnostics",
        "experiment_id": experiment_id,
        "protocol_fingerprint": fingerprint,
        "config": {
            "checkpoint": {"path": "mock.pt", "model_name": "mock"},
            "benchmark": {
                "backend": "mock",
                "suite": "mock_suite",
                "control_horizon": 2,
                "image_size": [8, 8],
            },
            "diagnostics": {
                "mode": "unconditional_future",
                "num_video_frames": 9,
                "num_inference_steps": 2,
                "static_motion_threshold": 1.0,
                "motion_epsilon": 1e-8,
                "probe_strategy": "first",
                "max_probes_per_episode": 1,
                "explicit_replan_indices": [],
            },
        },
        "provenance": {
            "git_commit": "project",
            "git_dirty": False,
            "fastwam_commit": "fastwam",
            "fastwam_dirty": False,
            "libero_commit": "libero",
            "libero_dirty": False,
            "libero_plus_commit": "plus",
            "libero_plus_dirty": False,
        },
    }
    (root / "diagnostic_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    media_root = root / "workers/rank_0"
    artifact_paths = {}
    for field, relative in (
        ("current_frame_path", f"current_frames/{job_id}.png"),
        ("predicted_video_path", f"predicted_futures/{job_id}.mp4"),
        ("actual_video_path", f"actual_futures/{job_id}.mp4"),
        ("side_by_side_video_path", f"side_by_side/{job_id}.mp4"),
    ):
        path = media_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{field}:{diagnostic_id}".encode())
        artifact_paths[field] = f"workers/rank_0/{relative}"
    row = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "diagnostic_id": diagnostic_id,
        "probe_id": diagnostic_id,
        "job_id": job_id,
        "task_id": 3,
        "task_name": "place_the_black_bowl_on_the_plate",
        "condition": condition,
        "episode_success": success,
        "success": success,
        "termination_reason": "success" if success else "max_steps",
        "perturbation_category": (
            "camera_viewpoints" if condition == "ood" else None
        ),
        "perturbation_level": "easy" if condition == "ood" else None,
        "episode_seed": 123,
        "status": "completed",
        "error": None,
        "metrics": {"future_latent_l1": 0.5},
        "predicted_actions": [[1.0] * 7],
        "executed_actions": [[1.0] * 7],
        "artifact_paths": artifact_paths,
        "extra": {"protocol_fingerprint": fingerprint},
    }
    diagnostic_path = media_root / "diagnostics.jsonl"
    diagnostic_path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> list[Path]:
    clean = tmp_path / "source_that_says_clean"
    ood = tmp_path / "source_that_says_ood"
    _write_diagnostic_root(
        clean,
        experiment_id="diagnostic_clean_secret",
        fingerprint="clean-fingerprint",
        diagnostic_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        job_id="111111111111111111111111",
        condition="clean",
        success=True,
    )
    _write_diagnostic_root(
        ood,
        experiment_id="diagnostic_ood_secret",
        fingerprint="ood-fingerprint",
        diagnostic_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        job_id="222222222222222222222222",
        condition="ood",
        success=False,
    )
    return [clean, ood]


def _write_reviewer_annotations(
    path: Path,
    *,
    packet: Path,
    reviewer: str,
    overrides: dict[str, dict[str, str]] | None = None,
    as_json: bool = False,
) -> None:
    public = json.loads((packet / PUBLIC_MANIFEST_NAME).read_text())
    rows = []
    for case_id in public["case_order"]:
        row = {
            "packet_id": public["packet_id"],
            "case_id": case_id,
            "reviewer": reviewer,
            "review_round": "blind",
            "video_validity": "valid",
            "future_goal_progress": "correct",
            "future_physical_plausibility": "plausible",
            "future_actual_agreement": "aligned",
            "action_execution_quality": "realized",
            "confidence": "high",
            "notes": "",
        }
        row.update((overrides or {}).get(case_id, {}))
        rows.append(row)
    if as_json:
        path.write_text(
            json.dumps(
                {
                    "packet_id": public["packet_id"],
                    "annotations": rows,
                }
            ),
            encoding="utf-8",
        )
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=BLIND_ANNOTATION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_blind_packet_separates_public_media_from_private_labels(tmp_path):
    inputs = _inputs(tmp_path)
    packet = tmp_path / "blind_packet"
    key = tmp_path / "private_key"
    result = prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=20260723,
    )
    assert result["cases"] == 2
    assert result["media_files"] == 8
    assert result["private_key_verified"] is True
    assert result["sensitive_public_keys"] == 0

    public_text = "\n".join(
        (packet / name).read_text(encoding="utf-8")
        for name in (
            PUBLIC_MANIFEST_NAME,
            "annotations.csv",
            "index.html",
        )
    )
    for secret in (
        "aaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbb",
        "111111111111111111111111",
        "222222222222222222222222",
        "diagnostic_clean_secret",
        "diagnostic_ood_secret",
        "source_that_says_clean",
        "source_that_says_ood",
    ):
        assert secret not in public_text
    assert "primary_failure_hypothesis" not in public_text

    private = json.loads((key / PRIVATE_KEY_NAME).read_text())
    assert private["selection"]["outcome_fields_used"] is False
    assert private["selection"]["condition_fields_used"] is False
    assert private["selection"]["metric_fields_used"] is False
    assert {
        case["source_record"]["condition"] for case in private["cases"]
    } == {"clean", "ood"}
    assert {
        case["source_record"]["episode_success"]
        for case in private["cases"]
    } == {True, False}
    assert stat.S_IMODE((key / PRIVATE_KEY_NAME).stat().st_mode) == 0o600

    second_packet = tmp_path / "blind_packet_repeat"
    second_key = tmp_path / "private_key_repeat"
    repeated = prepare_blind_review(
        packet_dir=second_packet,
        key_dir=second_key,
        input_dirs=inputs,
        seed=20260723,
    )
    assert repeated["packet_id"] == result["packet_id"]
    assert (
        second_packet / PUBLIC_MANIFEST_NAME
    ).read_bytes() == (packet / PUBLIC_MANIFEST_NAME).read_bytes()


def test_blind_packet_validation_detects_tampering_and_output_overlap(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        prepare_blind_review(
            packet_dir=inputs[0] / "packet",
            key_dir=tmp_path / "key",
            input_dirs=inputs,
            seed=1,
        )

    packet = tmp_path / "packet"
    key = tmp_path / "key"
    prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=1,
        max_cases=1,
    )
    public = json.loads((packet / PUBLIC_MANIFEST_NAME).read_text())
    media_path = (
        packet
        / public["cases"][0]["media"]["current_frame"]["path"]
    )
    media_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        validate_blind_review_packet(packet, key)


def test_blind_packet_validation_rejects_annotation_protocol_tampering(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    packet = tmp_path / "packet"
    key = tmp_path / "key"
    prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=2,
    )
    manifest_path = packet / PUBLIC_MANIFEST_NAME
    public = json.loads(manifest_path.read_text())
    public["annotation_options"]["confidence"].append("guessing")
    manifest_path.write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(RuntimeError, match="options differ"):
        validate_blind_review_packet(packet)


def test_blind_review_cli_prepares_and_validates(tmp_path, capsys):
    inputs = _inputs(tmp_path)
    packet = tmp_path / "packet"
    key = tmp_path / "key"
    arguments = [
        "prepare-blind-review",
        "--packet-dir",
        str(packet),
        "--key-dir",
        str(key),
        "--seed",
        "42",
    ]
    for source in inputs:
        arguments.extend(["--input-dir", str(source)])
    assert cli.main(arguments) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["cases"] == 2
    assert cli.main(
        [
            "validate-blind-review",
            "--packet-dir",
            str(packet),
            "--key-dir",
            str(key),
        ]
    ) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["packet_id"] == prepared["packet_id"]


def test_blind_packet_can_cap_probes_per_source_job(tmp_path):
    inputs = _inputs(tmp_path)
    baseline_packet = tmp_path / "baseline_packet"
    baseline_key = tmp_path / "baseline_key"
    prepare_blind_review(
        packet_dir=baseline_packet,
        key_dir=baseline_key,
        input_dirs=inputs,
        seed=5,
        max_cases=1,
        max_cases_per_job=1,
    )
    baseline_private = json.loads(
        (baseline_key / PRIVATE_KEY_NAME).read_text()
    )
    baseline_selected_job = baseline_private["cases"][0][
        "source_record"
    ]["job_id"]

    diagnostic_path = (
        inputs[0] / "workers/rank_0/diagnostics.jsonl"
    )
    first = json.loads(diagnostic_path.read_text())
    second = {
        **first,
        "diagnostic_id": "cccccccccccccccccccccccc",
        "probe_id": "cccccccccccccccccccccccc",
    }
    diagnostic_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    packet = tmp_path / "packet"
    key = tmp_path / "key"
    result = prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=5,
        max_cases_per_job=1,
    )
    assert result["cases"] == 2
    private = json.loads((key / PRIVATE_KEY_NAME).read_text())
    assert private["selection"]["reviewable_rows"] == 3
    assert private["selection"]["eligible_rows_after_job_cap"] == 2
    assert private["selection"]["max_cases_per_job"] == 1
    assert len(
        {
            case["source_record"]["job_id"]
            for case in private["cases"]
        }
    ) == 2

    capped_packet = tmp_path / "capped_packet"
    capped_key = tmp_path / "capped_key"
    prepare_blind_review(
        packet_dir=capped_packet,
        key_dir=capped_key,
        input_dirs=inputs,
        seed=5,
        max_cases=1,
        max_cases_per_job=1,
    )
    capped_private = json.loads(
        (capped_key / PRIVATE_KEY_NAME).read_text()
    )
    assert capped_private["cases"][0]["source_record"][
        "job_id"
    ] == baseline_selected_job


def test_blind_annotation_analysis_validates_and_reports_kappa(
    tmp_path,
    capsys,
):
    inputs = _inputs(tmp_path)
    packet = tmp_path / "packet"
    key = tmp_path / "key"
    prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=8,
    )
    public = json.loads((packet / PUBLIC_MANIFEST_NAME).read_text())
    first_case, second_case = public["case_order"]
    reviewer_a = tmp_path / "reviewer_a.csv"
    reviewer_b = tmp_path / "reviewer_b.json"
    _write_reviewer_annotations(
        reviewer_a,
        packet=packet,
        reviewer="reviewer-a",
        overrides={
            second_case: {
                "future_goal_progress": "uncertain",
                "future_actual_agreement": "conflict",
            }
        },
    )
    _write_reviewer_annotations(
        reviewer_b,
        packet=packet,
        reviewer="reviewer-b",
        overrides={
            second_case: {
                "future_goal_progress": "partial",
            }
        },
        as_json=True,
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (reviewer_a, reviewer_b)
    }
    output = tmp_path / "agreement"
    result = analyze_blind_review(
        packet_dir=packet,
        annotation_paths=[reviewer_a, reviewer_b],
        output_dir=output,
    )
    assert result["cases"] == 2
    assert result["reviewers"] == 2
    assert result["reviewer_pairs"] == 1
    assert result["outputs_verified"] == 5
    assert result["private_key_read"] is False
    assert result["source_files_rewritten"] is False
    assert before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (reviewer_a, reviewer_b)
    }

    with (output / "pairwise_agreement.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        agreement = {
            row["field"]: row for row in csv.DictReader(handle)
        }
    actual = agreement["future_actual_agreement"]
    assert int(actual["nonmissing_pairs"]) == 2
    assert float(actual["agreement_nonmissing"]) == pytest.approx(0.5)
    assert float(actual["cohen_kappa_nonmissing"]) == pytest.approx(0.0)
    goal = agreement["future_goal_progress"]
    assert int(goal["uncertain_pairs"]) == 1
    assert int(goal["decisive_pairs"]) == 1
    assert float(goal["agreement_decisive"]) == pytest.approx(1.0)
    physical = agreement["future_physical_plausibility"]
    assert physical["cohen_kappa_nonmissing"] == ""
    assert (
        physical["cohen_kappa_nonmissing_status"]
        == "degenerate_marginals"
    )
    assert "undefined" in (
        output / "agreement_report.md"
    ).read_text(encoding="utf-8")
    assert validate_blind_review_analysis(output)["analysis_id"] == result[
        "analysis_id"
    ]

    cli_output = tmp_path / "agreement_cli"
    assert cli.main(
        [
            "analyze-blind-review",
            "--packet-dir",
            str(packet),
            "--annotation",
            str(reviewer_a),
            "--annotation",
            str(reviewer_b),
            "--output-dir",
            str(cli_output),
        ]
    ) == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["cases"] == 2
    assert cli.main(
        [
            "validate-blind-review-analysis",
            "--analysis-dir",
            str(cli_output),
        ]
    ) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["analysis_id"] == cli_result["analysis_id"]


def test_blind_annotation_analysis_rejects_invalid_labels_and_reviewers(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    packet = tmp_path / "packet"
    key = tmp_path / "key"
    prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=9,
    )
    case_id = json.loads(
        (packet / PUBLIC_MANIFEST_NAME).read_text()
    )["case_order"][0]
    valid = tmp_path / "valid.csv"
    invalid = tmp_path / "invalid.csv"
    _write_reviewer_annotations(
        valid,
        packet=packet,
        reviewer="reviewer-a",
    )
    _write_reviewer_annotations(
        invalid,
        packet=packet,
        reviewer="reviewer-b",
        overrides={
            case_id: {"future_goal_progress": "looks_good_to_me"}
        },
    )
    output = tmp_path / "invalid_output"
    with pytest.raises(RuntimeError, match="Invalid future_goal_progress"):
        analyze_blind_review(
            packet_dir=packet,
            annotation_paths=[valid, invalid],
            output_dir=output,
        )
    assert not output.exists()

    same_reviewer = tmp_path / "same_reviewer.json"
    _write_reviewer_annotations(
        same_reviewer,
        packet=packet,
        reviewer="reviewer-a",
        as_json=True,
    )
    with pytest.raises(RuntimeError, match="distinct reviewer IDs"):
        analyze_blind_review(
            packet_dir=packet,
            annotation_paths=[valid, same_reviewer],
            output_dir=tmp_path / "same_reviewer_output",
        )


def test_blind_annotation_analysis_detects_derived_output_tampering(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    packet = tmp_path / "packet"
    key = tmp_path / "key"
    prepare_blind_review(
        packet_dir=packet,
        key_dir=key,
        input_dirs=inputs,
        seed=10,
    )
    reviewer_a = tmp_path / "reviewer_a.csv"
    reviewer_b = tmp_path / "reviewer_b.csv"
    _write_reviewer_annotations(
        reviewer_a,
        packet=packet,
        reviewer="reviewer-a",
    )
    _write_reviewer_annotations(
        reviewer_b,
        packet=packet,
        reviewer="reviewer-b",
    )
    output = tmp_path / "agreement"
    analyze_blind_review(
        packet_dir=packet,
        annotation_paths=[reviewer_a, reviewer_b],
        output_dir=output,
    )
    (output / "pairwise_agreement.csv").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="output hash changed"):
        validate_blind_review_analysis(output)
