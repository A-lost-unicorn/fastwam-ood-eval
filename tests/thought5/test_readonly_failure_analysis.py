from __future__ import annotations

import math

import pytest
import torch

from fastwam_ood_eval.thought5.readonly_failure_analysis import (
    effective_flow_sigma,
    low_rank_delta_statistics,
    summarize_action_horizon,
)


def test_effective_flow_sigma_replays_bf16_scheduler_sampling() -> None:
    assert effective_flow_sigma(0) == pytest.approx(0.83203125)
    assert effective_flow_sigma(1) == pytest.approx(0.94140625)
    assert effective_flow_sigma(2117827456) == pytest.approx(0.9609375)


def test_low_rank_statistics_match_dense_delta() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    left_a = torch.randn((3, 5), generator=generator)
    left_b = torch.randn((4, 3), generator=generator)
    right_a = torch.randn((3, 5), generator=generator)
    right_b = torch.randn((4, 3), generator=generator)
    scale = 0.5
    result = low_rank_delta_statistics(
        left_a, left_b, right_a, right_b, scale=scale
    )
    left = scale * (left_b @ left_a)
    right = scale * (right_b @ right_a)
    expected_cosine = torch.nn.functional.cosine_similarity(
        left.reshape(1, -1), right.reshape(1, -1)
    ).item()
    assert result["left_frobenius"] == pytest.approx(
        torch.linalg.vector_norm(left).item(), rel=1e-6
    )
    assert result["right_frobenius"] == pytest.approx(
        torch.linalg.vector_norm(right).item(), rel=1e-6
    )
    assert result["difference_frobenius"] == pytest.approx(
        torch.linalg.vector_norm(left - right).item(), rel=1e-6
    )
    assert result["cosine"] == pytest.approx(expected_cosine, rel=1e-6)


def _technical_panel() -> dict[str, object]:
    rows = []
    for index in range(8):
        condition = "clean" if index < 4 else "camera"
        row = {"sample_id": f"sample-{index}:{condition}"}
        for multiplier, contrast in enumerate(
            ("correct_null", "correct_shuffle", "null_shuffle"), start=1
        ):
            row[contrast] = {
                "per_timestep_l2": [
                    multiplier * float(action_index + 1) for action_index in range(32)
                ]
            }
        rows.append(row)
    return {"status": "complete", "action_denoise_steps": 20, "rows": rows}


def test_action_horizon_labels_sensitivity_without_claiming_utility() -> None:
    result, csv_rows = summarize_action_horizon(
        {variant: _technical_panel() for variant in ("B1", "G3", "G4")}
    )
    g3 = result["variants"]["G3"]["correct_null"]["all"]
    assert g3["peak_action_index"] == 31
    assert g3["segment_mean_l2"]["unexecuted_tail_21_31"] > g3[
        "segment_mean_l2"
    ]["executed_prefix_0_9"]
    assert result["per_action_slot_utility"]["status"].startswith("unavailable")
    assert result["inference_denoising_localization"]["status"].startswith(
        "unavailable"
    )
    assert len(csv_rows) == 3 * 3 * 3 * 32
    assert math.isfinite(g3["tail_over_executed_prefix_ratio"])
