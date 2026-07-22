from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastwam_ood_eval.envs.libero_plus_adapter import _resolve_libero_plus_init_state


@pytest.mark.parametrize(
    ("filename", "expected", "reshape_single"),
    [
        (
            "pick_up_the_bowl_view_0_0_100_2_352_initstate_0.pruned_init",
            "libero_spatial/pick_up_the_bowl.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_language_3.pruned_init",
            "libero_spatial/pick_up_the_bowl.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_table_4.pruned_init",
            "libero_spatial/pick_up_the_bowl.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_tb_7.pruned_init",
            "libero_spatial/pick_up_the_bowl.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_light_45.pruned_init",
            "libero_spatial/pick_up_the_bowl.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_add_2.pruned_init",
            "libero_newobj/libero_spatial/pick_up_the_bowl_add_2.pruned_init",
            True,
        ),
        (
            "pick_up_the_bowl_level3_sample2.pruned_init",
            "libero_newobj/libero_spatial/pick_up_the_bowl_level3_sample2.pruned_init",
            True,
        ),
    ],
)
def test_resolves_official_libero_plus_init_state_rules(filename, expected, reshape_single):
    path, reshape = _resolve_libero_plus_init_state("libero_spatial", filename)

    assert path == Path(expected)
    assert reshape is reshape_single


@pytest.mark.parametrize(
    ("filename", "expected", "reshape_single"),
    [
        (
            "pick_up_the_bowl_from_table_center_light_4.pruned_init",
            "libero_spatial/pick_up_the_bowl_from_table_center.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_from_table_center_tb_4.pruned_init",
            "libero_spatial/pick_up_the_bowl_from_table_center.pruned_init",
            False,
        ),
        (
            "pick_up_the_bowl_from_table_center_add_4.pruned_init",
            "libero_newobj/libero_spatial/pick_up_the_bowl_from_table_center_add_4.pruned_init",
            True,
        ),
    ],
)
def test_later_plus_rules_override_table_substring_in_base_task(filename, expected, reshape_single):
    path, reshape = _resolve_libero_plus_init_state("libero_spatial", filename)

    assert path == Path(expected)
    assert reshape is reshape_single


def test_all_classified_plus_tasks_resolve_to_existing_init_state_files():
    checkout = Path("third_party/LIBERO-plus/libero/libero")
    classification = checkout / "benchmark" / "task_classification.json"
    if not classification.is_file():
        pytest.skip("LIBERO-Plus checkout is unavailable")

    rows_by_suite = json.loads(classification.read_text(encoding="utf-8"))
    missing: list[str] = []
    total = 0
    for suite, rows in rows_by_suite.items():
        for row in rows:
            relative_path, _ = _resolve_libero_plus_init_state(
                suite,
                f"{row['name']}.pruned_init",
            )
            total += 1
            if not (checkout / "init_files" / relative_path).is_file():
                missing.append(f"{suite}:{row['name']} -> {relative_path}")

    assert total == 10_030
    assert missing == []
