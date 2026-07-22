"""Adapter for official pre-generated LIBERO-Plus task variants."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path
from typing import Any

from fastwam_ood_eval.envs.libero_adapter import LiberoAdapter


def _resolve_libero_plus_init_state(problem_folder: str, init_states_file: str) -> tuple[Path, bool]:
    """Mirror the pinned LIBERO-Plus init-state routing without loading pickle.

    Most visual perturbations reuse the base task's state. Object additions and
    level variants have dedicated files under ``libero_newobj``. The sequential
    overrides below intentionally match upstream: a base task name can itself
    contain ``_table_`` (for example ``from_table_center``), so later ``_tb_``,
    ``_light_`` and new-object rules must take precedence.
    """

    relative_path: Path | None = None
    reshape_single = False
    suffix = Path(init_states_file).suffix

    if "_language_" in init_states_file:
        relative_path = Path(problem_folder) / (init_states_file.split("_language_", 1)[0] + suffix)
    elif "_view_" in init_states_file:
        relative_path = Path(problem_folder) / (init_states_file.split("_view_", 1)[0] + suffix)
    else:
        if "_table_" in init_states_file:
            relative_path = Path(problem_folder) / re.sub(r"_table_\d+", "", init_states_file)
        if "_tb_" in init_states_file:
            relative_path = Path(problem_folder) / re.sub(r"_tb_\d+", "", init_states_file)
        if "_light_" in init_states_file:
            relative_path = Path(problem_folder) / (init_states_file.split("_light_", 1)[0] + suffix)
        if "_add_" in init_states_file or "_level" in init_states_file:
            relative_path = Path("libero_newobj") / problem_folder / init_states_file
            reshape_single = True

    if relative_path is None:
        relative_path = Path(problem_folder) / init_states_file
    return relative_path, reshape_single


class LiberoPlusAdapter(LiberoAdapter):
    def __init__(
        self,
        image_size: tuple[int, int],
        root: Path = Path("third_party/LIBERO-plus"),
        config_dir: Path = Path("outputs/runtime/libero_plus"),
    ) -> None:
        super().__init__(image_size=image_size, root=root, config_dir=config_dir)

    def _make_suite(self, suite_name: str) -> Any:
        # The pinned Plus checkout prints every task ID (up to 2,591 integers)
        # whenever a benchmark object is constructed. Keep evaluation logs
        # usable while leaving upstream code untouched.
        with contextlib.redirect_stdout(io.StringIO()):
            return super()._make_suite(suite_name)

    def _resolve_init_state(self, task: Any) -> tuple[Path, bool]:
        return _resolve_libero_plus_init_state(task.problem_folder, task.init_states_file)
