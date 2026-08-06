"""Deterministic, outcome-blind 4-suite x 2-task Phase 6 cohort selection."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastwam_ood_eval.thought6.config import SUITES, Thought6Config
from fastwam_ood_eval.thought6.schemas import Thought6Error, file_sha256, object_sha256


@dataclass(frozen=True)
class CanonicalTask:
    suite: str
    task_id: int
    task_name: str
    problem_folder: str
    bddl_file: str
    init_states_file: str

    @property
    def canonical_id(self) -> str:
        return f"{self.suite}/{self.task_id}"


def canonical_task_catalog(*, config_dir: Path) -> tuple[CanonicalTask, ...]:
    # Other test/evaluation paths may already have imported LIBERO-plus into
    # this interpreter. LIBERO explicitly forbids switching checkout in one
    # process, so isolate the canonical standard-LIBERO catalog read in a
    # short-lived subprocess. This also keeps Audit independent of import order.
    code = r'''
import json, sys
from pathlib import Path
from fastwam_ood_eval.envs.libero_adapter import configure_libero_package
configure_libero_package(Path("third_party/LIBERO"), Path(sys.argv[1]))
from libero.libero import benchmark
rows=[]
for suite in ("libero_spatial","libero_object","libero_goal","libero_10"):
    obj=benchmark.get_benchmark_dict()[suite]()
    for task_id in range(int(obj.n_tasks)):
        task=obj.get_task(task_id)
        rows.append({"suite":suite,"task_id":task_id,"task_name":str(task.language),"problem_folder":str(task.problem_folder),"bddl_file":str(task.bddl_file),"init_states_file":str(task.init_states_file)})
print("THOUGHT6_TASKS="+json.dumps(rows, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(config_dir)],
        check=True,
        text=True,
        capture_output=True,
    )
    marker = next(
        (line for line in completed.stdout.splitlines() if line.startswith("THOUGHT6_TASKS=")),
        None,
    )
    if marker is None:
        raise Thought6Error("canonical LIBERO task subprocess returned no catalog")
    return tuple(CanonicalTask(**row) for row in json.loads(marker.split("=", 1)[1]))


def historical_exclusions(cfg: Thought6Config) -> dict[str, Any]:
    thought3_path = Path("outputs/thought3/cache/phase_d_libero_goal_task0_v1/split_manifest.json")
    thought4_path = Path("outputs/thought4/phase4_geometry_action_diagnosis_v6/cohort_manifest.json")
    thought5_path = Path("outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/cohort_manifest.json")
    for path in (thought3_path, thought4_path, thought5_path):
        if not path.is_file():
            raise Thought6Error(f"historical cohort manifest is absent: {path}")
    thought3 = json.loads(thought3_path.read_text(encoding="utf-8"))
    thought4 = json.loads(thought4_path.read_text(encoding="utf-8"))
    thought5 = json.loads(thought5_path.read_text(encoding="utf-8"))
    if set(thought3.get("strata", {})) != {"libero_goal/task_0"}:
        raise Thought6Error("Thought3 historical task provenance is unexpected")
    thought4_tasks = {
        int(row["identity"]["task_id"]) for row in thought4.get("rows", [])
    }
    thought5_tasks = {int(row["task_index"]) for row in thought5.get("rows", [])}
    if thought4_tasks != {0} or thought5_tasks != {0}:
        raise Thought6Error("Thought4/5 historical task provenance is unexpected")
    episode_ids = {
        "thought3_development": sorted(thought3["development_episode_ids"]),
        "thought4_all": sorted(
            {str(row["identity"]["episode_id"]) for row in thought4["rows"]}
        ),
        "thought5_all": sorted({f"episode_{int(row['episode_index']):06d}" for row in thought5["rows"]}),
    }
    return {
        "excluded_tasks": {
            "libero_goal/0": [
                "Thought3 train/development Adapter cohort",
                "Thought4 formal geometry cohort",
                "Thought5 train/development/pilot-test cohort",
            ]
        },
        "excluded_episode_ids": episode_ids,
        "source_manifests": {
            str(thought3_path): file_sha256(thought3_path),
            str(thought4_path): file_sha256(thought4_path),
            str(thought5_path): file_sha256(thought5_path),
        },
    }


def _classification_candidates(
    classification: Mapping[str, Any], task: CanonicalTask
) -> list[dict[str, Any]]:
    base = Path(task.bddl_file).stem
    rows = []
    for row in classification[task.suite]:
        if row.get("category") != "Camera Viewpoints":
            continue
        name = str(row.get("name", ""))
        if name.startswith(base + "_"):
            rows.append(
                {
                    "classification_id": int(row["id"]),
                    "name": name,
                    "difficulty_level": row.get("difficulty_level"),
                }
            )
    return sorted(rows, key=lambda row: (row["classification_id"], row["name"]))


def _dataset_inventory(root: Path, selected: Iterable[CanonicalTask]) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "available": False,
            "root": str(root),
            "reason": "dataset_root_missing",
            "selected_task_episodes": {},
        }
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not tasks_path.is_file() or not episodes_path.is_file():
        return {
            "available": False,
            "root": str(root),
            "reason": "lerobot_metadata_missing",
            "selected_task_episodes": {},
        }
    task_rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line]
    episode_rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line]
    task_index_by_name = {
        str(row.get("task")): int(row.get("task_index", index))
        for index, row in enumerate(task_rows)
    }
    episode_task_index: dict[int, int] = {}
    for row in episode_rows:
        names = row.get("tasks")
        if not isinstance(names, list) or len(names) != 1:
            raise Thought6Error("each LeRobot episode must identify exactly one task")
        name = str(names[0])
        if name not in task_index_by_name:
            raise Thought6Error(f"episode references unknown task: {name}")
        episode_task_index[int(row["episode_index"])] = task_index_by_name[name]
    selected_episodes: dict[str, list[str]] = {}
    for task in selected:
        dataset_task_index = task_index_by_name.get(task.task_name)
        if dataset_task_index is None:
            selected_episodes[task.canonical_id] = []
            continue
        selected_episodes[task.canonical_id] = sorted(
            f"episode_{int(row['episode_index']):06d}"
            for row in episode_rows
            if episode_task_index[int(row["episode_index"])] == dataset_task_index
        )
    return {
        "available": all(len(values) >= 4 for values in selected_episodes.values()),
        "root": str(root),
        "metadata_sha256": {
            "tasks.jsonl": file_sha256(tasks_path),
            "episodes.jsonl": file_sha256(episodes_path),
        },
        "selected_task_episodes": selected_episodes,
    }


def select_phase6_tasks(cfg: Thought6Config) -> dict[str, Any]:
    # LIBERO insists on materializing a package-path config. Keep that
    # implementation detail in /tmp so CPU dry-run never mutates experiment
    # outputs and the task-selection artifact remains the only frozen record.
    with tempfile.TemporaryDirectory(prefix="thought6_libero_catalog_") as temporary:
        catalog = canonical_task_catalog(config_dir=Path(temporary))
    exclusions = historical_exclusions(cfg)
    excluded = set(exclusions["excluded_tasks"])
    classification = json.loads(cfg.classification_path.read_text(encoding="utf-8"))
    if set(classification) != set(SUITES):
        raise Thought6Error("LIBERO-Plus classification catalog suite set differs")
    candidates: list[dict[str, Any]] = []
    selected: list[CanonicalTask] = []
    for suite in SUITES:
        suite_rows = sorted(
            (task for task in catalog if task.suite == suite),
            key=lambda task: task.task_id,
        )
        eligible = [task for task in suite_rows if task.canonical_id not in excluded]
        if len(eligible) < cfg.tasks_per_suite:
            raise Thought6Error(f"{suite} has fewer than two unused tasks")
        chosen = eligible[: cfg.tasks_per_suite]
        selected.extend(chosen)
        for task in suite_rows:
            camera = _classification_candidates(classification, task)
            candidates.append(
                {
                    **asdict(task),
                    "canonical_id": task.canonical_id,
                    "excluded": task.canonical_id in excluded,
                    "exclusion_reasons": exclusions["excluded_tasks"].get(task.canonical_id, []),
                    "selected": task in chosen,
                    "camera_variant_count": len(camera),
                    "camera_variants": camera,
                    "selected_camera_variant": camera[0] if task in chosen and camera else None,
                }
            )
            if task in chosen and not camera:
                raise Thought6Error(f"selected task has no reliable Camera variant: {task.canonical_id}")
    inventories = {
        suite: _dataset_inventory(
            cfg.suite_dataset_roots[suite],
            [task for task in selected if task.suite == suite],
        )
        for suite in SUITES
    }
    selected_rows = [row for row in candidates if row["selected"]]
    for row in selected_rows:
        episodes = inventories[row["suite"]]["selected_task_episodes"].get(row["canonical_id"], [])
        row["available_episode_ids"] = episodes
        row["phase6b_episode_ids"] = episodes[: cfg.utility_episodes_per_task]
        row["phase6b_episode_count_ready"] = len(row["phase6b_episode_ids"]) >= cfg.utility_episodes_per_task
    payload = {
        "schema_version": "thought6.task_selection_manifest.v1",
        "status": (
            "ready" if all(inventory["available"] for inventory in inventories.values()) else "blocked_missing_suite_datasets"
        ),
        "selection_rule": "exclude_Thought3_4_5_then_canonical_task_id_ascending_take_first_two",
        "selection_is_outcome_blind": True,
        "outcome_fields_read": False,
        "suite_order": list(SUITES),
        "tasks_per_suite": cfg.tasks_per_suite,
        "candidates": candidates,
        "selected_tasks": selected_rows,
        "historical_exclusions": exclusions,
        "dataset_inventories": inventories,
        "classification_path": str(cfg.classification_path),
        "classification_sha256": file_sha256(cfg.classification_path),
        "selection_code_path": "src/fastwam_ood_eval/thought6/task_selection.py",
        "selection_code_sha256": file_sha256(__file__),
    }
    payload["task_hash"] = object_sha256(selected_rows)
    payload["manifest_sha256"] = object_sha256(payload)
    return payload


def assert_phase6b_data_ready(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "ready" or len(manifest.get("selected_tasks", [])) != 8:
        missing = [
            suite
            for suite, inventory in manifest.get("dataset_inventories", {}).items()
            if not inventory.get("available")
        ]
        raise Thought6Error(
            "Phase 6B is fail-closed until all four suite demonstration roots are ready; "
            f"missing={missing}"
        )
