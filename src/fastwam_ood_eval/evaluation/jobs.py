"""Deterministic job planning and LIBERO-Plus variant selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from fastwam_ood_eval.config import ConfigError, EvalConfig
from fastwam_ood_eval.envs.perturbation_registry import get_perturbation
from fastwam_ood_eval.reproducibility import episode_seed, stable_int


@dataclass(frozen=True)
class EvaluationJob:
    experiment_id: str
    job_id: str
    suite: str
    task_id: int
    task_name: str
    upstream_task_id: int
    upstream_task_name: str
    episode_index: int
    episode_seed: int
    initial_state_index: int
    condition: str
    perturbation_category: str | None = None
    perturbation_level: str | None = None
    perturbation_parameters: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    policy_variant: str = "unspecified"
    test_time_future_imagination: bool = False
    comparison_group: str | None = None
    training_recipe_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationJob":
        payload = dict(data)
        payload.setdefault("policy_variant", "unspecified")
        payload.setdefault("test_time_future_imagination", False)
        payload.setdefault("comparison_group", None)
        payload.setdefault("training_recipe_id", None)
        payload.setdefault("initial_state_index", int(payload.get("episode_index", 0)))
        return cls(**payload)


def _job_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _load_suite(path: Path, suite_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigError(f"Suite configuration does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("suite") != suite_name or not isinstance(raw.get("tasks"), list):
        raise ConfigError(f"Invalid suite configuration: {path}")
    tasks: list[dict[str, Any]] = []
    for item in raw["tasks"]:
        if not isinstance(item, dict) or "id" not in item or "name" not in item:
            raise ConfigError(f"Every task in {path} needs id and name")
        tasks.append({"id": int(item["id"]), "name": str(item["name"])})
    return tasks


def _selected_tasks(cfg: EvalConfig) -> list[dict[str, Any]]:
    tasks = _load_suite(cfg.benchmark.suite_config, cfg.benchmark.suite)
    if cfg.benchmark.tasks is None:
        return tasks
    by_id = {task["id"]: task for task in tasks}
    unknown = [task_id for task_id in cfg.benchmark.tasks if task_id not in by_id]
    if unknown:
        raise ConfigError(f"Task IDs not found in {cfg.benchmark.suite_config}: {unknown}")
    return [by_id[task_id] for task_id in cfg.benchmark.tasks]


def _classification_path(cfg: EvalConfig) -> Path:
    configured = cfg.perturbation.parameters.get("classification_path")
    return Path(str(configured or "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"))


def _load_classification(cfg: EvalConfig) -> list[dict[str, Any]]:
    path = _classification_path(cfg)
    if not path.is_file():
        raise ConfigError(
            f"LIBERO-Plus task classification is missing: {path}. Run scripts/fetch_upstreams.sh first."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid classification JSON: {path}: {exc}") from exc
    rows = raw.get(cfg.benchmark.suite)
    if not isinstance(rows, list):
        raise ConfigError(f"Suite {cfg.benchmark.suite!r} not found in {path}")
    return rows


def _variant_candidates(
    rows: list[dict[str, Any]], base_name: str, category: str, level: str
) -> list[dict[str, Any]]:
    definition = get_perturbation(category)
    allowed = set(definition.levels[level])
    prefix = base_name + "_"
    candidates = [
        row
        for row in rows
        if row.get("category") == definition.upstream_category
        and row.get("difficulty_level") in allowed
        and str(row.get("name", "")).startswith(prefix)
    ]
    return sorted(candidates, key=lambda row: (int(row["difficulty_level"]), int(row["id"]), str(row["name"])))


def _make_job(
    cfg: EvalConfig,
    task: dict[str, Any],
    episode_index: int,
    *,
    condition: str,
    upstream_task_id: int,
    upstream_task_name: str,
    category: str | None = None,
    level: str | None = None,
    parameters: dict[str, Any] | None = None,
    skip_reason: str | None = None,
    initial_state_index: int | None = None,
) -> EvaluationJob:
    seed = episode_seed(cfg.experiment.seed, cfg.benchmark.suite, task["name"], episode_index)
    identity = {
        "experiment_id": cfg.experiment.name,
        "suite": cfg.benchmark.suite,
        "task_id": task["id"],
        "task_name": task["name"],
        "upstream_task_id": upstream_task_id,
        "upstream_task_name": upstream_task_name,
        "episode_index": episode_index,
        "episode_seed": seed,
        "initial_state_index": episode_index if initial_state_index is None else initial_state_index,
        "condition": condition,
        "perturbation_category": category,
        "perturbation_level": level,
        "perturbation_parameters": parameters or {},
        "policy_variant": cfg.policy.variant,
        "test_time_future_imagination": cfg.policy.test_time_future_imagination,
        "comparison_group": cfg.policy.comparison_group,
        "training_recipe_id": cfg.policy.training_recipe_id,
    }
    return EvaluationJob(job_id=_job_id(identity), skip_reason=skip_reason, **identity)


def plan_jobs(cfg: EvalConfig) -> list[EvaluationJob]:
    tasks = _selected_tasks(cfg)
    jobs: list[EvaluationJob] = []
    if not cfg.perturbation.enabled:
        for task in tasks:
            for episode_index in range(cfg.benchmark.episodes_per_task):
                jobs.append(
                    _make_job(
                        cfg,
                        task,
                        episode_index,
                        condition="clean",
                        upstream_task_id=task["id"],
                        upstream_task_name=task["name"],
                    )
                )
        return jobs

    if cfg.benchmark.backend == "mock":
        rows: list[dict[str, Any]] | None = None
    else:
        rows = _load_classification(cfg)
    for task in tasks:
        for category in cfg.perturbation.categories:
            definition = get_perturbation(category)
            for level in cfg.perturbation.levels:
                if rows is None:
                    candidates = [
                        {
                            "id": task["id"] + 1,
                            "name": f"{task['name']}_{category}_{level}",
                            "category": definition.upstream_category,
                            "difficulty_level": definition.levels[level][0],
                        }
                    ]
                else:
                    candidates = _variant_candidates(rows, task["name"], category, level)
                if cfg.perturbation.variant_selection == "all_once":
                    selected = list(enumerate(candidates))
                else:
                    offset = stable_int(cfg.experiment.seed, task["name"], category, level)
                    selected = [
                        (episode_index, candidates[(offset + episode_index) % len(candidates)])
                        for episode_index in range(cfg.benchmark.episodes_per_task)
                    ] if candidates else []

                for selection_index, row in selected:
                    # LIBERO-Plus treats every classification row as its own task. The
                    # official num_trials_per_task=1 protocol therefore always uses
                    # trial/init-state index zero for the exhaustive all_once plan.
                    episode_index = (
                        0
                        if cfg.perturbation.variant_selection == "all_once"
                        else selection_index
                    )
                    if candidates:
                        parameters = {
                            "official_category": row["category"],
                            "official_difficulty": row["difficulty_level"],
                            "variant_name": row["name"],
                            "classification_id": int(row["id"]),
                            "variant_selection": cfg.perturbation.variant_selection,
                            "selection_index": selection_index,
                            "candidate_count": len(candidates),
                            "selection_with_replacement": (
                                cfg.perturbation.variant_selection == "sample"
                                and cfg.benchmark.episodes_per_task > len(candidates)
                            ),
                        }
                        jobs.append(
                            _make_job(
                                cfg,
                                task,
                                episode_index,
                                condition="ood",
                                upstream_task_id=int(row["id"]) - 1,
                                upstream_task_name=str(row["name"]),
                                category=category,
                                level=level,
                                parameters=parameters,
                                initial_state_index=(
                                    0
                                    if cfg.perturbation.variant_selection == "all_once"
                                    else selection_index
                                ),
                            )
                        )
                if not candidates:
                    reason = (
                        f"No official LIBERO-Plus variant for task={task['name']}, "
                        f"category={definition.upstream_category}, level={level}"
                    )
                    skipped_count = (
                        1
                        if cfg.perturbation.variant_selection == "all_once"
                        else cfg.benchmark.episodes_per_task
                    )
                    for episode_index in range(skipped_count):
                        jobs.append(
                            _make_job(
                                cfg,
                                task,
                                episode_index,
                                condition="ood",
                                upstream_task_id=-1,
                                upstream_task_name="",
                                category=category,
                                level=level,
                                parameters={
                                    "official_category": definition.upstream_category,
                                    "variant_selection": cfg.perturbation.variant_selection,
                                },
                                skip_reason=reason,
                                initial_state_index=0,
                            )
                        )
    return jobs


def write_jobs(path: Path, jobs: Iterable[EvaluationJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jobs(path: Path) -> list[EvaluationJob]:
    if not path.is_file():
        raise FileNotFoundError(f"Job manifest does not exist: {path}")
    jobs: list[EvaluationJob] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            jobs.append(EvaluationJob.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid job at {path}:{line_number}: {exc}") from exc
    return jobs


def shard_jobs(jobs: Iterable[EvaluationJob], rank: int, world_size: int) -> list[EvaluationJob]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank/world_size: rank={rank}, world_size={world_size}")
    return [job for job in jobs if int(job.job_id[:16], 16) % world_size == rank]
