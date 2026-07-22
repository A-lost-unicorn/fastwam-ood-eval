from __future__ import annotations

from dataclasses import replace

from conftest import write_config
from fastwam_ood_eval.config import load_config
from fastwam_ood_eval.evaluation import jobs as jobs_module
from fastwam_ood_eval.evaluation.jobs import plan_jobs, shard_jobs
from fastwam_ood_eval.reproducibility import episode_seed


def test_seed_assignment_is_stable_and_condition_independent():
    assert episode_seed(4, "suite", "task", 3) == episode_seed(4, "suite", "task", 3)
    assert episode_seed(4, "suite", "task", 3) != episode_seed(4, "suite", "task", 4)


def test_four_rank_shards_have_no_duplicate_or_omission(tmp_path):
    cfg = load_config(write_config(tmp_path, perturbation=True, episodes=8))
    jobs = plan_jobs(cfg)
    shards = [shard_jobs(jobs, rank, 4) for rank in range(4)]
    all_ids = [job.job_id for shard in shards for job in shard]
    assert len(all_ids) == len(set(all_ids)) == len(jobs)
    assert set(all_ids) == {job.job_id for job in jobs}


def test_clean_and_ood_seed_pairing(tmp_path):
    clean = load_config(write_config(tmp_path / "clean", perturbation=False, episodes=2, output_name="clean"))
    ood = load_config(write_config(tmp_path / "ood", perturbation=True, episodes=2, output_name="ood"))
    clean_seeds = {(job.task_name, job.episode_index): job.episode_seed for job in plan_jobs(clean)}
    assert all(clean_seeds[(job.task_name, job.episode_index)] == job.episode_seed for job in plan_jobs(ood))


def test_all_once_enumerates_each_official_variant_exactly_once(tmp_path, monkeypatch):
    cfg = load_config(write_config(tmp_path, perturbation=True, episodes=1))
    cfg = replace(
        cfg,
        benchmark=replace(cfg.benchmark, backend="libero_plus"),
        perturbation=replace(cfg.perturbation, variant_selection="all_once"),
    )
    rows = []
    row_id = 1
    for base_name in ("mock_task_zero", "mock_task_one"):
        for category in ("Camera Viewpoints", "Light Conditions"):
            for suffix, difficulty in (("a", 1), ("b", 2), ("c", 4)):
                rows.append(
                    {
                        "id": row_id,
                        "name": f"{base_name}_{category.lower().replace(' ', '_')}_{suffix}",
                        "category": category,
                        "difficulty_level": difficulty,
                    }
                )
                row_id += 1
    monkeypatch.setattr(jobs_module, "_load_classification", lambda _: rows)

    planned = plan_jobs(cfg)
    assert len(planned) == len(rows)
    assert len({job.upstream_task_id for job in planned}) == len(rows)
    assert all(job.episode_index == 0 for job in planned)
    assert all(job.initial_state_index == 0 for job in planned)
    assert all(job.perturbation_parameters["variant_selection"] == "all_once" for job in planned)
    assert all(not job.perturbation_parameters["selection_with_replacement"] for job in planned)
