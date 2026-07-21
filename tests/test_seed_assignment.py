from __future__ import annotations

from conftest import write_config
from fastwam_ood_eval.config import load_config
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

