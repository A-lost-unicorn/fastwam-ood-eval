from __future__ import annotations

import json

from conftest import write_config
from fastwam_ood_eval.config import load_config
from fastwam_ood_eval.evaluation.jobs import plan_jobs
from fastwam_ood_eval.evaluation.resume import filter_jobs_for_resume, load_result_records


def test_resume_skips_completed_job(tmp_path):
    jobs = plan_jobs(load_config(write_config(tmp_path, episodes=2)))
    records = {jobs[0].job_id: {"job_id": jobs[0].job_id, "termination_reason": "success"}}
    pending = filter_jobs_for_resume(jobs, records)
    assert jobs[0] not in pending
    assert len(pending) == len(jobs) - 1


def test_failed_filter_only_retries_failures(tmp_path):
    jobs = plan_jobs(load_config(write_config(tmp_path, episodes=2)))
    records = {
        jobs[0].job_id: {"job_id": jobs[0].job_id, "termination_reason": "success"},
        jobs[1].job_id: {"job_id": jobs[1].job_id, "termination_reason": "exception"},
    }
    pending = filter_jobs_for_resume(jobs[:2], records, rerun="failed")
    assert pending == [jobs[1]]


def test_partial_crash_line_preserves_previous_result(tmp_path):
    path = tmp_path / "episode_results.jsonl"
    path.write_text(json.dumps({"job_id": "complete", "termination_reason": "success"}) + "\n{broken", encoding="utf-8")
    assert set(load_result_records([path])) == {"complete"}

