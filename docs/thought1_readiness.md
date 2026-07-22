# Thought 1 readiness audit

Audit date: 2026-07-22

This document distinguishes implementation readiness from empirical completion. A planned or dry-run experiment is not evidence of generalization.

## Requirement-by-requirement status

| Requirement | Implementation evidence | Empirical evidence required | Status |
| --- | --- | --- | --- |
| Three-GPU evaluation | Full configs use devices 0–2; `run_3gpu_eval.sh`; host doctor sees 3×47.37 GiB; 3-rank dry-run partitions all jobs | At least one real smoke episode per rank | Ready, not executed |
| Cross-environment generalization | Same-checkpoint Clean/LIBERO-Plus pairing; five official categories; three difficulty bands; matched seed; per-policy checkpoint-hash guard | Clean and OOD episode JSONL plus aggregate report | Ready, no results |
| Cross-object unseen generalization | Per-object/task aggregation and `libero_object` suite plan | A checkpoint trained with a frozen, disjoint object holdout and results on that holdout | Not identifiable with release checkpoint |
| Cross-task unseen generalization | All four suite plans and task-level aggregation | A checkpoint trained without the frozen held-out tasks/suite and results on those tasks | Not identifiable with release checkpoint |
| Cross-platform generalization | The limitation is represented in `configs/studies/thought1.yaml` | One policy/checkpoint with a defined LIBERO→target-platform observation/action mapping and target-platform rollouts | No compatible cross-platform policy/adapter |
| Future imagination improves unseen generalization | Explicit Fast-WAM/Joint/IDM identity, class/checkpoint guard, paired success difference, bootstrap CI and exact McNemar test | Recipe-matched no-future/future checkpoints evaluated on identical OOD jobs across multiple training seeds | No official matched future checkpoint |

## Authoritative current-state evidence

- `outputs/thought1/` contains eight full manifests, but no worker episode-result JSONL.
- `outputs/thought1_pilot/` contains eight pilot manifests: 64 planned jobs, 59 runnable and 5 explicitly skipped; no episode was executed.
- `outputs/ablations/` contains Joint WAM smoke manifests only; no Joint checkpoint or result exists.
- Host hardware validation passes for three CUDA devices. The configured 23 GiB memory budget is below each reported 47.37 GiB capacity.
- `pytest -q` passes 26 tests, covering configuration guards, deterministic sharding, resume, aggregation, mixed-policy handling and paired future/no-future statistics. These tests validate machinery, not robot-task performance.

## Missing external artifacts

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
third_party/LIBERO-plus/libero/libero/assets/
checkpoints/fastwam_ablation/libero_joint_2cam224.pt
checkpoints/fastwam_ablation/libero_joint_2cam224_dataset_stats.json
```

The first three unblock Fast-WAM Clean/OOD smoke. The final two only unblock an exploratory Joint WAM comparison unless training parity with the no-future checkpoint is independently established.

## Decisions required to unblock the complete research question

1. Authorize downloading the official Fast-WAM checkpoint/stats and LIBERO-Plus assets, then run only the 2-job Clean and 4-job OOD smoke.
2. Decide whether a third-party Joint/IDM checkpoint may be used as an explicitly associational baseline. Do not set `training_recipe_id` for such a comparison without stronger provenance.
3. If the desired claim is truly unseen-object, unseen-task and cross-platform causal generalization, authorize a separate training/split project. The released checkpoint was trained on all four evaluated LIBERO suites, and the current stage's no-training constraint cannot produce those missing counterfactual checkpoints.

Until these conditions are met, the scientifically correct result is “not yet measured,” not “Fast-WAM generalizes” or “future imagination helps.”
