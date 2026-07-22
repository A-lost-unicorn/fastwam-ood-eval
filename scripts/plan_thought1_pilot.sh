#!/usr/bin/env bash
set -euo pipefail

# Planning only: 4 suites x (1 clean + 5 perturbations x 3 levels) x 1 episode = 64 jobs.
for suite in libero_spatial libero_object libero_goal libero_10; do
  max_steps=400
  if [[ "$suite" == "libero_10" ]]; then
    max_steps=700
  fi

  python -m fastwam_ood_eval.cli plan \
    --config configs/eval_clean_full.yaml \
    --set "experiment.name=thought1_pilot_fastwam_${suite}_clean" \
    --set "experiment.output_dir=outputs/thought1_pilot/fastwam/${suite}/clean" \
    --set "benchmark.suite=${suite}" \
    --set "benchmark.suite_config=configs/suites/${suite}.yaml" \
    --set 'benchmark.tasks=[0]' \
    --set benchmark.episodes_per_task=1 \
    --set "benchmark.max_steps=${max_steps}"

  python -m fastwam_ood_eval.cli plan \
    --config configs/eval_ood_full.yaml \
    --set "experiment.name=thought1_pilot_fastwam_${suite}_ood" \
    --set "experiment.output_dir=outputs/thought1_pilot/fastwam/${suite}/ood" \
    --set "benchmark.suite=${suite}" \
    --set "benchmark.suite_config=configs/suites/${suite}.yaml" \
    --set 'benchmark.tasks=[0]' \
    --set benchmark.episodes_per_task=1 \
    --set perturbation.variant_selection=sample \
    --set "benchmark.max_steps=${max_steps}"
done
