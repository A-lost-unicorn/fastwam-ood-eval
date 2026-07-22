#!/usr/bin/env bash
set -euo pipefail

# Planning only: this script does not load a checkpoint, simulator, or model.
for suite in libero_spatial libero_object libero_goal libero_10; do
  max_steps=400
  if [[ "$suite" == "libero_10" ]]; then
    max_steps=700
  fi

  python -m fastwam_ood_eval.cli plan \
    --config configs/eval_clean_full.yaml \
    --set "experiment.name=thought1_fastwam_${suite}_clean" \
    --set "experiment.output_dir=outputs/thought1/fastwam/${suite}/clean" \
    --set "benchmark.suite=${suite}" \
    --set "benchmark.suite_config=configs/suites/${suite}.yaml" \
    --set "benchmark.max_steps=${max_steps}"

  python -m fastwam_ood_eval.cli plan \
    --config configs/eval_ood_full.yaml \
    --set "experiment.name=thought1_fastwam_${suite}_ood" \
    --set "experiment.output_dir=outputs/thought1/fastwam/${suite}/ood" \
    --set "benchmark.suite=${suite}" \
    --set "benchmark.suite_config=configs/suites/${suite}.yaml" \
    --set "benchmark.max_steps=${max_steps}"
done
