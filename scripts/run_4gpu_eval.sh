#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_FULL_EVAL:-}" != "YES" ]]; then
  echo "Formal evaluation was not started. Review the config, then set CONFIRM_FULL_EVAL=YES." >&2
  exit 2
fi
config="${1:-configs/eval_ood_full.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
python -m fastwam_ood_eval.cli plan --config "$config"
torchrun --standalone --nproc_per_node=4 -m fastwam_ood_eval.cli distributed-evaluate --config "$config"

