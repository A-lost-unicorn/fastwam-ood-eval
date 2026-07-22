#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_FULL_EVAL:-}" != "YES" ]]; then
  echo "Formal evaluation was not started. Review the plan, then set CONFIRM_FULL_EVAL=YES." >&2
  exit 2
fi

config="${1:-configs/eval_ood_full.yaml}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
extra_args=("$@")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -m fastwam_ood_eval.cli doctor --config "$config" "${extra_args[@]}"
python -m fastwam_ood_eval.cli plan --config "$config" "${extra_args[@]}"
torchrun --standalone --nproc_per_node=3 \
  -m fastwam_ood_eval.cli distributed-evaluate \
  --config "$config" "${extra_args[@]}"
