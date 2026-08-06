#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT6_PHASE6B:-}" != "YES" ]]; then
  echo "Refusing Phase 6B: set CONFIRM_THOUGHT6_PHASE6B=YES" >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
gpu_ids="${THOUGHT6_GPU_IDS:-}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "Phase 6B requires exactly three comma-separated physical GPU IDs" >&2
  exit 2
fi
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${project_root}/.conda/envs/fastwam-ood/bin/python" -m fastwam_ood_eval.cli \
  thought6-phase6b-utility --config configs/thought6/phase6b_offline_utility.yaml --device cuda:0 "$@"
