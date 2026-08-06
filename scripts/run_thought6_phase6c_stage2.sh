#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT6_PHASE6C_STAGE2:-}" != "YES" ]]; then
  echo "Refusing Phase 6C Stage 2: set CONFIRM_THOUGHT6_PHASE6C_STAGE2=YES" >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
gpu_ids="${THOUGHT6_GPU_IDS:-}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "Phase 6C Stage 2 requires exactly three physical GPU IDs" >&2
  exit 2
fi
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${project_root}/.conda/envs/fastwam-ood/bin/python" -m fastwam_ood_eval.cli \
  thought6-phase6c-stage2 --config configs/thought6/phase6c_rollout_stage2.yaml --device cuda:0 "$@"
