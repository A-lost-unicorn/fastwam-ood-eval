#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT3_PHASE_C:-}" != "YES" ]]; then
  echo "Refusing real Phase C: set CONFIRM_THOUGHT3_PHASE_C=YES" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

physical_gpu_id="${THOUGHT3_GPU_ID:-1}"
used_mib="$(
  nvidia-smi \
    --id="${physical_gpu_id}" \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits
)"
used_mib="${used_mib//[[:space:]]/}"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse GPU ${physical_gpu_id} memory.used: ${used_mib}" >&2
  exit 2
fi
if (( used_mib > 1024 )); then
  echo "GPU ${physical_gpu_id} is not an idle Phase C window (${used_mib} MiB used)" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu_id}"
export THOUGHT3_PHYSICAL_GPU_ID="${physical_gpu_id}"
export HF_DATASETS_CACHE="${THOUGHT3_HF_DATASETS_CACHE:-/tmp/thought3_phase_c_hf_cache}"
export PYTHONPATH="${project_root}/src:${project_root}/third_party/FastWAM:${project_root}/third_party/FastWAM/experiments/libero${PYTHONPATH:+:${PYTHONPATH}}"

exec "${project_root}/.conda/envs/fastwam-ood/bin/python" -m fastwam_ood_eval.cli \
  thought3-smoke-real \
  --config configs/thought3/phase_c_single_sample.yaml \
  --device cuda:0 \
  "$@"
