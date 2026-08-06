#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT6_PHASE6A:-}" != "YES" ]]; then
  echo "Refusing real Phase 6A: set CONFIRM_THOUGHT6_PHASE6A=YES" >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing real Phase 6A: commit a clean project snapshot first" >&2
  exit 2
fi
gpu_ids="${THOUGHT6_GPU_IDS:-}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+$ ]]; then
  echo "THOUGHT6_GPU_IDS must contain exactly one physical GPU integer" >&2
  exit 2
fi
used_mib="$(nvidia-smi --id="${gpu_ids}" --query-gpu=memory.used --format=csv,noheader,nounits)"
used_mib="${used_mib//[[:space:]]/}"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu_ids} is not an idle 24 GiB window (${used_mib} MiB used)" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${gpu_ids}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_DATASETS_CACHE="${THOUGHT6_HF_DATASETS_CACHE:-/tmp/thought6_hf_cache}"
export PYTHONPATH="${project_root}/src:${project_root}/third_party/FastWAM:${project_root}/third_party/FastWAM/experiments/libero:${project_root}/third_party/LIBERO-plus${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p outputs/thought6/phase6_sigma_aware_future_fusion_v1/logs
"${project_root}/.conda/envs/fastwam-ood/bin/python" -m fastwam_ood_eval.cli \
  thought6-phase6a-smoke --config configs/thought6/phase6a_smoke.yaml --device cuda:0 \
  "$@" 2>&1 | tee -a outputs/thought6/phase6_sigma_aware_future_fusion_v1/logs/phase6a.log
