#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT5_SMOKE:-}" != "YES" ]]; then
  echo "Refusing real Thought5 smoke: set CONFIRM_THOUGHT5_SMOKE=YES" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing real Thought5 smoke: commit a clean project snapshot first" >&2
  exit 2
fi

gpu_ids="${THOUGHT5_GPU_IDS:-}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+$ ]]; then
  echo "THOUGHT5_GPU_IDS must contain exactly one physical GPU integer" >&2
  exit 2
fi

status_path="${project_root}/outputs/thought5/phase5_camera_equivariant_geo_repa_smoke_v4/run_status.json"
if [[ -f "${status_path}" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "${status_path}"; then
  echo "Refusing to mutate completed Thought5 smoke output" >&2
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
export HF_DATASETS_CACHE="${THOUGHT5_HF_DATASETS_CACHE:-/tmp/thought5_hf_cache}"
export PYTHONPATH="${project_root}/src:${project_root}/third_party/FastWAM:${project_root}/third_party/FastWAM/experiments/libero${PYTHONPATH:+:${PYTHONPATH}}"

output_root="${project_root}/outputs/thought5/phase5_camera_equivariant_geo_repa_smoke_v4"
mkdir -p "${output_root}/logs"
log_path="${output_root}/logs/run.log"

set +e
"${project_root}/.conda/envs/fastwam-ood/bin/python" \
  -m fastwam_ood_eval.cli \
  thought5-smoke \
  --config configs/thought5/phase5_smoke_v4.yaml \
  --device cuda:0 \
  "$@" 2>&1 | tee -a "${log_path}"
status="${PIPESTATUS[0]}"
set -e
exit "${status}"
