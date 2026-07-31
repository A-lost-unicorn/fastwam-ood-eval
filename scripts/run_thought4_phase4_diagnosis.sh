#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT4_PHASE4_FORMAL:-}" != "YES" ]]; then
  echo "Refusing formal Thought4 diagnosis: set CONFIRM_THOUGHT4_PHASE4_FORMAL=YES" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing formal Thought4 diagnosis: commit a clean project snapshot first" >&2
  exit 2
fi

status_path="${project_root}/outputs/thought4/phase4_geometry_action_diagnosis_v1/run_status.json"
if [[ -f "${status_path}" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "${status_path}"; then
  echo "Refusing to mutate completed Thought4 formal output" >&2
  exit 2
fi

physical_gpu_id="${THOUGHT4_GPU_ID:-}"
if [[ ! "${physical_gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "THOUGHT4_GPU_ID must be exactly one physical GPU integer" >&2
  exit 2
fi

used_mib="$(
  nvidia-smi \
    --id="${physical_gpu_id}" \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits
)"
used_mib="${used_mib//[[:space:]]/}"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${physical_gpu_id} is not an idle 24 GiB window (${used_mib} MiB used)" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu_id}"
export MUJOCO_GL=egl
# robosuite uses the physical ID from CUDA_VISIBLE_DEVICES; torch sees cuda:0.
export MUJOCO_EGL_DEVICE_ID="${physical_gpu_id}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_DATASETS_CACHE="${THOUGHT4_HF_DATASETS_CACHE:-/tmp/thought4_hf_cache}"
export PYTHONPATH="${project_root}/src:${project_root}/third_party/FastWAM:${project_root}/third_party/FastWAM/experiments/libero${PYTHONPATH:+:${PYTHONPATH}}"

output_root="${project_root}/outputs/thought4/phase4_geometry_action_diagnosis_v1"
mkdir -p "${output_root}/logs"
log_path="${output_root}/logs/run.log"

set +e
"${project_root}/.conda/envs/fastwam-ood/bin/python" \
  -m fastwam_ood_eval.cli \
  thought4-phase4-diagnosis \
  --config configs/thought4/phase4_geometry_action_diagnosis_v1.yaml \
  --device cuda:0 \
  "$@" 2>&1 | tee -a "${log_path}"
status="${PIPESTATUS[0]}"
set -e
exit "${status}"
