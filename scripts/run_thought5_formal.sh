#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT5_FORMAL:-}" != "YES" ]]; then
  echo "Refusing Thought5 formal: set CONFIRM_THOUGHT5_FORMAL=YES" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing Thought5 formal: commit a clean project snapshot first" >&2
  exit 2
fi

gpu_ids="${THOUGHT5_GPU_IDS:-}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "THOUGHT5_GPU_IDS must contain exactly four comma-separated physical IDs" >&2
  exit 2
fi
IFS=',' read -r gpu0 gpu1 gpu2 gpu3 <<< "${gpu_ids}"
if [[ "$(printf '%s\n' "${gpu0}" "${gpu1}" "${gpu2}" "${gpu3}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "Thought5 formal requires four distinct GPUs" >&2
  exit 2
fi

freeze_path="outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v2/formal_protocol_frozen.json"
if [[ ! -f "${freeze_path}" ]] || ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"frozen"' "${freeze_path}"; then
  echo "Formal remains locked: pilot has not sealed formal_protocol_frozen.json" >&2
  exit 2
fi

formal_status="outputs/thought5/phase5_camera_equivariant_geo_repa_v2/run_status.json"
if [[ -f "${formal_status}" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "${formal_status}"; then
  echo "Refusing to mutate completed Thought5 formal output" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${gpu0}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_DATASETS_CACHE="${THOUGHT5_HF_DATASETS_CACHE:-/tmp/thought5_hf_cache}"
export PYTHONPATH="${project_root}/src:${project_root}/third_party/FastWAM:${project_root}/third_party/FastWAM/experiments/libero${PYTHONPATH:+:${PYTHONPATH}}"

output_root="${project_root}/outputs/thought5/phase5_camera_equivariant_geo_repa_v2"
mkdir -p "${output_root}/logs"
"${project_root}/.conda/envs/fastwam-ood/bin/python" \
  -m fastwam_ood_eval.cli \
  thought5-formal \
  --config configs/thought5/phase5_formal_v2.yaml \
  --device cuda:0 \
  "$@" 2>&1 | tee -a "${output_root}/logs/run.log"
