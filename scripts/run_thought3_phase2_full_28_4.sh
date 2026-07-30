#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT3_PHASE2_FULL:-}" != "YES" ]]; then
  echo "Refusing real Phase 2: set CONFIRM_THOUGHT3_PHASE2_FULL=YES" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

resume_args=()
if (( $# > 1 )); then
  echo "Usage: $0 [--resume]" >&2
  exit 2
fi
if (( $# == 1 )); then
  if [[ "$1" != "--resume" ]]; then
    echo "Usage: $0 [--resume]" >&2
    exit 2
  fi
  resume_args=(--resume)
fi

gpu_pair="${THOUGHT3_GPU_IDS:-1,2}"
IFS=',' read -r gpu_a gpu_b extra_gpu <<< "${gpu_pair}"
if [[ -n "${extra_gpu:-}" || -z "${gpu_a:-}" || -z "${gpu_b:-}" ]]; then
  echo "THOUGHT3_GPU_IDS must contain exactly two comma-separated GPUs" >&2
  exit 2
fi
if [[ ! "${gpu_a}" =~ ^[0-9]+$ || ! "${gpu_b}" =~ ^[0-9]+$ ]]; then
  echo "THOUGHT3_GPU_IDS entries must be physical GPU integers" >&2
  exit 2
fi
if [[ "${gpu_a}" == "${gpu_b}" ]]; then
  echo "Phase 2 A0/A1 tracks require two distinct physical GPUs" >&2
  exit 2
fi

check_idle_gpu() {
  local physical_gpu_id="$1"
  local used_mib
  used_mib="$(
    nvidia-smi \
      --id="${physical_gpu_id}" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits
  )"
  used_mib="${used_mib//[[:space:]]/}"
  if [[ ! "${used_mib}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse GPU ${physical_gpu_id} memory.used: ${used_mib}" >&2
    return 2
  fi
  if (( used_mib > 1024 )); then
    echo "GPU ${physical_gpu_id} is not idle (${used_mib} MiB used)" >&2
    return 2
  fi
}

check_idle_gpu "${gpu_a}"
check_idle_gpu "${gpu_b}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${project_root}/src:${project_root}/third_party/FastWAM:${project_root}/third_party/FastWAM/experiments/libero${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${project_root}/.conda/envs/fastwam-ood/bin/python"
config_path="configs/thought3/phase2_full_28_4_a0_a1.yaml"
output_root="${project_root}/outputs/thought3/phase2_full_28_4_a0_a1_v1"
log_dir="${output_root}/logs"
mkdir -p "${log_dir}"

run_gpu_stage() {
  local physical_gpu_id="$1"
  local stage="$2"
  local cache_dir="$3"
  local log_path="$4"
  set +e
  CUDA_VISIBLE_DEVICES="${physical_gpu_id}" \
  THOUGHT3_PHYSICAL_GPU_ID="${physical_gpu_id}" \
  HF_DATASETS_CACHE="${cache_dir}" \
  "${python_bin}" -m fastwam_ood_eval.cli \
    thought3-train-phase2-full \
    --config "${config_path}" \
    --stage "${stage}" \
    "${resume_args[@]}" 2>&1 | tee -a "${log_path}"
  local command_status="${PIPESTATUS[0]}"
  set -e
  return "${command_status}"
}

echo "Phase 2 calibration: physical GPU ${gpu_a}" | tee -a "${log_dir}/launcher.log"
run_gpu_stage \
  "${gpu_a}" \
  "calibrate" \
  "${THOUGHT3_PHASE2_CALIBRATION_HF_CACHE:-/tmp/thought3_phase2_calibration_hf_cache}" \
  "${log_dir}/calibrate.log"

# Calibration releases its model before the matched tracks start. Recheck both
# cards so a newly occupied card cannot silently cause a partial parallel run.
check_idle_gpu "${gpu_a}"
check_idle_gpu "${gpu_b}"

echo "Phase 2 matched tracks: A0 GPU ${gpu_a}, A1 GPU ${gpu_b}" \
  | tee -a "${log_dir}/launcher.log"

set +e
(
  run_gpu_stage \
    "${gpu_a}" \
    "A0" \
    "${THOUGHT3_PHASE2_A0_HF_CACHE:-/tmp/thought3_phase2_a0_hf_cache}" \
    "${log_dir}/a0.log"
) &
a0_pid=$!
(
  run_gpu_stage \
    "${gpu_b}" \
    "A1" \
    "${THOUGHT3_PHASE2_A1_HF_CACHE:-/tmp/thought3_phase2_a1_hf_cache}" \
    "${log_dir}/a1.log"
) &
a1_pid=$!

wait "${a0_pid}"
a0_status=$?
wait "${a1_pid}"
a1_status=$?
set -e

if (( a0_status != 0 || a1_status != 0 )); then
  echo "Phase 2 track failure: A0=${a0_status}, A1=${a1_status}" \
    | tee -a "${log_dir}/launcher.log" >&2
  exit 1
fi

echo "Phase 2 CPU finalize" | tee -a "${log_dir}/launcher.log"
set +e
CUDA_VISIBLE_DEVICES="" \
"${python_bin}" -m fastwam_ood_eval.cli \
  thought3-train-phase2-full \
  --config "${config_path}" \
  --stage finalize \
  "${resume_args[@]}" 2>&1 | tee -a "${log_dir}/finalize.log"
finalize_status="${PIPESTATUS[0]}"
set -e

exit "${finalize_status}"
