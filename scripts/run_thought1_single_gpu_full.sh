#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  CONFIRM_FULL_EVAL=YES GPU_ID=0 \
    bash scripts/run_thought1_single_gpu_full.sh [all|clean|ood]

Phases:
  all    Run 800 Clean and 6,771 OOD rollouts, then build a combined report.
  clean  Run only the 800 Clean baseline rollouts.
  ood    Run only the 6,771 OOD rollouts.

Environment:
  CONFIRM_FULL_EVAL          Must be YES. Prevents accidental full evaluation.
  GPU_ID                     Physical GPU index exposed to the process (default: 0).
  EGL_DEVICE_ID              EGL device index (default: GPU_ID).
  MIN_FREE_GPU_MEMORY_MB     Required free memory at startup (default: 24000).

The script activates the project-local environment itself. Evaluation uses the
default incomplete-only resume mode: existing completed/max_steps/skipped jobs
are not repeated, including records written previously by other ranks.
EOF
}

phase="${1:-all}"
case "${phase}" in
  all|clean|ood)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Invalid phase: ${phase}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${CONFIRM_FULL_EVAL:-}" != "YES" ]]; then
  echo "Formal evaluation was not started." >&2
  echo "Review the manifests, then set CONFIRM_FULL_EVAL=YES." >&2
  exit 2
fi

gpu_id="${GPU_ID:-0}"
egl_device_id="${EGL_DEVICE_ID:-${gpu_id}}"
min_free_gpu_memory_mb="${MIN_FREE_GPU_MEMORY_MB:-24000}"

if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative physical GPU index, got: ${gpu_id}" >&2
  exit 2
fi
if [[ ! "${egl_device_id}" =~ ^[0-9]+$ ]]; then
  echo "EGL_DEVICE_ID must be a non-negative integer, got: ${egl_device_id}" >&2
  exit 2
fi
if [[ ! "${min_free_gpu_memory_mb}" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_GPU_MEMORY_MB must be a non-negative integer." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

# Make background logs visible immediately and keep all model downloads offline.
export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="${project_root}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${egl_device_id}"

# shellcheck source=scripts/activate_env.sh
source "${project_root}/scripts/activate_env.sh"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing an unattended full run." >&2
  exit 1
fi

if ! gpu_name="$(
  nvidia-smi -i "${gpu_id}" --query-gpu=name --format=csv,noheader 2>&1
)"; then
  echo "Failed to query physical GPU ${gpu_id} with nvidia-smi:" >&2
  printf '%s\n' "${gpu_name}" >&2
  exit 1
fi
gpu_name="${gpu_name//$'\r'/}"
if [[ -z "${gpu_name}" || "${gpu_name}" == *$'\n'* ]]; then
  echo "Unexpected GPU name returned for physical GPU ${gpu_id}: ${gpu_name@Q}" >&2
  exit 1
fi

if ! free_gpu_memory_mb="$(
  nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1
)"; then
  echo "Failed to query free memory for physical GPU ${gpu_id} with nvidia-smi:" >&2
  printf '%s\n' "${free_gpu_memory_mb}" >&2
  exit 1
fi
free_gpu_memory_mb="${free_gpu_memory_mb//[[:space:]]/}"
if [[ ! "${free_gpu_memory_mb}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse free GPU memory from nvidia-smi: ${free_gpu_memory_mb@Q}" >&2
  exit 1
fi
log "physical_gpu=${gpu_id} name=${gpu_name} free_memory_mb=${free_gpu_memory_mb}"
if (( free_gpu_memory_mb < min_free_gpu_memory_mb )); then
  echo "GPU ${gpu_id} has only ${free_gpu_memory_mb} MiB free." >&2
  echo "Fast-WAM pilot peaked near 23.8 GiB; require at least ${min_free_gpu_memory_mb} MiB." >&2
  echo "Stop other GPU processes before retrying." >&2
  exit 1
fi

lock_dir="${project_root}/outputs/thought1/fastwam"
mkdir -p "${lock_dir}"
lock_file="${lock_dir}/.thought1_full.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another Thought 1 full runner holds ${lock_file}." >&2
  echo "Do not run single-GPU and three-GPU evaluators against the same outputs." >&2
  exit 1
fi

suites=(libero_spatial libero_object libero_goal libero_10)

max_steps_for_suite() {
  if [[ "$1" == "libero_10" ]]; then
    printf '700\n'
  else
    printf '400\n'
  fi
}

run_experiment() {
  local condition="$1"
  local suite="$2"
  local config
  local output_dir
  local experiment_name
  local max_steps

  config="configs/eval_${condition}_full.yaml"
  output_dir="outputs/thought1/fastwam/${suite}/${condition}"
  experiment_name="thought1_fastwam_${suite}_${condition}"
  max_steps="$(max_steps_for_suite "${suite}")"

  local overrides=(
    --set "experiment.name=${experiment_name}"
    --set "experiment.output_dir=${output_dir}"
    --set "hardware.devices=[0]"
    --set "hardware.workers_per_gpu=1"
    --set "benchmark.suite=${suite}"
    --set "benchmark.suite_config=configs/suites/${suite}.yaml"
    --set "benchmark.max_steps=${max_steps}"
  )

  log "START condition=${condition} suite=${suite} output=${output_dir}"
  python -m fastwam_ood_eval.cli doctor \
    --config "${config}" \
    "${overrides[@]}"
  python -m fastwam_ood_eval.cli plan \
    --config "${config}" \
    "${overrides[@]}"

  # Avoid loading the 12 GB checkpoint/model when a resumed stage has no work.
  local preview_json
  local pending
  preview_json="$(
    python -m fastwam_ood_eval.cli evaluate \
      --config "${config}" \
      --device cuda:0 \
      --dry-run \
      --rerun incomplete \
      "${overrides[@]}" \
      | tee /dev/stderr \
      | tail -n 1
  )"
  pending="$(
    python -c 'import json, sys; print(json.load(sys.stdin)["pending"])' \
      <<<"${preview_json}"
  )"

  if [[ "${pending}" == "0" ]]; then
    log "SKIP condition=${condition} suite=${suite}; no incomplete jobs"
  else
    log "RUN condition=${condition} suite=${suite} pending=${pending}"
    python -m fastwam_ood_eval.cli evaluate \
      --config "${config}" \
      --device cuda:0 \
      --rerun incomplete \
      "${overrides[@]}"
  fi

  python -m fastwam_ood_eval.cli aggregate \
    --experiment-dir "${output_dir}"
  log "DONE condition=${condition} suite=${suite}"
}

if [[ "${phase}" == "all" || "${phase}" == "clean" ]]; then
  for suite in "${suites[@]}"; do
    run_experiment clean "${suite}"
  done
fi

if [[ "${phase}" == "all" || "${phase}" == "ood" ]]; then
  for suite in "${suites[@]}"; do
    run_experiment ood "${suite}"
  done
fi

if [[ "${phase}" == "all" ]]; then
  combined_args=(
    --experiment-dir outputs/thought1/fastwam/combined
  )
  for suite in "${suites[@]}"; do
    combined_args+=(
      --input-dir "outputs/thought1/fastwam/${suite}/clean"
      --input-dir "outputs/thought1/fastwam/${suite}/ood"
    )
  done
  python -m fastwam_ood_eval.cli aggregate "${combined_args[@]}"
  log "Combined report: outputs/thought1/fastwam/combined/summary/report.md"
fi

log "Thought 1 single-GPU phase '${phase}' finished successfully."
