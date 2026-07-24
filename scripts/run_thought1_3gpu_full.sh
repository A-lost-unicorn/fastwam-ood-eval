#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  CONFIRM_FULL_EVAL=YES GPU_IDS=0,1,2 \
    bash scripts/run_thought1_3gpu_full.sh [all|clean|ood]

Phases:
  all    Run 800 Clean and 6,771 OOD rollouts, then build a combined report.
  clean  Run only the 800 Clean baseline rollouts.
  ood    Run only the 6,771 OOD rollouts.

Environment:
  CONFIRM_FULL_EVAL          Must be YES. Prevents accidental full evaluation.
  GPU_IDS                    Three unique physical GPU indices (default: 0,1,2).
  MIN_FREE_GPU_MEMORY_MB     Required free memory on every GPU (default: 24000).
  REQUIRED_GIT_BRANCH        Required branch for provenance (default: main).

The script activates the project-local environment itself. Evaluation uses
three torchrun workers and incomplete-only resume. Existing completed,
max_steps and skipped records in any rank directory are not repeated.
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

gpu_ids_csv="${GPU_IDS:-0,1,2}"
min_free_gpu_memory_mb="${MIN_FREE_GPU_MEMORY_MB:-24000}"
required_git_branch="${REQUIRED_GIT_BRANCH:-main}"

if [[ ! "${gpu_ids_csv}" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "GPU_IDS must contain exactly three physical indices, for example 0,1,2." >&2
  exit 2
fi
if [[ ! "${min_free_gpu_memory_mb}" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_GPU_MEMORY_MB must be a non-negative integer." >&2
  exit 2
fi

IFS=',' read -r -a physical_gpu_ids <<<"${gpu_ids_csv}"
declare -A seen_gpu_ids=()
for gpu_id in "${physical_gpu_ids[@]}"; do
  if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
    echo "GPU_IDS contains a duplicate physical index: ${gpu_id}" >&2
    exit 2
  fi
  seen_gpu_ids["${gpu_id}"]=1
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

current_git_branch="$(git branch --show-current)"
if [[ "${current_git_branch}" != "${required_git_branch}" ]]; then
  echo "Formal evaluation requires branch ${required_git_branch}; current branch is ${current_git_branch}." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal evaluation requires a clean Git worktree:" >&2
  git status --short >&2
  exit 1
fi

# Make background logs visible immediately and keep all model downloads offline.
export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="${project_root}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES="${gpu_ids_csv}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# distributed_evaluate sets one logical EGL device per LOCAL_RANK. A global
# value would force all workers onto the same renderer.
unset MUJOCO_EGL_DEVICE_ID

# shellcheck source=scripts/activate_env.sh
source "${project_root}/scripts/activate_env.sh"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing an unattended full run." >&2
  exit 1
fi

for gpu_id in "${physical_gpu_ids[@]}"; do
  gpu_name="$(
    nvidia-smi --id="${gpu_id}" --query-gpu=name --format=csv,noheader \
      | head -n 1 \
      | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  )"
  free_gpu_memory_mb="$(
    nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits \
      | head -n 1 \
      | tr -d '[:space:]'
  )"
  if [[ ! "${free_gpu_memory_mb}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse free memory for physical GPU ${gpu_id}." >&2
    exit 1
  fi
  log "physical_gpu=${gpu_id} name=${gpu_name} free_memory_mb=${free_gpu_memory_mb}"
  if (( free_gpu_memory_mb < min_free_gpu_memory_mb )); then
    echo "GPU ${gpu_id} has only ${free_gpu_memory_mb} MiB free." >&2
    echo "Fast-WAM pilot peaked near 23.8 GiB per worker; require at least ${min_free_gpu_memory_mb} MiB." >&2
    echo "Stop other GPU processes before retrying." >&2
    exit 1
  fi
done

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
    --set "hardware.devices=[0,1,2]"
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

  # A world-size-one dry run sees the entire manifest and all rank result files.
  # This avoids loading three copies of the model when a resumed stage is done.
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
    log "RUN condition=${condition} suite=${suite} pending=${pending} world_size=3"
    torchrun \
      --standalone \
      --nproc_per_node=3 \
      -m fastwam_ood_eval.cli distributed-evaluate \
      --config "${config}" \
      --rerun incomplete \
      "${overrides[@]}"
  fi

  python -m fastwam_ood_eval.cli aggregate \
    --experiment-dir "${output_dir}"
  log "DONE condition=${condition} suite=${suite}"
}

log "git_branch=${current_git_branch} git_commit=$(git rev-parse HEAD)"
log "phase=${phase} physical_gpu_ids=${gpu_ids_csv}"

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

log "Thought 1 three-GPU phase '${phase}' finished successfully."
