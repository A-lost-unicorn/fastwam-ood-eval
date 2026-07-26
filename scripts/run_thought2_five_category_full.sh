#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  CONFIRM_PHASE2_FIVE_CATEGORY=YES ACCEPT_STATIC_THRESHOLD=YES \
    GPU_IDS=0,1,2 \
    bash scripts/run_thought2_five_category_full.sh [--background] \
      [all|calibrate|diagnose|aggregate]

Phases:
  all        Ratify the exact 732-job cohort, run the 200-job null
             calibration, run all eight suite×condition diagnostics, aggregate.
  calibrate  Run/resume 100 Clean + 100 OOD no-op calibration jobs and report.
  diagnose   Ratify the cohort, run/resume 200 Clean + 532 OOD diagnostics.
  aggregate  Rebuild per-run and four-suite combined diagnostic reports only.

Evidence boundary:
  The eight source cohort drafts were selected before Phase 1 outcome JSONL
  existed. They were not frozen then. This runner ratifies the exact same job
  IDs before formal Phase 2 future metrics and records that Phase 1 outcomes
  already existed. It never claims pre-registration before Phase 1 outcomes.

Environment:
  CONFIRM_PHASE2_FIVE_CATEGORY
      Must be YES for any run. Prevents accidental 15–18 GPU-hour execution.
  ACCEPT_STATIC_THRESHOLD
      Must be YES for all/diagnose. Confirms use of the 200-job calibration
      candidate only when every automatic freeze-eligibility gate passes.
  GPU_IDS
      One to four unique physical GPU indices (default: 0,1,2). Three GPUs are
      the previously validated configuration.
  MIN_FREE_GPU_MEMORY_MB
      Required free memory on every GPU (default: 24000).
  REQUIRED_GIT_BRANCH
      Required branch for provenance (default: main).
  RUN_ROOT
      Fresh/resumable output namespace
      (default: outputs/thought2/five_category_formal_v1).

Background:
  --background launches the same phase with nohup, writes a timestamped log
  under RUN_ROOT/logs, and prints the PID/log path. The child keeps the same
  safety environment and uses incomplete-only resume.
EOF
}

background=false
if [[ "${1:-}" == "--background" ]]; then
  background=true
  shift
fi

phase="${1:-all}"
case "${phase}" in
  all|calibrate|diagnose|aggregate)
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

run_root="${RUN_ROOT:-outputs/thought2/five_category_formal_v1}"

if [[ "${background}" == "true" ]]; then
  if [[ "${CONFIRM_PHASE2_FIVE_CATEGORY:-}" != "YES" ]]; then
    echo "Thought 2 five-category run was not started." >&2
    echo "Review the protocol, then set CONFIRM_PHASE2_FIVE_CATEGORY=YES." >&2
    exit 2
  fi
  if [[ "${phase}" == "all" || "${phase}" == "diagnose" ]]; then
    if [[ "${ACCEPT_STATIC_THRESHOLD:-}" != "YES" ]]; then
      echo "Background diagnostics were not started." >&2
      echo "Set ACCEPT_STATIC_THRESHOLD=YES to accept the automatic threshold gate." >&2
      exit 2
    fi
  fi
  mkdir -p "${run_root}/logs"
  lock_file="${run_root}/.runner.lock"
  exec 8>"${lock_file}"
  if ! flock -n 8; then
    echo "Another Thought 2 runner holds ${lock_file}." >&2
    exit 1
  fi
  export THOUGHT2_RUNNER_LOCK_HELD=YES
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_path="${run_root}/logs/runner_${phase}_${timestamp}.log"
  pid_path="${run_root}/runner.pid"
  nohup bash "${script_dir}/run_thought2_five_category_full.sh" "${phase}" \
    >"${log_path}" 2>&1 </dev/null &
  child_pid="$!"
  printf '%s\n' "${child_pid}" >"${pid_path}"
  printf 'Started Thought 2 phase=%s pid=%s\n' "${phase}" "${child_pid}"
  printf 'Log: %s\n' "${log_path}"
  printf 'Monitor: tail -f %q\n' "${log_path}"
  printf 'Status: cat %q\n' "${run_root}/run_status.txt"
  exit 0
fi

if [[ "${CONFIRM_PHASE2_FIVE_CATEGORY:-}" != "YES" ]]; then
  echo "Thought 2 five-category run was not started." >&2
  echo "Review the protocol, then set CONFIRM_PHASE2_FIVE_CATEGORY=YES." >&2
  exit 2
fi
if [[ "${phase}" == "all" || "${phase}" == "diagnose" ]]; then
  if [[ "${ACCEPT_STATIC_THRESHOLD:-}" != "YES" ]]; then
    echo "Diagnostics were not started." >&2
    echo "Set ACCEPT_STATIC_THRESHOLD=YES only after accepting the automatic" >&2
    echo "200-job calibration eligibility gate as the threshold lock." >&2
    exit 2
  fi
fi

gpu_ids_csv="${GPU_IDS:-0,1,2}"
min_free_gpu_memory_mb="${MIN_FREE_GPU_MEMORY_MB:-24000}"
required_git_branch="${REQUIRED_GIT_BRANCH:-main}"

if [[ ! "${gpu_ids_csv}" =~ ^[0-9]+(,[0-9]+){0,3}$ ]]; then
  echo "GPU_IDS must contain one to four physical indices, e.g. 0,1,2." >&2
  exit 2
fi
if [[ ! "${min_free_gpu_memory_mb}" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_GPU_MEMORY_MB must be a non-negative integer." >&2
  exit 2
fi

IFS=',' read -r -a physical_gpu_ids <<<"${gpu_ids_csv}"
declare -A seen_gpu_ids=()
logical_gpu_ids=()
for index in "${!physical_gpu_ids[@]}"; do
  gpu_id="${physical_gpu_ids[${index}]}"
  if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
    echo "GPU_IDS contains a duplicate physical index: ${gpu_id}" >&2
    exit 2
  fi
  seen_gpu_ids["${gpu_id}"]=1
  logical_gpu_ids+=("${index}")
done
world_size="${#physical_gpu_ids[@]}"
logical_devices_yaml="[$(IFS=,; printf '%s' "${logical_gpu_ids[*]}")]"

current_git_branch="$(git branch --show-current)"
if [[ "${current_git_branch}" != "${required_git_branch}" ]]; then
  echo "Formal diagnostics require branch ${required_git_branch}; current branch is ${current_git_branch}." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Formal diagnostics require a clean project worktree:" >&2
  git status --short >&2
  echo "Commit the reviewed runner/config/code changes before launching." >&2
  exit 1
fi

check_upstream_clean() {
  local path="$1"
  local ignored_prefix="${2:-}"
  local entry
  local relevant=""
  while IFS= read -r entry; do
    if [[ -z "${entry}" ]]; then
      continue
    fi
    if [[ -n "${ignored_prefix}" && "${entry}" == "?? ${ignored_prefix}"* ]]; then
      continue
    fi
    relevant+="${entry}"$'\n'
  done < <(git -C "${path}" status --porcelain=v1 --untracked-files=all)
  if [[ -n "${relevant}" ]]; then
    echo "Formal diagnostics require a clean upstream tree: ${path}" >&2
    printf '%s' "${relevant}" >&2
    exit 1
  fi
}

check_upstream_clean third_party/FastWAM
check_upstream_clean third_party/LIBERO
check_upstream_clean third_party/LIBERO-plus ".downloads/"

mkdir -p "${run_root}"
lock_file="${run_root}/.runner.lock"
if [[ "${THOUGHT2_RUNNER_LOCK_HELD:-}" == "YES" ]]; then
  # The --background parent acquired FD 8 before writing runner.pid. Retain
  # that same open-file-description lock on FD 9 for this child's lifetime.
  exec 9>&8
  exec 8>&-
  unset THOUGHT2_RUNNER_LOCK_HELD
else
  exec 9>"${lock_file}"
  if ! flock -n 9; then
    echo "Another Thought 2 runner holds ${lock_file}." >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="${project_root}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES="${gpu_ids_csv}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset MUJOCO_EGL_DEVICE_ID

# shellcheck source=scripts/activate_env.sh
source "${project_root}/scripts/activate_env.sh"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*"
}

run_status_path="${run_root}/run_status.txt"
run_started_at="$(date '+%F %T')"
on_exit() {
  local status="$?"
  local label="failed"
  if [[ "${status}" == "0" ]]; then
    label="completed"
  fi
  printf 'status=%s\nphase=%s\nexit_code=%s\nstarted_at=%s\nfinished_at=%s\n' \
    "${label}" "${phase}" "${status}" "${run_started_at}" "$(date '+%F %T')" \
    >"${run_status_path}"
}
trap on_exit EXIT

needs_gpu=false
if [[ "${phase}" == "all" || "${phase}" == "calibrate" || "${phase}" == "diagnose" ]]; then
  needs_gpu=true
fi
if [[ "${needs_gpu}" == "true" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing an unattended run." >&2
    exit 1
  fi
  for gpu_id in "${physical_gpu_ids[@]}"; do
    if ! gpu_name="$(
      nvidia-smi -i "${gpu_id}" --query-gpu=name --format=csv,noheader 2>&1
    )"; then
      echo "Failed to query physical GPU ${gpu_id}:" >&2
      printf '%s\n' "${gpu_name}" >&2
      exit 1
    fi
    gpu_name="${gpu_name//$'\r'/}"
    if ! free_gpu_memory_mb="$(
      nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1
    )"; then
      echo "Failed to query free memory for physical GPU ${gpu_id}:" >&2
      printf '%s\n' "${free_gpu_memory_mb}" >&2
      exit 1
    fi
    free_gpu_memory_mb="${free_gpu_memory_mb//[[:space:]]/}"
    if [[ ! "${free_gpu_memory_mb}" =~ ^[0-9]+$ ]]; then
      echo "Could not parse free memory for GPU ${gpu_id}: ${free_gpu_memory_mb@Q}" >&2
      exit 1
    fi
    log "physical_gpu=${gpu_id} name=${gpu_name} free_memory_mb=${free_gpu_memory_mb}"
    if (( free_gpu_memory_mb < min_free_gpu_memory_mb )); then
      echo "GPU ${gpu_id} has only ${free_gpu_memory_mb} MiB free; require ${min_free_gpu_memory_mb} MiB." >&2
      exit 1
    fi
  done
fi

suites=(libero_spatial libero_object libero_goal libero_10)
conditions=(clean ood)
draft_root="outputs/thought2_outcome_blind_cohort_draft_v2"
cohort_root="${run_root}/cohorts"
static_root="${run_root}/static"
diagnostic_root="${run_root}/diagnostics"
comparison_root="${run_root}/combined"
diagnostic_config="configs/studies/thought2_unconditional_formal_five_category.yaml"

max_steps_for_suite() {
  if [[ "$1" == "libero_10" ]]; then
    printf '700\n'
  else
    printf '400\n'
  fi
}

ratify_cohorts() {
  local suite
  local condition
  local source_dir
  local draft
  local ratified
  mkdir -p "${cohort_root}"
  for suite in "${suites[@]}"; do
    mkdir -p "${cohort_root}/${suite}"
    for condition in "${conditions[@]}"; do
      source_dir="outputs/thought1/fastwam/${suite}/${condition}"
      draft="${draft_root}/${suite}/${condition}.json"
      ratified="${cohort_root}/${suite}/${condition}.json"
      if [[ -f "${ratified}" ]]; then
        log "VALIDATE ratified cohort suite=${suite} condition=${condition}"
        python -m fastwam_ood_eval.cli validate-diagnostic-cohort \
          --manifest "${ratified}" \
          --source-dir "${source_dir}"
      else
        log "RATIFY exact cohort suite=${suite} condition=${condition}"
        python -m fastwam_ood_eval.cli ratify-diagnostic-cohort \
          --draft-manifest "${draft}" \
          --source-dir "${source_dir}" \
          --output "${ratified}"
      fi
    done
  done

  selected_total="$(
    python -c \
      'import json,sys; print(sum(len(json.load(open(p, encoding="utf-8"))["selected_jobs"]) for p in sys.argv[1:]))' \
      "${cohort_root}"/libero_*/clean.json \
      "${cohort_root}"/libero_*/ood.json
  )"
  if [[ "${selected_total}" != "732" ]]; then
    echo "Ratified five-category cohort must contain exactly 732 jobs; found ${selected_total}." >&2
    exit 1
  fi
  log "COHORT ready selected_jobs=${selected_total}"
}

run_static_condition() {
  local condition="$1"
  local config="configs/studies/thought2_static_calibration_formal_${condition}.yaml"
  local output_dir="${static_root}/${condition}"
  local experiment_name="thought2_five_category_formal_static_${condition}"
  local overrides=(
    --set "experiment.name=${experiment_name}"
    --set "experiment.output_dir=${output_dir}"
    --set "hardware.devices=${logical_devices_yaml}"
    --set "hardware.workers_per_gpu=1"
  )

  log "STATIC PREFLIGHT condition=${condition}"
  python -m fastwam_ood_eval.cli doctor \
    --config "${config}" \
    "${overrides[@]}"

  preview_json="$(
    python -m fastwam_ood_eval.cli calibrate-static \
      --config "${config}" \
      --device cuda:0 \
      --dry-run \
      --rerun incomplete \
      "${overrides[@]}" \
      | tee /dev/stderr \
      | tail -n 1
  )"
  pending="$(
    python -c 'import json,sys; print(json.load(sys.stdin)["pending"])' \
      <<<"${preview_json}"
  )"
  if [[ "${pending}" == "0" ]]; then
    log "STATIC SKIP condition=${condition}; no incomplete jobs"
  else
    log "STATIC RUN condition=${condition} pending=${pending} world_size=${world_size}"
    torchrun \
      --standalone \
      --nproc_per_node="${world_size}" \
      -m fastwam_ood_eval.cli distributed-calibrate-static \
      --config "${config}" \
      --rerun incomplete \
      "${overrides[@]}"
  fi
}

aggregate_static() {
  log "STATIC AGGREGATE"
  python -m fastwam_ood_eval.cli aggregate-static-calibration \
    --experiment-dir "${static_root}/combined" \
    --input-dir "${static_root}/clean" \
    --input-dir "${static_root}/ood"
  python -m fastwam_ood_eval.cli report-static-calibration \
    --experiment-dir "${static_root}/combined"
}

formal_threshold() {
  local summary="${static_root}/combined/summary/static_calibration_summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "Formal static calibration summary is missing: ${summary}" >&2
    exit 1
  fi
  python -c \
    'import json,math,sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
value=p.get("candidate_static_motion_threshold")
if p.get("freeze_eligible") is not True or p.get("threshold_status") != "eligible_for_manual_freeze":
    raise SystemExit("static calibration has not passed every freeze-eligibility gate")
if not isinstance(value, (int,float)) or not math.isfinite(float(value)) or float(value) < 0:
    raise SystemExit("static calibration threshold is unavailable or invalid")
print(repr(float(value)))' \
    "${summary}"
}

run_diagnostic_condition() {
  local condition="$1"
  local suite="$2"
  local threshold="$3"
  local max_steps
  local output_dir="${diagnostic_root}/${suite}/${condition}"
  local source_dir="outputs/thought1/fastwam/${suite}/${condition}"
  local source_id="thought1_fastwam_${suite}_${condition}"
  local cohort="${cohort_root}/${suite}/${condition}.json"
  local experiment_name="thought2_five_category_${suite}_${condition}"
  max_steps="$(max_steps_for_suite "${suite}")"

  local overrides=(
    --set "experiment.name=${experiment_name}"
    --set "experiment.output_dir=${output_dir}"
    --set "hardware.devices=${logical_devices_yaml}"
    --set "hardware.workers_per_gpu=1"
    --set "benchmark.suite=${suite}"
    --set "benchmark.suite_config=configs/suites/${suite}.yaml"
    --set "benchmark.tasks=all"
    --set "benchmark.max_steps=${max_steps}"
    --set "diagnostics.source_experiment_id=${source_id}"
    --set "diagnostics.source_output_dir=${source_dir}"
    --set "diagnostics.cohort_manifest_path=${cohort}"
    --set "diagnostics.require_frozen_cohort=true"
    --set "diagnostics.static_motion_threshold=${threshold}"
  )

  if [[ "${condition}" == "clean" ]]; then
    overrides+=(
      --set "benchmark.backend=libero"
      --set "benchmark.episodes_per_task=20"
      --set "perturbation.enabled=false"
      --set "perturbation.category=[]"
      --set "perturbation.level=[]"
      --set "perturbation.variant_selection=sample"
      --set "perturbation.parameters={}"
    )
  else
    overrides+=(
      --set "benchmark.backend=libero_plus"
      --set "benchmark.episodes_per_task=1"
      --set "perturbation.enabled=true"
      --set "perturbation.category=[camera_viewpoints,light_conditions,background_textures,robot_initial_states,objects_layout]"
      --set "perturbation.level=[easy,medium,hard]"
      --set "perturbation.variant_selection=all_once"
      --set "perturbation.parameters.classification_path=third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    )
  fi

  log "DIAGNOSTIC PREFLIGHT suite=${suite} condition=${condition}"
  python -m fastwam_ood_eval.cli doctor \
    --config "${diagnostic_config}" \
    "${overrides[@]}"

  preview_json="$(
    python -m fastwam_ood_eval.cli diagnose-future \
      --config "${diagnostic_config}" \
      --device cuda:0 \
      --dry-run \
      --rerun incomplete \
      "${overrides[@]}" \
      | tee /dev/stderr \
      | tail -n 1
  )"
  pending="$(
    python -c 'import json,sys; print(json.load(sys.stdin)["pending"])' \
      <<<"${preview_json}"
  )"
  if [[ "${pending}" == "0" ]]; then
    log "DIAGNOSTIC SKIP suite=${suite} condition=${condition}; no incomplete jobs"
  else
    log "DIAGNOSTIC RUN suite=${suite} condition=${condition} pending=${pending} world_size=${world_size}"
    torchrun \
      --standalone \
      --nproc_per_node="${world_size}" \
      -m fastwam_ood_eval.cli distributed-diagnose-future \
      --config "${diagnostic_config}" \
      --rerun incomplete \
      "${overrides[@]}"
  fi

  python -m fastwam_ood_eval.cli aggregate-diagnostics \
    --experiment-dir "${output_dir}"
  python -m fastwam_ood_eval.cli report-diagnostics \
    --experiment-dir "${output_dir}"
  log "DIAGNOSTIC DONE suite=${suite} condition=${condition}"
}

aggregate_diagnostics_all() {
  local args=(
    --experiment-dir "${comparison_root}"
  )
  local suite
  local condition
  for suite in "${suites[@]}"; do
    for condition in "${conditions[@]}"; do
      args+=(
        --input-dir "${diagnostic_root}/${suite}/${condition}"
      )
    done
  done
  log "DIAGNOSTIC AGGREGATE suites=4 conditions=2"
  python -m fastwam_ood_eval.cli aggregate-diagnostics "${args[@]}"
  python -m fastwam_ood_eval.cli report-diagnostics \
    --experiment-dir "${comparison_root}"

  python -c \
    'import json,sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
d=p["denominators"]
if d.get("planned_jobs") != 732:
    raise SystemExit("expected 732 planned jobs, found %r" % d.get("planned_jobs"))
if p.get("episodes") != 732:
    raise SystemExit("expected 732 diagnostic episodes, found %r" % p.get("episodes"))
if d.get("error_clips") != 0:
    raise SystemExit("diagnostic aggregate contains %r error clips" % d.get("error_clips"))
print(json.dumps({"episodes":p["episodes"],"clips":p["clips"],"aligned_future_frames":d["aligned_future_frames"],"error_clips":d["error_clips"]}, sort_keys=True))' \
    "${comparison_root}/summary/diagnostic_metrics.json"
  log "COMBINED REPORT ${comparison_root}/summary/thought2_report.md"
}

log "START phase=${phase} branch=${current_git_branch} commit=$(git rev-parse HEAD)"
log "RUN_ROOT=${run_root} physical_gpu_ids=${gpu_ids_csv} world_size=${world_size}"

if [[ "${phase}" == "all" || "${phase}" == "diagnose" ]]; then
  ratify_cohorts
fi

if [[ "${phase}" == "all" || "${phase}" == "calibrate" ]]; then
  run_static_condition clean
  run_static_condition ood
  aggregate_static
fi

if [[ "${phase}" == "all" || "${phase}" == "diagnose" ]]; then
  threshold="$(formal_threshold)"
  threshold_sha256="$(
    sha256sum "${static_root}/combined/summary/static_calibration_summary.json" \
      | awk '{print $1}'
  )"
  log "STATIC THRESHOLD accepted=${threshold} summary_sha256=${threshold_sha256}"
  for suite in "${suites[@]}"; do
    run_diagnostic_condition clean "${suite}" "${threshold}"
  done
  for suite in "${suites[@]}"; do
    run_diagnostic_condition ood "${suite}" "${threshold}"
  done
  aggregate_diagnostics_all
fi

if [[ "${phase}" == "aggregate" ]]; then
  aggregate_static
  aggregate_diagnostics_all
fi

log "Thought 2 five-category phase '${phase}' finished successfully."
