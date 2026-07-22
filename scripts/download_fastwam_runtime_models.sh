#!/usr/bin/env bash
set -euo pipefail

# Download the common Wan runtime components loaded by FastWAM before the
# release checkpoint is applied. ModelScope's snapshot API accepts branch/tag
# revisions here ("master"), not the commit SHA returned by model metadata.
# Content reproducibility is enforced with SHA-256 checks after downloading.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
destination_root="${1:-${project_root}/checkpoints}"
max_attempts="${MODELSCOPE_DOWNLOAD_MAX_ATTEMPTS:-20}"
retry_delay_seconds="${MODELSCOPE_DOWNLOAD_RETRY_DELAY_SECONDS:-5}"

if ! [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODELSCOPE_DOWNLOAD_MAX_ATTEMPTS must be a positive integer: ${max_attempts}" >&2
  exit 2
fi
if ! [[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "MODELSCOPE_DOWNLOAD_RETRY_DELAY_SECONDS must be a non-negative integer: ${retry_delay_seconds}" >&2
  exit 2
fi
if ! python -c 'from modelscope import snapshot_download' >/dev/null 2>&1; then
  echo "The ModelScope Python SDK is missing. Activate the project environment first." >&2
  exit 2
fi

mkdir -p "${destination_root}"

download_snapshot() {
  local model_id="$1"
  local destination="$2"
  shift 2
  local patterns=("$@")
  local attempt=1

  while true; do
    echo "Downloading ${model_id} (attempt ${attempt}/${max_attempts})"
    if python - "${model_id}" "${destination}" "${patterns[@]}" <<'PY'
import sys

from modelscope import snapshot_download

model_id, destination, *patterns = sys.argv[1:]
snapshot_download(
    model_id=model_id,
    revision="master",
    local_dir=destination,
    allow_file_pattern=patterns,
    max_workers=1,
)
PY
    then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      echo "Download failed after ${max_attempts} attempts; partial files were kept for resume." >&2
      return 1
    fi

    echo "Download failed; retrying in ${retry_delay_seconds}s..." >&2
    attempt=$((attempt + 1))
    sleep "${retry_delay_seconds}"
  done
}

verify_file() {
  local expected_hash="$1"
  local file="$2"
  local actual_hash

  if [[ ! -f "${file}" ]]; then
    echo "Downloaded file is missing: ${file}" >&2
    return 1
  fi

  actual_hash="$(sha256sum "${file}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "SHA-256 mismatch: ${file}" >&2
    echo "  expected: ${expected_hash}" >&2
    echo "  actual:   ${actual_hash}" >&2
    return 1
  fi
  printf '%s  %s\n' "${actual_hash}" "${file}"
}

converted_dir="${destination_root}/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
tokenizer_dir="${destination_root}/Wan-AI/Wan2.1-T2V-1.3B"

# Fetch the small tokenizer first so revision and connectivity problems fail
# before either multi-gigabyte weight begins transferring.
download_snapshot \
  "Wan-AI/Wan2.1-T2V-1.3B" \
  "${tokenizer_dir}" \
  "google/umt5-xxl/special_tokens_map.json" \
  "google/umt5-xxl/spiece.model" \
  "google/umt5-xxl/tokenizer.json" \
  "google/umt5-xxl/tokenizer_config.json"

download_snapshot \
  "DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
  "${converted_dir}" \
  "models_t5_umt5-xxl-enc-bf16.safetensors" \
  "Wan2.2_VAE.safetensors"

echo "Verifying FastWAM runtime components"
verify_file \
  "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23" \
  "${converted_dir}/models_t5_umt5-xxl-enc-bf16.safetensors"
verify_file \
  "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996" \
  "${converted_dir}/Wan2.2_VAE.safetensors"
verify_file \
  "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d" \
  "${tokenizer_dir}/google/umt5-xxl/special_tokens_map.json"
verify_file \
  "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458" \
  "${tokenizer_dir}/google/umt5-xxl/spiece.model"
verify_file \
  "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b" \
  "${tokenizer_dir}/google/umt5-xxl/tokenizer.json"
verify_file \
  "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05" \
  "${tokenizer_dir}/google/umt5-xxl/tokenizer_config.json"

echo "FastWAM runtime model download and verification complete."
