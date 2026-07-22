#!/usr/bin/env bash
set -euo pipefail

# Download only the public LIBERO-Plus assets archive. The default endpoint is
# an unofficial regional mirror, scoped to this process rather than the whole
# Conda environment. Override HF_ENDPOINT to use the official Hub or another
# trusted endpoint.
destination="${1:-third_party/LIBERO-plus/.downloads}"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
max_attempts="${HF_DOWNLOAD_MAX_ATTEMPTS:-20}"
retry_delay_seconds="${HF_DOWNLOAD_RETRY_DELAY_SECONDS:-5}"
bypass_proxy="${FASTWAM_HF_BYPASS_PROXY:-1}"

if ! [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HF_DOWNLOAD_MAX_ATTEMPTS must be a positive integer: ${max_attempts}" >&2
  exit 2
fi
if ! [[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "HF_DOWNLOAD_RETRY_DELAY_SECONDS must be a non-negative integer: ${retry_delay_seconds}" >&2
  exit 2
fi
if [[ "${bypass_proxy}" != "0" && "${bypass_proxy}" != "1" ]]; then
  echo "FASTWAM_HF_BYPASS_PROXY must be 0 or 1: ${bypass_proxy}" >&2
  exit 2
fi
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is missing. Activate the project environment first." >&2
  exit 2
fi

mkdir -p "${destination}"

export HF_ENDPOINT="${endpoint}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# The archive is public. Do not forward a token saved by `huggingface-cli login`
# to an unofficial mirror unless the caller deliberately overrides this value.
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"

if [[ "${endpoint}" == "https://hf-mirror.com"* && "${bypass_proxy}" == "1" ]]; then
  mirror_no_proxy="hf-mirror.com,.hf-mirror.com"
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${mirror_no_proxy}"
  export no_proxy="${no_proxy:+${no_proxy},}${mirror_no_proxy}"
  unset mirror_no_proxy
fi

echo "Downloading public LIBERO-Plus assets from ${HF_ENDPOINT}"
echo "Destination: ${destination}"
echo "Existing .incomplete files are retained for automatic resume."

attempt=1
while true; do
  if huggingface-cli download Sylvest/LIBERO-plus \
    assets.zip \
    --repo-type dataset \
    --local-dir "${destination}" \
    --max-workers 1; then
    break
  fi

  if (( attempt >= max_attempts )); then
    echo "Download failed after ${max_attempts} attempts; partial data was kept for resume." >&2
    exit 1
  fi

  echo "Download attempt ${attempt}/${max_attempts} failed; retrying in ${retry_delay_seconds}s..." >&2
  attempt=$((attempt + 1))
  sleep "${retry_delay_seconds}"
done

archive="${destination}/assets.zip"
if [[ ! -f "${archive}" ]]; then
  echo "Download command finished but archive is missing: ${archive}" >&2
  exit 1
fi

ls -lh "${archive}"
sha256sum "${archive}"
echo "Download complete. Do not extract into an existing assets/ subdirectory."

