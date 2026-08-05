#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${project_root}/.conda/envs/fastwam-ood/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "Missing project interpreter: ${python_bin}" >&2
  exit 2
fi

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" scripts/analyze_thought5_pilot_v4_readonly.py "$@"
