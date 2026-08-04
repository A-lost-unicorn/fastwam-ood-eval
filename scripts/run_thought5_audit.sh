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

"${python_bin}" -m fastwam_ood_eval.cli \
  thought5-audit \
  --config configs/thought5/phase5_audit_v2.yaml

"${python_bin}" -m fastwam_ood_eval.cli \
  thought5-dry-run \
  --config configs/thought5/phase5_formal_v2.yaml
