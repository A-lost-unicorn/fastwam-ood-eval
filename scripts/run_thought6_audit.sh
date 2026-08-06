#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${project_root}/.conda/envs/fastwam-ood/bin/python" -m fastwam_ood_eval.cli \
  thought6-audit --config configs/thought6/phase6_audit.yaml "$@"
