#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_THOUGHT3_E9_V21_AUDIT:-}" != "YES" ]]; then
  echo "Refusing formal E.9a-v2.1 audit: set CONFIRM_THOUGHT3_E9_V21_AUDIT=YES" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "E.9a-v2.1 is CPU-only; unset CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${project_root}/.conda/envs/fastwam-ood/bin/python" -m fastwam_ood_eval.cli \
  thought3-audit-e9-v2-artifacts \
  --config configs/thought3/audits/phase_e9_v2_1_readonly_audit.yaml \
  "$@"
