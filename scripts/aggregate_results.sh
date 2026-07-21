#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 outputs/<experiment>" >&2
  exit 2
fi
python -m fastwam_ood_eval.cli aggregate --experiment-dir "$1"

