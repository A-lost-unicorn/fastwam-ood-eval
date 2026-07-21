#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/eval_clean_full.yaml}"
python -m fastwam_ood_eval.cli plan --config "$config"
python -m fastwam_ood_eval.cli evaluate --config "$config" --device cuda:0

