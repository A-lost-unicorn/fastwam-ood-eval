#!/usr/bin/env bash
set -euo pipefail

python -m fastwam_ood_eval.cli doctor --config configs/eval_clean_smoke.yaml
pytest -q

