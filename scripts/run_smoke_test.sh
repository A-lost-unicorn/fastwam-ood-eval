#!/usr/bin/env bash
set -euo pipefail

python -m fastwam_ood_eval.cli doctor --config configs/eval_clean_smoke.yaml
python -m fastwam_ood_eval.cli plan --config configs/eval_clean_smoke.yaml
python -m fastwam_ood_eval.cli evaluate --config configs/eval_clean_smoke.yaml --device cuda:0
python -m fastwam_ood_eval.cli plan --config configs/eval_ood_smoke.yaml
python -m fastwam_ood_eval.cli evaluate --config configs/eval_ood_smoke.yaml --device cuda:0
python -m fastwam_ood_eval.cli aggregate --experiment-dir outputs/clean_smoke
python -m fastwam_ood_eval.cli aggregate --experiment-dir outputs/ood_smoke

