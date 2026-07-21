#!/usr/bin/env bash
set -euo pipefail

environment_name="${1:-fastwam-ood}"
conda create -n "$environment_name" python=3.10 -y
conda run -n "$environment_name" python -m pip install -U pip
conda run -n "$environment_name" python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
conda run -n "$environment_name" python -m pip install -e third_party/FastWAM
# Do not install both same-named libero packages. The evaluator selects one checkout through sys.path.
conda run -n "$environment_name" python -m pip install mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 cloudpickle==2.1.0 easydict==1.9 scikit-image Wand 'usd-core>=25.5'
conda run -n "$environment_name" python -m pip install -e ".[dev]"
echo "Created $environment_name. Activate it with: conda activate $environment_name"
