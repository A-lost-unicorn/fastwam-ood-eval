#!/usr/bin/env bash
set -euo pipefail

environment_name="${1:-fastwam-ood}"
conda create -n "$environment_name" python=3.10 -y
# The Python Wand wrapper used by LIBERO-Plus needs the MagickWand shared library.
# tree is included so the documented structure check works inside the environment.
conda install -n "$environment_name" -c conda-forge --strict-channel-priority imagemagick tree -y
conda run -n "$environment_name" python -m pip install -U pip
conda run -n "$environment_name" python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
conda run -n "$environment_name" python -m pip install -e third_party/FastWAM
# Do not install both same-named libero packages. The evaluator selects one checkout through sys.path.
conda run -n "$environment_name" python -m pip install \
  mujoco==3.3.2 \
  robosuite==1.4.0 \
  bddl==1.0.1 \
  gym==0.25.2 \
  cloudpickle==2.1.0 \
  easydict==1.9 \
  future==0.18.2 \
  matplotlib==3.5.3 \
  robomimic==0.2.0 \
  thop==0.1.1.post2209072238 \
  opencv-python==4.11.0.86 \
  scikit-image \
  Wand \
  'usd-core>=25.5'
# OpenCV 5 requires NumPy 2, while FastWAM intentionally locks NumPy 1.26.
conda run -n "$environment_name" python -m pip install numpy==1.26.4 opencv-python==4.11.0.86
conda run -n "$environment_name" python -m pip install -e ".[dev]"
conda run -n "$environment_name" python -m pip check
echo "Created $environment_name. Activate it with: conda activate $environment_name"
