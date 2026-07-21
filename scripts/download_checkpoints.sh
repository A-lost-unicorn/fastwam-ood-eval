#!/usr/bin/env bash
set -euo pipefail

destination="${1:-checkpoints/fastwam_release}"
mkdir -p "$destination"
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir "$destination"
echo "Checkpoint files downloaded to $destination"

