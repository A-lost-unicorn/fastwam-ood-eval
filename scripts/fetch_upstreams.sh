#!/usr/bin/env bash
set -euo pipefail

clone_or_report() {
  local url="$1"
  local target="$2"
  if [[ -d "$target/.git" ]]; then
    echo "$target already exists: $(git -C "$target" rev-parse HEAD)"
    return
  fi
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite non-Git path: $target" >&2
    exit 1
  fi
  git clone --depth 1 "$url" "$target"
  echo "$target cloned: $(git -C "$target" rev-parse HEAD)"
}

mkdir -p third_party
clone_or_report https://github.com/yuantianyuan01/FastWAM.git third_party/FastWAM
clone_or_report https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
clone_or_report https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus

