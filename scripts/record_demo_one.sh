#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <scenario> <experiment>"
  exit 2
fi

scenario="$1"
experiment="$2"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
benchmark_root="$repo_root/results/mode3_comrad_benchmark_20260503_134649"
train_dir="$benchmark_root/$scenario"
experiment_dir="$train_dir/$experiment"
checkpoint_dir="$experiment_dir/checkpoint_p0"
output_dir="$repo_root/results/demo"

if [[ ! -d "$experiment_dir" ]]; then
  echo "Missing experiment directory: $experiment_dir"
  exit 1
fi

if [[ ! -d "$checkpoint_dir" ]]; then
  echo "Missing checkpoint directory: $checkpoint_dir"
  exit 1
fi

if ! find "$checkpoint_dir" -maxdepth 1 -name 'best_*' -print -quit | grep -q .; then
  echo "Missing best checkpoint in: $checkpoint_dir"
  exit 1
fi

mkdir -p "$output_dir"

cd "$repo_root"

UV_CACHE_DIR=/tmp/uv-cache uv run python -m comrad.record_video \
  --env="$scenario" \
  --algo=IPPO \
  --train_dir="$train_dir" \
  --experiment="$experiment" \
  --device=cpu \
  --load_checkpoint_kind=best \
  --eval_deterministic=True \
  --output_dir="$output_dir" \
  --video_prefix=demo \
  --resolution=1280x720 \
  --video_fps=35 \
  --max_num_episodes=1 \
  --overwrite_video
