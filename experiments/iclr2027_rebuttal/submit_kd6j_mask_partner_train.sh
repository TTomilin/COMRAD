#!/bin/bash
# ==============================================================================
# Submit the Reviewer kd6J "trained with the partner masked" runs to Snellius `gpu_mig` via `./comrad/send.sh 3`.
#
# Profile `mask_partner_training` (experiments/iclr2027_rebuttal/profiles.py): MAPPO Pixel-Partial trained from scratch with
# `--partner_occlusion_mode=both` (partner sprite replaced by background + partner-transmitted cues zeroed), i.e. the
# same intervention as the zero-shot Mask-partner evaluation, on all 13 scenarios x seeds {42, 68} at the 50M kd6J
# budget = 26 single-GPU jobs. Paired baselines are the kd6J Pixel-Partial MAPPO checkpoints (obs_modality_2x2,
# visibility_telemetry). The home quota is small, so checkpoints, sbatch files and SLURM logs are written to
# /scratch-shared/$HPC_USERNAME/ViZDoom/train_dir (COMRAD_TRAIN_DIR_ROOT).
#
# Usage (from local repo root):
#   experiments/iclr2027_rebuttal/submit_kd6j_mask_partner_train.sh [--dry-run] [seeds]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/comrad/.env"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

export COMRAD_EXPERIMENT_PROFILE="mask_partner_training"
export COMRAD_KD6J_SEEDS="${1:-42,68}"
export COMRAD_TRAIN_DIR_ROOT="${COMRAD_TRAIN_DIR_ROOT:-/scratch-shared/${HPC_USERNAME}/ViZDoom/train_dir}"
# 50M at ~10k FPS on a gpu_mig slice is ~1.4h; labels_buffer + per-frame masking add overhead, keep a wide margin.
export COMRAD_TIME_LIMIT="${COMRAD_TIME_LIMIT:-5:00:00}"

if [ "$PARTITION" != "gpu_mig" ] || [ "$CPUS_PER_TASK" != "9" ] || [ "$MEM" != "60G" ]; then
    echo "comrad/.env must set PARTITION=gpu_mig CPUS_PER_TASK=9 MEM=60G (got $PARTITION/$CPUS_PER_TASK/$MEM)"
    exit 1
fi

echo "[$(date +%T)] profile=$COMRAD_EXPERIMENT_PROFILE seeds=$COMRAD_KD6J_SEEDS train_dir_root=$COMRAD_TRAIN_DIR_ROOT time=$COMRAD_TIME_LIMIT"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run python -c '
from experiments.runs import RUN_DESCRIPTION, TOTAL_RUNS
for cmd, name, root, _ in RUN_DESCRIPTION.generate_experiments("TRAIN_DIR_ROOT", makedirs=False):
    print(f"  {root}/{name}")
print(f"  total runs: {TOTAL_RUNS}")
'

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

./comrad/send.sh 3
