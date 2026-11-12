#!/bin/bash
# ==============================================================================
# Submit the Reviewer C2K6 frozen-partner winners to Snellius `gpu_mig` via `./comrad/send.sh 3`.
#
# Profile `frozen_partner_winners` (experiments/iclr2027_rebuttal/profiles.py) trains the Table 3 winning methods that have no Pixel-Partial
# checkpoint on Snellius yet, at the 100M Table 3 budget:
#   - IPPO (alpha=0): coop_health_gathering, dumb_enemies, ammo_carrier   x seeds {42, 68}
#   - QMIX (canonical replay buffer): stealth_labyrinth                   x seeds {42, 68, 81}
# = 9 single-GPU jobs. The home quota is small, so checkpoints, sbatch files and SLURM logs are written to
# /scratch-shared/$HPC_USERNAME/ViZDoom/train_dir (COMRAD_TRAIN_DIR_ROOT). WandB is disabled by the profile.
#
# Usage (from local repo root):
#   experiments/iclr2027_rebuttal/submit_c2k6_frozen_partner_train.sh [--dry-run]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/comrad/.env"

export COMRAD_EXPERIMENT_PROFILE="frozen_partner_winners"
export COMRAD_TRAIN_DIR_ROOT="${COMRAD_TRAIN_DIR_ROOT:-/scratch-shared/${HPC_USERNAME}/ViZDoom/train_dir}"
# ~10k FPS on a gpu_mig slice with the benchmark worker layout -> ~2.8h for 100M; keep margin for startup.
export COMRAD_TIME_LIMIT="${COMRAD_TIME_LIMIT:-5:00:00}"

if [ "$PARTITION" != "gpu_mig" ] || [ "$CPUS_PER_TASK" != "9" ] || [ "$MEM" != "60G" ]; then
    echo "comrad/.env must set PARTITION=gpu_mig CPUS_PER_TASK=9 MEM=60G (got $PARTITION/$CPUS_PER_TASK/$MEM)"
    exit 1
fi

echo "[$(date +%T)] profile=$COMRAD_EXPERIMENT_PROFILE train_dir_root=$COMRAD_TRAIN_DIR_ROOT time=$COMRAD_TIME_LIMIT"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run python -c '
from experiments.runs import RUN_DESCRIPTION, TOTAL_RUNS
for cmd, name, root, _ in RUN_DESCRIPTION.generate_experiments("TRAIN_DIR_ROOT", makedirs=False):
    print(f"  {root}/{name}")
print(f"  total runs: {TOTAL_RUNS}")
'

if [ "${1:-}" = "--dry-run" ]; then
    exit 0
fi

./comrad/send.sh 3
