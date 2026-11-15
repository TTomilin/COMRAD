#!/bin/bash
#SBATCH --job-name=kd6j_eval
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=train_dir/kd6j_eval_logs/slurm-%j.out
#SBATCH --error=train_dir/kd6j_eval_logs/slurm-%j.err
# ==============================================================================
# Zero-shot evaluations for Reviewer kd6J on Snellius `gpu_mig`:
#   1. Uniform Random lower bound across all 13 scenarios (5 seeds x 5 held-out WADs x 20 eps)
#   2. Visibility & occlusion telemetry (`labels_buffer` + 3D frustum) on frozen 100M MAPPO & QMIX checkpoints
#   3. Partner-occlusion intervention (`--partner-occlusion-mode=both`) on frozen 100M checkpoints
#   4. Zero-shot blind policy evaluation (`--mode=blind_eval`) on frozen 100M checkpoints
#
# Usage (from repo root on Snellius):
#   sbatch experiments/iclr2027_rebuttal/submit_rebuttal_kd6j_eval.sh [random|visibility|partner_occlusion|blind_eval|all]
# ==============================================================================
set -euo pipefail

MODE="${1:-all}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$REPO_DIR/train_dir/comrad_benchmark}"
BATCH_ROOT="${BATCH_ROOT:-$REPO_DIR/results/heldout_wad_batches}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results/rebuttal_kd6j}"
EPISODES_PER_WAD="${EPISODES_PER_WAD:-20}"
SEEDS="${SEEDS:-42,68,81,97,154}"

mkdir -p "$REPO_DIR/train_dir/kd6j_eval_logs" "$OUT_DIR" "$REPO_DIR/_vizdoom"
cd "$REPO_DIR"

if [ -f "$REPO_DIR/.venv311/bin/activate" ]; then
    source "$REPO_DIR/.venv311/bin/activate"
elif [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi

PORT_BASE=$((12000 + (${SLURM_JOB_ID:-0} % 10) * 4000))
export DOOM_DEFAULT_UDP_PORT="$PORT_BASE"

run_mode() {
    local eval_mode="$1"
    local device_arg="$2"
    echo "[$(date +%T)] Running mode=$eval_mode on device=$device_arg (port=$DOOM_DEFAULT_UDP_PORT)"
    python -m experiments.iclr2027_rebuttal.rebuttal_kd6j_eval \
        --mode "$eval_mode" \
        --archive-root "$ARCHIVE_ROOT" \
        --batch-root "$BATCH_ROOT" \
        --output-dir "$OUT_DIR" \
        --train-seeds "$SEEDS" \
        --episodes-per-wad "$EPISODES_PER_WAD" \
        --device "$device_arg"
}

if [ "$MODE" = "all" ] || [ "$MODE" = "random" ]; then
    run_mode "random" "cpu"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "visibility" ]; then
    run_mode "visibility" "gpu"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "partner_occlusion" ]; then
    run_mode "partner_occlusion" "gpu"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "blind_eval" ]; then
    run_mode "blind_eval" "gpu"
fi

echo "[$(date +%T)] Completed Reviewer kd6J evaluations in $OUT_DIR"
