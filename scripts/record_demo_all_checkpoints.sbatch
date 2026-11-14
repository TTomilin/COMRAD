#!/bin/bash
#SBATCH --job-name=record_demo_all
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=12:12:00
#SBATCH --output=train_dir/record_demo_all_checkpoints_%j.out
#SBATCH --error=train_dir/record_demo_all_checkpoints_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

module purge
if ! command -v uv &>/dev/null; then
  module load uv
fi

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv311/bin/activate"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PORT_SLOT_COUNT=4
PORT_SLOT_SIZE=13000
PORT_BASE_START=12000
PORT_SLOT=$((${SLURM_JOB_ID:-0} % PORT_SLOT_COUNT))
export DOOM_DEFAULT_UDP_PORT=$((PORT_BASE_START + PORT_SLOT * PORT_SLOT_SIZE))
echo "DOOM_DEFAULT_UDP_PORT=${DOOM_DEFAULT_UDP_PORT}"

DEVICE="${DEVICE:-gpu}" ./scripts/record_demo_all_checkpoints.sh
