#!/bin/bash
#SBATCH --job-name=record_demo_all
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=12:12:00
#SBATCH --output=/home/knguyen2/ViZDoom/train_dir/record_demo_all_checkpoints_%j.out
#SBATCH --error=/home/knguyen2/ViZDoom/train_dir/record_demo_all_checkpoints_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/knguyen2/ViZDoom}"
RUN_DIR="${PROJECT_DIR}/train_dir/record_demo_all_checkpoints_j${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

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
PORT_SLOT=$((SLURM_JOB_ID % PORT_SLOT_COUNT))
export DOOM_DEFAULT_UDP_PORT=$((PORT_BASE_START + PORT_SLOT * PORT_SLOT_SIZE))
echo "DOOM_DEFAULT_UDP_PORT=${DOOM_DEFAULT_UDP_PORT}"

DEVICE="${DEVICE:-gpu}" ./scripts/record_demo_all_checkpoints.sh
