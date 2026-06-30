#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../comrad/.env"

HPC="${HPC_USERNAME}@${HPC_HOST}"

rsync -avz --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='train_dir' \
    --exclude='wandb' \
    --exclude='results' \
    "${SCRIPT_DIR}/../" "${HPC}:${HPC_PROJECT_DIR}/"

SCENARIOS=(
    "ammo_carrier:12000"
    "armory_siege:13000"
    "coop_health_gathering:14000"
    "coop_puzzle:16000"
    "dumb_enemies:18000"
    "foraging_commons:19000"
    "lava_maze:20000"
    "lavapit:22000"
    "platform_chain:24000"
    "rhythm_sync_dense:26000"
    "smart_enemies:28000"
    "stag_hunt_arena:29000"
    "stealth_labyrinth:30000"
)

for entry in "${SCENARIOS[@]}"; do
    SCENARIO="${entry%%:*}"
    PORT="${entry##*:}"

    echo "Submitting: $SCENARIO (port=$PORT) ..."

    JOB_ID=$(ssh "${HPC}" "cd ${HPC_PROJECT_DIR} && sbatch --parsable --export=SCENARIO=${SCENARIO},DOOM_DEFAULT_UDP_PORT=${PORT} --job-name=m_${SCENARIO} scripts/run_marker_eval.sh")

    echo "    Job ID: ${JOB_ID%%;*}"
    sleep 2
done

echo ""
echo "=== Queue ==="
ssh "${HPC}" 'squeue -u '"${HPC_USERNAME}"
