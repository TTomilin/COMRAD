#!/usr/bin/env bash
# One-off fresh deployment of COMRAD to the HPC in comrad/.env (e.g. snellius.surf.nl):
#   1) rsync the repo with the same excludes as comrad/send.sh
#   2) create ${HPC_PROJECT_DIR}/.venv311 (Python 3.11) and `uv sync` it on the login node
#   3) print a short import sanity check (torch, vizdoom, comrad env registry)
# Usage: ./scripts/snellius_setup.sh [--sync-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/comrad/.env"
REMOTE="${HPC_USERNAME}@${HPC_HOST}"

ssh "$REMOTE" "mkdir -p ${HPC_PROJECT_DIR} ${HPC_PROJECT_DIR}/train_dir"

rsync -az --exclude='.git' \
    --exclude='.hpc.bak/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='results/' \
    --exclude='eval_runs/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --exclude='*~' \
    --exclude='train_dir' \
    --exclude='wandb' \
    "$PROJECT_ROOT/" "${REMOTE}:${HPC_PROJECT_DIR}/"
echo "synced to ${REMOTE}:${HPC_PROJECT_DIR}"

if [[ "${1:-}" == "--sync-only" ]]; then
    exit 0
fi

ssh "$REMOTE" "bash -lc '
    set -e
    module purge
    command -v uv >/dev/null 2>&1 || module load uv
    cd ${HPC_PROJECT_DIR}
    export UV_CACHE_DIR=\${UV_CACHE_DIR:-\$HOME/.cache/uv}
    if [ ! -x .venv311/bin/python ]; then
        uv venv -p 3.11 .venv311
    fi
    source .venv311/bin/activate
    uv sync --active -p .venv311
    python -c \"import sys, torch, vizdoom; print(sys.version.split()[0], torch.__version__, vizdoom.__version__)\"
    python -c \"from comrad.utils.doom_utils import doom_env_by_name as d; [d(n) for n in (\\\"stealth_labyrinth_scripted_scout\\\", \\\"lava_maze_oracle_msg\\\")]; print(\\\"env registry ok\\\")\"
'"
