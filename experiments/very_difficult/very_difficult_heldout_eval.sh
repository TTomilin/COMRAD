#!/bin/bash
#SBATCH --job-name=vd_eval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=train_dir/vd_eval_logs/slurm-%j.out
#SBATCH --error=train_dir/vd_eval_logs/slurm-%j.err
# ======================================================================
# Held-out evaluation of the very-difficult learnability references (App. very_difficult) under the Table 2
# protocol: 5 held-out layouts (seeds 1009,2027,3037,4051,5099) x EPISODES_PER_WAD (default 20) = 100 stochastic
# episodes per frozen checkpoint. Scripted-partner runs use the 50M-step checkpoint (single learner = same
# per-agent experience as the 100M two-learner frontier), oracle runs the 100M checkpoint.
#
# Each "env:algo:seed1,seed2,..." spec launches one evaluation process per training seed in parallel, each with its
# own DOOM_DEFAULT_UDP_PORT slot, writing results/heldout_very_difficult/<env>/<algo>/seed_<s>/summary.json.
# Aggregate afterwards with experiments/very_difficult/very_difficult_reference_table.py.
#
# Usage (on Snellius, from ~/ViZDoom):
#   sbatch experiments/very_difficult/very_difficult_heldout_eval.sh lava_maze_oracle_msg:MAPPO:42,68,81,97,154 [more specs...]
#   EPISODES_PER_WAD=1 OUT_ROOT=results/heldout_very_difficult_smoke sbatch experiments/very_difficult/very_difficult_heldout_eval.sh ...
# ======================================================================
set -u

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRIPTED_ARCHIVE="${SCRIPTED_ARCHIVE:-$REPO_DIR/train_dir/mode3_comrad_very_difficult_scripted_20260926_005603}"
CONVERSE_ARCHIVE="${CONVERSE_ARCHIVE:-$REPO_DIR/train_dir/mode3_comrad_very_difficult_converse_20261108_173747}"
ORACLE_ARCHIVE="${ORACLE_ARCHIVE:-$REPO_DIR/train_dir/mode3_comrad_very_difficult_oracle_20260926_010155}"
EPISODES_PER_WAD="${EPISODES_PER_WAD:-20}"
OUT_ROOT="${OUT_ROOT:-results/heldout_very_difficult}"
BATCH_ROOT="${BATCH_ROOT:-results/heldout_wad_batches}"

echo "Start: $(date) Node: ${SLURM_NODELIST:-local} JOB=${SLURM_JOB_ID:-none}"
module purge
if ! command -v uv &> /dev/null; then
    module load uv
fi
cd "$REPO_DIR"
source "$REPO_DIR/.venv311/bin/activate"
# concurrent ViZDoom peers race on creating the cwd-local ./_vizdoom dir and can segfault; create it up front
mkdir -p _vizdoom

PIDS=()
SLOT=0
# eval jobs can share a node: a fixed port base per slot made co-located jobs bind the same ports
# ("BindToPort: Address already in use" / "The game is full"), so offset the base by the job id.
# 10 job slots x 5 seed slots x 1000 ports, all below 65535.
JOB_BASE=$((10000 + (${SLURM_JOB_ID:-0} % 10) * 5000))
for spec in "$@"; do
    IFS=: read -r ENV_NAME ALGO SEEDS <<< "$spec"
    case "$ENV_NAME" in
        *_scripted_gunner|*_scripted_navigator) ARCHIVE="$CONVERSE_ARCHIVE"; TARGET=50000000 ;;
        *_scripted_*) ARCHIVE="$SCRIPTED_ARCHIVE"; TARGET=50000000 ;;
        *) ARCHIVE="$ORACLE_ARCHIVE"; TARGET=100000000 ;;
    esac
    for SEED in ${SEEDS//,/ }; do
        OUT="$OUT_ROOT/$ENV_NAME/$ALGO/seed_$SEED"
        mkdir -p "$OUT"
        PORT=$((JOB_BASE + (SLOT % 5) * 1000))
        SLOT=$((SLOT + 1))
        echo "launch $ENV_NAME $ALGO seed=$SEED target=$TARGET port=$PORT out=$OUT"
        DOOM_DEFAULT_UDP_PORT=$PORT python scripts/heldout_wad_evaluation.py \
            --archive-root "$ARCHIVE" --batch-root "$BATCH_ROOT" --output-dir "$OUT" \
            --scenarios "$ENV_NAME" --algorithms "$ALGO" --train-seeds "$SEED" \
            --target-steps "$TARGET" --episodes-per-wad "$EPISODES_PER_WAD" --device gpu \
            > "$OUT/eval.log" 2>&1 &
        PIDS+=("$!:$OUT")
        sleep 5
    done
done

FAILED=0
for entry in "${PIDS[@]}"; do
    PID="${entry%%:*}"
    OUT="${entry#*:}"
    if wait "$PID"; then
        echo "done $OUT: $(python -c "import json,sys; d=json.load(open(sys.argv[1]))['policy_summaries'][0]; print(d['checkpoint_steps'], round(d['mean_true_objective'], 4), d['episodes'])" "$OUT/summary.json")"
    else
        FAILED=$((FAILED + 1))
        echo "FAILED $OUT"
        tail -n 20 "$OUT/eval.log"
    fi
done
echo "End: $(date) failed=$FAILED"
