#!/bin/bash
#SBATCH --job-name=kd6j_eval
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=train_dir/kd6j_eval_logs/heldout_%A_%a.out
#SBATCH --error=train_dir/kd6j_eval_logs/heldout_%A_%a.err
# ==============================================================================
# Held-out evaluation of one kd6J evaluation set (kd6j_manifest.py: kd6j_50m, full_obs_global_v2, state_entity).
# The manifest module selects the runs, output JSONs, step targets and interventions, and aborts on a missing or
# duplicated run before any evaluation starts. Each task takes one gpu_mig slice and runs PARALLEL evals; the task of
# rank r in an N-task array job (array_task_rank.sh) processes manifest lines r, r+N, ...
# A JSON only counts as done if it is complete and written by the current protocol (kd6j_protocol.py).
# A task exits non-zero if any evaluation failed or wrote an incomplete JSON.
#
#   sbatch --array=0-3 --export=ALL,EVAL_SET=kd6j_50m experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh
# ==============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EVAL_SET="${EVAL_SET:?set EVAL_SET to an evaluation set of kd6j_manifest.py}"
EPISODES_PER_WAD="${EPISODES_PER_WAD:-20}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_SPLIT="$(bash "$REPO_DIR/experiments/iclr2027_rebuttal/array_task_rank.sh")"
read -r TASK_RANK NUM_TASKS <<< "$TASK_SPLIT"
PARALLEL="${PARALLEL:-6}"
STALE_LOG_MIN="${STALE_LOG_MIN:-15}"
LOG_DIR="$REPO_DIR/train_dir/kd6j_eval_logs"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
source "$REPO_DIR/.venv311/bin/activate"
PORT_BASE=$((12000 + (TASK_ID % 8) * 6000))
# Each slot gets 6000 / PARALLEL ports so slots never spill into another task's port range.
PORT_STRIDE=$((6000 / PARALLEL))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The manifest is frozen once per array job: tasks start at different times, and re-listing unfinished runs
# after other tasks completed some would shift the i % N split and silently drop runs.
MANIFEST="$LOG_DIR/.manifest_${EVAL_SET}_${SLURM_ARRAY_JOB_ID:-local_$$}.tsv"
MANIFEST_TMP="$(mktemp "$LOG_DIR/.manifest_tmp_XXXXXX")"
python -m experiments.iclr2027_rebuttal.kd6j_manifest --eval-set "$EVAL_SET" --pending-only > "$MANIFEST_TMP"
# Hard link is atomic and fails if the manifest exists, so the first task to start wins.
ln "$MANIFEST_TMP" "$MANIFEST" 2>/dev/null || true
rm -f "$MANIFEST_TMP"

result_is_current() {
    python -m experiments.iclr2027_rebuttal.kd6j_protocol "$1"
}

i=0
FAILED=0
declare -a SLOT_PIDS=()
declare -A PID_RUN=()
collect_slot() {
    local pid="${SLOT_PIDS[$1]:-}" status=0
    [ -z "$pid" ] && return 0
    wait "$pid" || status=$?
    if [ "$status" -ne 0 ]; then
        echo "[FAIL] ${PID_RUN[$pid]:-pid $pid} (exit $status)"
        FAILED=$((FAILED + 1))
    fi
    SLOT_PIDS[$1]=""
}
while IFS=$'\t' read -r run_dir scenario algo cell seed interventions target_steps out_json; do
    if [ $((i % NUM_TASKS)) -eq "$TASK_RANK" ]; then
        log_file="${out_json%.json}.log"
        if result_is_current "$out_json"; then
            echo "[skip] $out_json"
        elif [ -n "$(find "$log_file" -mmin -"$STALE_LOG_MIN" 2>/dev/null)" ]; then
            # Log written recently but no JSON yet: another task is still evaluating this run.
            echo "[skip-in-progress] $out_json"
        else
            flags=()
            if [ "$interventions" = "1" ]; then
                flags+=("--run-interventions")
            fi
            # Take a slot whose previous eval has exited, so concurrent evals never share a port range.
            slot=-1
            while [ "$slot" -lt 0 ]; do
                for s in $(seq 0 $((PARALLEL - 1))); do
                    pid="${SLOT_PIDS[$s]:-}"
                    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then collect_slot "$s"; slot=$s; break; fi
                done
                if [ "$slot" -lt 0 ]; then sleep 5; fi
            done
            echo "[$(date +%T)] EVAL slot=$slot $run_dir -> $out_json"
            (
                export DOOM_DEFAULT_UDP_PORT=$((PORT_BASE + slot * PORT_STRIDE))
                python -m experiments.iclr2027_rebuttal.eval_kd6j_checkpoint_pack \
                    --run-dir "$run_dir" --scenario "$scenario" --algorithm "$algo" --cell-tag "$cell" \
                    --train-seed "$seed" --episodes-per-wad "$EPISODES_PER_WAD" --target-steps "$target_steps" \
                    --output-json "$out_json" --device gpu "${flags[@]}" \
                    > "$log_file" 2>&1
                result_is_current "$out_json"
            ) &
            SLOT_PIDS[$slot]=$!
            PID_RUN[$!]="$run_dir (see $log_file)"
        fi
    fi
    i=$((i + 1))
done < "$MANIFEST"
for s in $(seq 0 $((PARALLEL - 1))); do collect_slot "$s"; done
echo "[$(date +%T)] task $TASK_ID (rank $TASK_RANK/$NUM_TASKS) done ($i runs in manifest, $FAILED failed)"
[ "$FAILED" -eq 0 ]
