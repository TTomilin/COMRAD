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
# Held-out evaluation of the Reviewer kd6J minimal design (81 runs = 2 seeds {42,68}, 50M steps):
#   - every run: 5 held-out WADs x EPISODES_PER_WAD (default 20 -> 100 episodes)
#   - Pixel-Partial runs additionally get partner-occlusion + zero-shot blind interventions.
# Inference is light and evals are CPU-bound (ViZDoom), so each task takes one gpu_mig slice (9 CPUs / 60G) and
# runs PARALLEL evals concurrently. Only unfinished runs enter the manifest (heaviest Pixel-Partial runs first);
# the task of rank r in an N-task array job (array_task_rank.sh) processes manifest lines r, r+N, ...
# A JSON only counts as done if it is complete and written by the current protocol (kd6j_protocol.py).
# A task exits non-zero if any evaluation failed or wrote an incomplete JSON.
#
# Usage (on Snellius, after training finished):
#   sbatch --array=0-3 experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh
# Clean Full-vs-Partial follow-up (3 seeds; only unfinished runs are evaluated):
#   sbatch --partition=gpu_mig --cpus-per-task=9 --mem=60G --array=0-1 \
#       --export=ALL,KD6J_SEEDS=42:68:81,KD6J_PROFILES=full_obs_clean:visibility_telemetry,PARALLEL=6 \
#       experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh
#   python -m experiments.iclr2027_rebuttal.aggregate_kd6j_results --runs-dir results/rebuttal_kd6j/heldout_runs --min-steps 45000000
# True full observability (global state, 3 seeds):
#   sbatch --partition=gpu_mig --cpus-per-task=9 --mem=60G --array=0 \
#       --export=ALL,KD6J_SEEDS=42:68:81,KD6J_PROFILES=full_obs_global,PARALLEL=6 experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh
# True full observability v2 (global state v2, layout-pool training, 100M, Table 3 seeds; runs live on /scratch-shared).
# results/ is not synced by comrad/send.sh: rsync results/heldout_wad_batches/<scenario> first (Co-op Puzzle was
# regenerated with the USER7-14 state export), then compare Full against MAPPO in tab:baseline_results:
#   sbatch --partition=gpu_mig --cpus-per-task=9 --mem=60G --array=0-2 \
#       --export=ALL,KD6J_SEEDS=42:68:81:97:154,KD6J_PROFILES=full_obs_global_v2,TARGET_STEPS=100000000,PARALLEL=5,TRAIN_ROOT=/scratch-shared/$USER/ViZDoom/train_dir \
#       experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh
#   python -m experiments.iclr2027_rebuttal.aggregate_kd6j_results --runs-dir results/rebuttal_kd6j/heldout_runs --min-steps 95000000
# Entity state vs Pixel-Partial (Table tab:po_state). A separate OUT_DIR, since its seed-81 Pixel-Partial runs share
# output names with the full_obs_clean ones; interventions are not needed for this table:
#   sbatch --partition=gpu_mig --cpus-per-task=9 --mem=60G --array=0 \
#       --export=ALL,KD6J_SEEDS=42:68:81,KD6J_PROFILES=state_entity,RUN_INTERVENTIONS=0,PARALLEL=6,OUT_DIR=$PWD/results/rebuttal_kd6j/heldout_runs_state_entity \
#       experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh
# ==============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Where the mode3_comrad_<profile>_<timestamp> run dirs are (full_obs_global_v2 trained under /scratch-shared).
TRAIN_ROOT="${TRAIN_ROOT:-$REPO_DIR/train_dir}"
EPISODES_PER_WAD="${EPISODES_PER_WAD:-20}"
TARGET_STEPS="${TARGET_STEPS:-50000000}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results/rebuttal_kd6j/heldout_runs}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_SPLIT="$(bash "$REPO_DIR/experiments/iclr2027_rebuttal/array_task_rank.sh")"
read -r TASK_RANK NUM_TASKS <<< "$TASK_SPLIT"
PARALLEL="${PARALLEL:-6}"
RUN_INTERVENTIONS="${RUN_INTERVENTIONS:-1}"
STALE_LOG_MIN="${STALE_LOG_MIN:-15}"
KD6J_SEEDS="${KD6J_SEEDS:-42,68}"
# Restrict which training profiles are scanned (the cancelled seed-81 dirs of obs_modality_2x2 would otherwise map to the
# same output names as the full_obs_clean seed-81 Pixel-Partial runs).
KD6J_PROFILES="${KD6J_PROFILES:-obs_modality_2x2,blind_ablation,visibility_telemetry,full_obs_clean}"
# sbatch --export splits on commas, so lists may also be passed colon-separated (e.g. KD6J_SEEDS=42:68:81).
KD6J_SEEDS="${KD6J_SEEDS//:/,}"
KD6J_PROFILES="${KD6J_PROFILES//:/,}"

mkdir -p "$REPO_DIR/train_dir/kd6j_eval_logs" "$OUT_DIR"
cd "$REPO_DIR"
source "$REPO_DIR/.venv311/bin/activate"
PORT_BASE=$((12000 + (TASK_ID % 8) * 6000))
# Each slot gets 6000 / PARALLEL ports so slots never spill into another task's port range.
PORT_STRIDE=$((6000 / PARALLEL))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The manifest is frozen once per array job: tasks start at different times, and re-listing unfinished runs
# after other tasks completed some would shift the i % N split and silently drop runs.
MANIFEST="$OUT_DIR/.manifest_${SLURM_ARRAY_JOB_ID:-local_$$}.tsv"
MANIFEST_TMP="$(mktemp "$OUT_DIR/.manifest_tmp_XXXXXX")"
python - "$TRAIN_ROOT" "$KD6J_SEEDS" "$OUT_DIR" "$KD6J_PROFILES" > "$MANIFEST_TMP" <<'EOF'
import re, sys
from pathlib import Path
from experiments.iclr2027_rebuttal.kd6j_protocol import is_current_result_file
root = Path(sys.argv[1])
seeds = {s.strip() for s in sys.argv[2].split(",") if s.strip()}
out_dir = Path(sys.argv[3])
rows = []
pat = re.compile(r"^\d+_(MAPPO|QMIX)_(pixel_partial|pixel_full_clean|pixel_full_global_v2|pixel_full_global|pixel_full|state_partial|state_full|state_entity|blind)_env_(.+?)_(?:s\.r\.alp_[\d.]+_)?see_(\d+)$")
for prof in [p.strip() for p in sys.argv[4].split(",") if p.strip()]:
    # The suffix is the launch timestamp (20YYMMDD_...), so profile full_obs_global does not also pick up full_obs_global_v2.
    for workdir in sorted(root.glob(f"mode3_comrad_{prof}_2*")):
        for run_dir in sorted(p for p in workdir.rglob("*") if p.is_dir() and pat.match(p.name)):
            m = pat.match(run_dir.name)
            done = is_current_result_file(out_dir / f"{m.group(3)}_{m.group(1)}_{m.group(2)}_s{m.group(4)}.json")
            if m.group(4) in seeds and (run_dir / "checkpoint_p0").is_dir() and not done:
                rows.append([str(run_dir), m.group(3), m.group(1), m.group(2), m.group(4)])
# Pixel-Partial runs also do 2 interventions (3x episodes): schedule them first to shorten the tail.
rows.sort(key=lambda r: r[3] != "pixel_partial")
for r in rows:
    print("\t".join(r))
EOF
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
while IFS=$'\t' read -r run_dir scenario algo cell seed; do
    if [ $((i % NUM_TASKS)) -eq "$TASK_RANK" ]; then
        out_json="$OUT_DIR/${scenario}_${algo}_${cell}_s${seed}.json"
        log_file="$OUT_DIR/${scenario}_${algo}_${cell}_s${seed}.log"
        if result_is_current "$out_json"; then
            echo "[skip] $out_json"
        elif [ -n "$(find "$log_file" -mmin -"$STALE_LOG_MIN" 2>/dev/null)" ]; then
            # Log written recently but no JSON yet: another task is still evaluating this run.
            echo "[skip-in-progress] $out_json"
        else
            flags=()
            if [ "$cell" = "pixel_partial" ] && [ "$RUN_INTERVENTIONS" = "1" ]; then
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
                    --train-seed "$seed" --episodes-per-wad "$EPISODES_PER_WAD" --target-steps "$TARGET_STEPS" \
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
