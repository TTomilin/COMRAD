#!/bin/bash
#SBATCH --job-name=c2k6_eval
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/%u/ViZDoom/results/rebuttal_c2k6/logs/c2k6_%A_%a.out
#SBATCH --error=/scratch-shared/%u/ViZDoom/results/rebuttal_c2k6/logs/c2k6_%A_%a.err
# ==============================================================================
# Reviewer C2K6 frozen-partner baseline (experiments/iclr2027_rebuttal/eval_frozen_partner.py) on Snellius `gpu_mig`.
#
# Manifest = the 9 reused 50M Pixel-Partial Table 3 winners (fixed below) + every run of the 100M
# `frozen_partner_winners` profile found under $SCRATCH_TRAIN_DIR (all candidate seeds; the single top checkpoint per
# scenario is chosen afterwards by experiments/iclr2027_rebuttal/generate_c2k6_frozen_partner_table.py on the `both_learned` pass).
# Each checkpoint: 5 held-out WADs x (20 both-learned + 4 x 10 seat-replaced) episodes in one live env per WAD.
# All outputs/logs go to /scratch-shared (home is at quota). Snellius gives every job a private TMPDIR, so the ViZDoom
# port reservation locks are moved to a node-keyed scratch dir (DOOM_PORT_LOCK_DIR) shared with the training jobs;
# each task additionally starts in its own 3000-port window above the training bases. Task i processes manifest
# lines i, i+N, ... A task exits non-zero if any evaluation failed; JSONs are written atomically and a cached JSON is
# only reused if it matches the protocol version, checkpoint run dir, WAD list and episode counts.
#
# Usage (on Snellius, after the frozen_partner_winners training finished):
#   mkdir -p /scratch-shared/$USER/ViZDoom/results/rebuttal_c2k6/logs
#   sbatch --array=0-2 experiments/iclr2027_rebuttal/c2k6_frozen_partner_array.sh
#   # reused winners only (no new training needed):
#   sbatch --array=0-1 --export=ALL,INCLUDE_NEW=0 experiments/iclr2027_rebuttal/c2k6_frozen_partner_array.sh
# ==============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/$USER/ViZDoom}"
SCRATCH_TRAIN_DIR="${SCRATCH_TRAIN_DIR:-$SCRATCH_ROOT/train_dir}"
OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/results/rebuttal_c2k6/heldout_runs}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/results/rebuttal_c2k6/logs}"
BATCH_ROOT="${BATCH_ROOT:-$SCRATCH_ROOT/results/heldout_wad_batches}"
BOTH_EPS="${BOTH_EPS:-20}"
SEAT_EPS="${SEAT_EPS:-10}"
INCLUDE_NEW="${INCLUDE_NEW:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${NUM_TASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
PARALLEL="${PARALLEL:-6}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$BATCH_ROOT"
# Reuse the exact held-out WADs of the kd6J evaluation (read from home, written to scratch only).
if [ -d "$REPO_DIR/results/heldout_wad_batches" ]; then
    rsync -a --ignore-existing "$REPO_DIR/results/heldout_wad_batches/" "$BATCH_ROOT/"
fi
cd "$REPO_DIR"
source "$REPO_DIR/.venv311/bin/activate"
if [ "$NUM_TASKS" -gt 5 ] || [ "$TASK_ID" -ge 5 ]; then
    echo "At most 5 array tasks fit into the evaluation port range (got task $TASK_ID of $NUM_TASKS)"; exit 1
fi
PORT_BASE=$((50000 + TASK_ID * 3000))
PORT_STRIDE=$((3000 / PARALLEL))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export DOOM_PORT_LOCK_DIR="/scratch-shared/$USER/doom_port_locks/$(hostname -s)"  # same dir as run_algo.sh
mkdir -p "$DOOM_PORT_LOCK_DIR"

# valid_json <json> <run_dir> <target_steps>: same checks as generate_c2k6_frozen_partner_table.validate_run.
valid_json() {
    python experiments/iclr2027_rebuttal/generate_c2k6_frozen_partner_table.py --validate-json "$1" --expect-run-dir "$2" \
        --expect-steps "$3" --both-episodes-per-wad "$BOTH_EPS" --seat-episodes-per-wad "$SEAT_EPS" > /dev/null 2>&1
}

TRAIN_HOME="$REPO_DIR/train_dir"
MANIFEST="$OUT_DIR/.manifest_${SLURM_ARRAY_JOB_ID:-local_$$}.tsv"
MANIFEST_TMP="$(mktemp "$OUT_DIR/.manifest_tmp_XXXXXX")"
python - "$TRAIN_HOME" "$SCRATCH_TRAIN_DIR" "$INCLUDE_NEW" > "$MANIFEST_TMP" <<'EOF'
import re, sys
from pathlib import Path
home, scratch, include_new = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
# run_dir (relative to home train_dir), scenario, algorithm, seed, target steps
fixed = [
    ("mode3_comrad_visibility_telemetry_20261109_171317/visibility_telemetry/00_MAPPO_pixel_partial_env_stag_hunt_arena_s.r.alp_0.0_see_42", "stag_hunt_arena", "MAPPO", 42),
    ("mode3_comrad_obs_modality_2x2_20261109_170851/obs_modality_2x2/01_MAPPO_pixel_partial_env_rhythm_sync_dense_s.r.alp_0.0_see_68", "rhythm_sync_dense", "MAPPO", 68),
    ("mode3_comrad_visibility_telemetry_20261109_171317/visibility_telemetry/00_MAPPO_pixel_partial_env_foraging_commons_s.r.alp_0.0_see_42", "foraging_commons", "MAPPO", 42),
    ("mode3_comrad_full_obs_clean_20261111_061449/full_obs_clean/00_MAPPO_pixel_partial_env_coop_puzzle_s.r.alp_0.0_see_68", "coop_puzzle", "MAPPO", 68),
    ("mode3_comrad_visibility_telemetry_20261109_171317/visibility_telemetry/00_MAPPO_pixel_partial_env_platform_chain_s.r.alp_0.0_see_42", "platform_chain", "MAPPO", 42),
    ("mode3_comrad_obs_modality_2x2_20261109_170851/obs_modality_2x2/01_QMIX_pixel_partial_env_armory_siege_see_68", "armory_siege", "QMIX", 68),
    ("mode3_comrad_visibility_telemetry_20261109_171317/visibility_telemetry/00_MAPPO_pixel_partial_env_lavapit_s.r.alp_0.0_see_42", "lavapit", "MAPPO", 42),
    ("mode3_comrad_visibility_telemetry_20261109_171317/visibility_telemetry/00_MAPPO_pixel_partial_env_smart_enemies_s.r.alp_0.0_see_42", "smart_enemies", "MAPPO", 42),
    ("mode3_comrad_visibility_telemetry_20261109_171317/visibility_telemetry/00_MAPPO_pixel_partial_env_lava_maze_s.r.alp_1.0_see_42", "lava_maze", "MAPPO", 42),
]
rows = []
for rel, scenario, algo, seed in fixed:
    run_dir = home / rel
    if not (run_dir / "checkpoint_p0").is_dir():
        sys.exit(f"missing reused checkpoint dir: {run_dir}")
    rows.append([str(run_dir), scenario, algo, str(seed), "50000000"])
if include_new:
    pat = re.compile(r"^\d+_(IPPO|QMIX)_pixel_partial_env_(.+?)_(?:s\.r\.alp_[\d.]+_)?see_(\d+)$")
    found = []
    for run_dir in sorted(scratch.glob("mode3_comrad_frozen_partner_winners_*/frozen_partner_winners/*")):
        m = pat.match(run_dir.name)
        if m and (run_dir / "checkpoint_p0").is_dir():
            found.append([str(run_dir), m.group(2), m.group(1), m.group(3), "100000000"])
    keys = [tuple(r[1:4]) for r in found]
    if len(keys) != len(set(keys)):
        sys.exit(f"duplicate frozen_partner_winners runs under {scratch}; move stale ones away first")
    rows.extend(found)
for r in rows:
    print("\t".join(r))
EOF
ln "$MANIFEST_TMP" "$MANIFEST" 2>/dev/null || true
rm -f "$MANIFEST_TMP"

i=0
FAILED=0
declare -a SLOT_PIDS=()
declare -A PID_RUN=()
# Collect the exit status of the child in slot $1 (bash keeps statuses of finished, unwaited children by PID).
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
while IFS=$'\t' read -r run_dir scenario algo seed target_steps; do
    if [ $((i % NUM_TASKS)) -eq "$TASK_ID" ]; then
        out_json="$OUT_DIR/${scenario}_${algo}_s${seed}.json"
        log_file="$LOG_DIR/${scenario}_${algo}_s${seed}.log"
        if [ -f "$out_json" ] && valid_json "$out_json" "$run_dir" "$target_steps"; then
            echo "[skip] $out_json"
        else
            slot=-1
            while [ "$slot" -lt 0 ]; do
                for s in $(seq 0 $((PARALLEL - 1))); do
                    pid="${SLOT_PIDS[$s]:-}"
                    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then collect_slot "$s"; slot=$s; break; fi
                done
                [ "$slot" -lt 0 ] && sleep 5
            done
            echo "[$(date +%T)] EVAL slot=$slot $run_dir -> $out_json"
            (
                export DOOM_DEFAULT_UDP_PORT=$((PORT_BASE + slot * PORT_STRIDE))
                python experiments/iclr2027_rebuttal/eval_frozen_partner.py \
                    --run-dir "$run_dir" --scenario "$scenario" --algorithm "$algo" --train-seed "$seed" \
                    --target-steps "$target_steps" --batch-root "$BATCH_ROOT" \
                    --both-episodes-per-wad "$BOTH_EPS" --seat-episodes-per-wad "$SEAT_EPS" \
                    --output-json "$out_json" --device gpu \
                    > "$log_file" 2>&1
                valid_json "$out_json" "$run_dir" "$target_steps"
            ) &
            SLOT_PIDS[$slot]=$!
            PID_RUN[$!]="$run_dir (see $log_file)"
        fi
    fi
    i=$((i + 1))
done < "$MANIFEST"
for s in $(seq 0 $((PARALLEL - 1))); do collect_slot "$s"; done
echo "[$(date +%T)] task $TASK_ID/$NUM_TASKS done ($i runs in manifest, $FAILED failed)"
[ "$FAILED" -eq 0 ]
