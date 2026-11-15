#!/bin/bash
#SBATCH --job-name=kd6j_mig
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=train_dir/kd6j_eval_logs/pack_%a_%j.out
#SBATCH --error=train_dir/kd6j_eval_logs/pack_%a_%j.err
# ==============================================================================
# Packed Snellius `gpu_mig` runner for Reviewer kd6J experiments.
# Designed for SLURM array `--array=0-7` on `gpu_mig` (each task gets a disjoint
# 6000-port UDP window so co-located MIG slices on gcn[2-5] never collide).
# ==============================================================================
set -euo pipefail

PACK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
TRAIN_SECONDS="${TRAIN_SECONDS:-22}"
EVAL_EPS_PER_WAD="${EVAL_EPS_PER_WAD:-1}"
SEEDS_STR="${SEEDS_STR:-42,68,81}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results/rebuttal_kd6j/mig_runs}"
TRAIN_ROOT="${TRAIN_ROOT:-$REPO_DIR/train_dir/kd6j_mig_pack_${PACK_ID}}"

mkdir -p "$REPO_DIR/train_dir/kd6j_eval_logs" "$OUT_DIR" "$TRAIN_ROOT" "$REPO_DIR/_vizdoom"
cd "$REPO_DIR"

module purge 2>/dev/null || true
source "$REPO_DIR/.venv311/bin/activate"

export DOOM_DEFAULT_UDP_PORT=$((12000 + PACK_ID * 6000))
echo "[$(date +%T)] PACK_ID=$PACK_ID Node=${SLURM_NODELIST:-local} UDP_PORT=$DOOM_DEFAULT_UDP_PORT"

run_train_and_eval() {
    local env_name="$1"
    local algo="$2"
    local cell_tag="$3"
    local obs_modality="$4"
    local full_obs="$5"
    local seed="$6"
    local run_interventions="${7:-0}"

    local out_json="$OUT_DIR/${env_name}_${algo}_${cell_tag}_s${seed}.json"
    if [ -f "$out_json" ]; then
        echo "[skip] $out_json already exists"
        return 0
    fi

    local exp_name="${algo}_${cell_tag}_env_${env_name}_see_${seed}"
    local exp_dir="$TRAIN_ROOT/$exp_name"

    local extra_flags=(
        "--set" "seed=$seed"
        "--set" "train_for_seconds=$TRAIN_SECONDS"
        "--set" "save_every_sec=5"
        "--set" "keep_checkpoints=2"
        "--set" "num_workers=6"
        "--set" "num_envs_per_worker=4"
        "--set" "policy_workers_per_policy=1"
        "--set" "batch_size=1024"
        "--set" "with_wandb=False"
        "--set" "wandb_record_every=0"
        "--set" "obs_modality=$obs_modality"
        "--set" "full_observability=$full_obs"
        "--set" "experiment=$exp_name"
        "--set" "train_dir=$TRAIN_ROOT"
    )
    if [ "$obs_modality" = "state" ]; then
        extra_flags+=("--set" "obs_scale=1.0")
    fi
    if [ "$algo" = "QMIX" ]; then
        extra_flags+=(
            "--set" "learning_starts=2048"
            "--set" "qmix_sequence_batch_size=16"
            "--set" "replay_buffer_size=9042"
        )
    fi

    local cmd_line
    local -a cmd
    cmd_line=$(python -m experiments.very_difficult.very_difficult_cmd --env "$env_name" --algo "$algo" "${extra_flags[@]}")
    read -r -a cmd <<< "$cmd_line"
    echo "[$(date +%T)] TRAIN $exp_name"
    timeout --kill-after=20 $((TRAIN_SECONDS + 120)) "${cmd[@]}" > "$TRAIN_ROOT/${exp_name}_train.log" 2>&1 || true

    if ! ls "$exp_dir"/checkpoint_p0/*.pth >/dev/null 2>&1; then
        echo "[WARN] No checkpoint found in $exp_dir/checkpoint_p0, tailing log:"
        tail -n 25 "$TRAIN_ROOT/${exp_name}_train.log" || true
        return 1
    fi

    local eval_flags=(
        "--run-dir" "$exp_dir"
        "--scenario" "$env_name"
        "--algorithm" "$algo"
        "--cell-tag" "$cell_tag"
        "--train-seed" "$seed"
        "--episodes-per-wad" "$EVAL_EPS_PER_WAD"
        "--output-json" "$out_json"
        "--device" "gpu"
    )
    if [ "$run_interventions" = "1" ]; then
        eval_flags+=("--run-interventions")
    fi

    echo "[$(date +%T)] EVAL $exp_name -> $out_json"
    python -m experiments.iclr2027_rebuttal.eval_kd6j_checkpoint_pack "${eval_flags[@]}" > "$TRAIN_ROOT/${exp_name}_eval.log" 2>&1
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_STR"

case "$PACK_ID" in
    0|1|2|3)
        REP_ENVS=("rhythm_sync_dense" "armory_siege" "ammo_carrier" "stealth_labyrinth")
        ENV_NAME="${REP_ENVS[$PACK_ID]}"
        for SEED in "${SEEDS[@]}"; do
            for ALGO in MAPPO QMIX; do
                run_int="0"
                if [ "$SEED" = "42" ]; then
                    run_int="1"
                fi
                run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_partial" "pixel" "False" "$SEED" "$run_int"
                run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_full"    "pixel" "True"  "$SEED" "0"
                run_train_and_eval "$ENV_NAME" "$ALGO" "state_partial" "state" "False" "$SEED" "0"
                run_train_and_eval "$ENV_NAME" "$ALGO" "state_full"    "state" "True"  "$SEED" "0"
                run_train_and_eval "$ENV_NAME" "$ALGO" "blind"         "blind" "False" "$SEED" "0"
            done
        done
        ;;
    4)
        for ENV_NAME in stag_hunt_arena foraging_commons; do
            for SEED in "${SEEDS[@]}"; do
                for ALGO in MAPPO QMIX; do
                    if [ "$SEED" = "42" ]; then
                        run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_partial" "pixel" "False" "$SEED" "1"
                    fi
                    run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_full"    "pixel" "True"  "$SEED" "0"
                done
            done
        done
        ;;
    5)
        for ENV_NAME in coop_puzzle platform_chain; do
            for SEED in "${SEEDS[@]}"; do
                for ALGO in MAPPO QMIX; do
                    if [ "$SEED" = "42" ]; then
                        run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_partial" "pixel" "False" "$SEED" "1"
                    fi
                    run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_full"    "pixel" "True"  "$SEED" "0"
                done
            done
        done
        ;;
    6)
        for ENV_NAME in coop_health_gathering lavapit; do
            for SEED in "${SEEDS[@]}"; do
                for ALGO in MAPPO QMIX; do
                    if [ "$SEED" = "42" ]; then
                        run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_partial" "pixel" "False" "$SEED" "1"
                    fi
                    run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_full"    "pixel" "True"  "$SEED" "0"
                done
            done
        done
        ;;
    7)
        for ENV_NAME in smart_enemies dumb_enemies lava_maze; do
            for SEED in "${SEEDS[@]}"; do
                for ALGO in MAPPO QMIX; do
                    if [ "$SEED" = "42" ]; then
                        run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_partial" "pixel" "False" "$SEED" "1"
                    fi
                    run_train_and_eval "$ENV_NAME" "$ALGO" "pixel_full"    "pixel" "True"  "$SEED" "0"
                done
            done
        done
        ;;
esac

echo "[$(date +%T)] PACK_ID=$PACK_ID completed successfully!"
