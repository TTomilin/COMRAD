#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="${TRAIN_ROOT:-/home/20231193/ViZDoom/train_dir/mode3_comrad_benchmark_20260503_134649}"
ENV_NAME="${ENV_NAME:-ammo_carrier}"
TRAIN_DIR="${TRAIN_ROOT}/${ENV_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/videos/${ENV_NAME}_navigation_heatmap}"
DEVICE="${DEVICE:-gpu}"
EPISODES="${EPISODES:-3}"
RESOLUTION="${RESOLUTION:-1600x1200}"
DENSITY_GRID_RESOLUTION="${DENSITY_GRID_RESOLUTION:-200x150}"
VIDEO_FPS="${VIDEO_FPS:-35}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -d "${TRAIN_DIR}" ]]; then
    echo "Train dir not found: ${TRAIN_DIR}" >&2
    exit 1
fi

if [[ -z "${UV_CACHE_DIR:-}" ]]; then
    export UV_CACHE_DIR=/tmp/uv-cache
fi
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
    export MPLCONFIGDIR=/tmp/matplotlib
fi

mkdir -p "${OUTPUT_ROOT}"

mapfile -t EXPERIMENT_DIRS < <(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#EXPERIMENT_DIRS[@]} -eq 0 ]]; then
    echo "No experiment directories found in ${TRAIN_DIR}" >&2
    exit 1
fi

FOUND_ANY=0
for experiment_dir in "${EXPERIMENT_DIRS[@]}"; do
    experiment_name="$(basename "${experiment_dir}")"
    checkpoint_dir="${experiment_dir}/checkpoint_p0"

    if [[ ! -d "${checkpoint_dir}" ]]; then
        echo "Skipping ${experiment_name}: missing checkpoint_p0"
        continue
    fi

    shopt -s nullglob
    best_checkpoints=("${checkpoint_dir}"/best_*.pth)
    shopt -u nullglob
    if [[ ${#best_checkpoints[@]} -eq 0 ]]; then
        echo "Skipping ${experiment_name}: no best_*.pth checkpoint"
        continue
    fi

    FOUND_ANY=1
    experiment_output_dir="${OUTPUT_ROOT}/${experiment_name}"
    mkdir -p "${experiment_output_dir}"

    cmd=(
        python -m comrad.record_topdown_heatmap
        "--env=${ENV_NAME}"
        "--train_dir=${TRAIN_DIR}"
        "--experiment=${experiment_name}"
        "--device=${DEVICE}"
        "--max_num_episodes=${EPISODES}"
        "--resolution=${RESOLUTION}"
        "--density_grid_resolution=${DENSITY_GRID_RESOLUTION}"
        "--video_fps=${VIDEO_FPS}"
        "--output_dir=${experiment_output_dir}"
    )

    if [[ "${OVERWRITE}" == "1" ]]; then
        cmd+=(--overwrite)
    fi

    echo "Rendering ${experiment_name}"
    printf '  %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
done

if [[ "${FOUND_ANY}" != "1" ]]; then
    echo "No ammo_carrier experiments with best checkpoints were found in ${TRAIN_DIR}" >&2
    exit 1
fi

echo "Finished ammo_carrier topdown renders into ${OUTPUT_ROOT}"
