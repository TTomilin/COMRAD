#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
benchmark_name="mode3_comrad_benchmark_20260503_134649"
snellius_root="/home/knguyen2/ViZDoom/train_dir/${benchmark_name}"
local_root="${repo_root}/results/${benchmark_name}"

if [[ -d "${snellius_root}" ]]; then
  default_train_root="${snellius_root}"
else
  default_train_root="${local_root}"
fi

TRAIN_ROOT="${TRAIN_ROOT:-${default_train_root}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/results/demo_all_checkpoints}"
DEVICE="${DEVICE:-cpu}"
EPISODES="${EPISODES:-1}"
MAX_FRAMES="${MAX_FRAMES:-10000}"
RESOLUTION="${RESOLUTION:-1280x720}"
VIDEO_FPS="${VIDEO_FPS:-35}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "Train root not found: ${TRAIN_ROOT}" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mapfile -t checkpoints < <(
  find "${TRAIN_ROOT}" -mindepth 4 -maxdepth 4 -type f \
    -path "*/checkpoint_p0/*.pth" \
    ! -path "*/failed_*" \
    ! -path "*_IDQN_env_*" \
    ! -path "*_VDN_env_*" \
    | sort
)

if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No non-IDQN/VDN checkpoints found in ${TRAIN_ROOT}" >&2
  exit 1
fi

if (( LIMIT > 0 && LIMIT < ${#checkpoints[@]} )); then
  checkpoints=("${checkpoints[@]:0:LIMIT}")
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${repo_root}"

for index in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$index]}"
  experiment_dir="$(dirname "$(dirname "${checkpoint}")")"
  train_dir="$(dirname "${experiment_dir}")"
  scenario="$(basename "${train_dir}")"
  experiment="$(basename "${experiment_dir}")"
  checkpoint_name="$(basename "${checkpoint}" .pth)"
  algo="${experiment#*_}"
  algo="${algo%%_env_*}"
  experiment_output_dir="${OUTPUT_ROOT}/${scenario}/${experiment}"
  video_prefix="demo_${checkpoint_name}"

  cmd=(
    uv run python -m comrad.record_video
    "--env=${scenario}"
    "--algo=${algo}"
    "--train_dir=${train_dir}"
    "--experiment=${experiment}"
    "--device=${DEVICE}"
    "--load_checkpoint_file=${checkpoint}"
    "--eval_deterministic=True"
    "--output_dir=${experiment_output_dir}"
    "--video_prefix=${video_prefix}"
    "--resolution=${RESOLUTION}"
    "--video_fps=${VIDEO_FPS}"
    "--max_num_episodes=${EPISODES}"
    "--max_num_frames=${MAX_FRAMES}"
  )

  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite_video)
  fi

  printf '[%d/%d] %s %s %s\n' "$((index + 1))" "${#checkpoints[@]}" "${scenario}" "${algo}" "${checkpoint_name}"
  printf '  %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${experiment_output_dir}"
    "${cmd[@]}"
  fi
done

echo "Finished recording ${#checkpoints[@]} checkpoints into ${OUTPUT_ROOT}"
