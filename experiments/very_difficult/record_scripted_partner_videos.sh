#!/usr/bin/env bash
# High-quality presentation videos of the scripted-partner learning references (paper App. very_difficult_reference):
#   Stealth Labyrinth: scripted scout -> learned IDQN gunner (seed 154, best held-out seed)
#   Lava Maze:         scripted observer -> learned IPPO navigator (seed 154, best held-out seed)
# Uses the 50M-step checkpoints evaluated in tab:very_difficult_reference. The policy plays at its training resolution
# (greedy Q-actions for IDQN, as in scripts/heldout_wad_evaluation.py) while the host records a .lmp demo, which is then
# replayed at RESOLUTION from both players' views (comrad/utils/record_demo_video.py). Fetch the checkpoints first (repo root):
#   rsync -a --include='*/' --include='config.json' --include='checkpoint_p0/***' --exclude='*' \
#     <hpc_host>:<hpc_repo>/train_dir/mode3_comrad_very_difficult_scripted_20260926_005603/very_difficult_scripted/<EXP> \
#     train_dir/very_difficult_scripted/
# Usage: experiments/very_difficult/record_scripted_partner_videos.sh [stealth|lava|all]   (env: ATTEMPTS, RESOLUTION, OUTPUT_ROOT, EXTRA_ARGS)
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${repo_root}"
mkdir -p _vizdoom

TRAIN_DIR="${TRAIN_DIR:-train_dir/very_difficult_scripted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/scripted_partner_videos}"
RESOLUTION="${RESOLUTION:-1920x1080}"
ATTEMPTS="${ATTEMPTS:-3}"
MAX_FRAMES="${MAX_FRAMES:-30000}"
WHICH="${1:-all}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"  # e.g. EXTRA_ARGS=--eval_deterministic=True

record() {
  local env="$1" algo="$2" experiment="$3" checkpoint="$4" name="$5"
  local out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"
  for attempt in $(seq 1 "${ATTEMPTS}"); do
    echo "=== ${name} attempt ${attempt}/${ATTEMPTS}"
    python -m comrad.utils.record_demo_video \
      "--env=${env}" "--algo=${algo}" "--train_dir=${TRAIN_DIR}" "--experiment=${experiment}" \
      --device=cpu "--load_checkpoint_file=${TRAIN_DIR}/${experiment}/checkpoint_p0/${checkpoint}" \
      "--output_dir=${out}" "--video_prefix=${name}_ep${attempt}" "--resolution=${RESOLUTION}" \
      --video_quality=10 --video_fps=35 --max_num_episodes=1 "--max_num_frames=${MAX_FRAMES}" --overwrite_video \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      2>&1 | grep -E "finished after|Replayed|Error|Traceback" | tee "${out}/${name}_ep${attempt}.log"
  done
}

if [[ "${WHICH}" == "stealth" || "${WHICH}" == "all" ]]; then
  record stealth_labyrinth_scripted_scout IDQN 04_IDQN_env_stealth_labyrinth_scripted_scout_see_154 \
    checkpoint_000194544_50002944.pth stealth_labyrinth_scripted_scout_IDQN_s154
fi
if [[ "${WHICH}" == "lava" || "${WHICH}" == "all" ]]; then
  record lava_maze_scripted_observer IPPO 04_IPPO_env_lava_maze_scripted_observer_s.r.alp_1.0_see_154 \
    checkpoint_000012196_50020352.pth lava_maze_scripted_observer_IPPO_s154
fi
