#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
runner="$repo_root/scripts/record_mode3_demo_one.sh"

if [[ ! -x "$runner" ]]; then
  echo "Missing helper: $runner"
  exit 1
fi

entries=(
  "foraging_commons 00_IPPO_env_foraging_commons_s.r.alp_0.0_see_42"
  "lava_maze 00_IPPO_env_lava_maze_s.r.alp_0.0_see_42"
  "lavapit 00_IPPO_env_lavapit_s.r.alp_1.0_see_42"
  "platform_chain 00_IPPO_env_platform_chain_s.r.alp_1.0_see_42"
  "rhythm_sync_dense 00_IPPO_env_rhythm_sync_dense_s.r.alp_1.0_see_42"
  "smart_enemies 00_IPPO_env_smart_enemies_s.r.alp_0.0_see_42"
  "stag_hunt_arena 00_IPPO_env_stag_hunt_arena_s.r.alp_0.0_see_42"
  "stealth_labyrinth 00_IPPO_env_stealth_labyrinth_s.r.alp_0.0_see_42"
)

for index in "${!entries[@]}"; do
  read -r scenario experiment <<<"${entries[$index]}"
  printf '[%d/%d] Recording %s\n' "$((index + 1))" "${#entries[@]}" "$scenario"
  "$runner" "$scenario" "$experiment"
done
