#!/usr/bin/env bash
# Regenerate the web-sized images used by README.md and the MkDocs site (docs/assets/).
# Requires macOS `sips` and `ffmpeg`. Run from the repository root: ./scripts/build_docs_assets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/assets"
mkdir -p "$OUT/scenarios" "$OUT/figures"

# Scenario screenshots: results/scenarios/<i>.png -> docs/assets/scenarios/<scenario>.jpg
# (mapping from scripts/scenario_overview_figure.py)
SCENARIOS=(
  "1:stag_hunt_arena" "2:rhythm_sync" "3:foraging_commons" "4:coop_puzzle"
  "5:platform_chain" "6:armory_siege" "7:coop_health_gathering" "8:lavapit"
  "9:smart_dumb_enemies" "10:stealth_labyrinth" "11:ammo_carrier" "12:lava_maze"
)
for entry in "${SCENARIOS[@]}"; do
  idx="${entry%%:*}"
  name="${entry##*:}"
  sips -s format jpeg -s formatOptions 80 -Z 640 "$ROOT/results/scenarios/$idx.png" \
    --out "$OUT/scenarios/$name.jpg" >/dev/null
done

# Result figures, downscaled to at most 1600 px on the longest side.
for fig in baseline_results_heatmap baseline_learning_curves agent_scaling_learning agent_scaling_throughput; do
  sips -Z 1600 "$ROOT/results/$fig.png" --out "$OUT/figures/$fig.png" >/dev/null
done

# Hero GIF (~5 MB): 720 px wide, 8 fps, first 12 s of the demo video, no dithering so frames compress well.
SRC_VIDEO="$ROOT/results/demo_all_checkpoints/comrad_vid_540.mp4"
FILTERS="fps=8,scale=720:-1:flags=lanczos"
ffmpeg -v error -y -t 12 -i "$SRC_VIDEO" -vf "$FILTERS,palettegen=max_colors=64:stats_mode=diff" /tmp/comrad_palette.png
ffmpeg -v error -y -t 12 -i "$SRC_VIDEO" -i /tmp/comrad_palette.png \
  -lavfi "$FILTERS [x]; [x][1:v] paletteuse=dither=none:diff_mode=rectangle" "$OUT/comrad_demo.gif"
rm -f /tmp/comrad_palette.png

du -sh "$OUT"/*
