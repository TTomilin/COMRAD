#!/bin/bash
# ==============================================================================
# Submit Reviewer kd6J training experiments to Snellius `gpu_mig` via `./comrad/send.sh 3`.
#
# Profiles supported (defined in `experiments/iclr2027_rebuttal/profiles.py`):
#   - P0 `obs_modality_2x2`: {Pixel, 1D State} x {Partial, Full} on Rhythm Sync, Armory Siege,
#                            Ammo Carrier, and Stealth Labyrinth for MAPPO and QMIX.
#   - P0 `blind_ablation`:   Zeroed RGB + normalized step counter t/H on the 4 representative
#                            scenarios for MAPPO only (SMACv2-style diagnostic).
#   - P0 `visibility_telemetry`: 1-seed MAPPO Pixel-Partial on the other 9 scenarios (telemetry checkpoints).
#   - Optional `full_observability`: Pixel-Partial vs Pixel-Full on the other 9 scenarios (MAPPO, QMIX).
#   - `full_obs_clean`: unconfounded MAPPO Pixel-Partial vs Pixel-Full (no AdditionalInput in either cell) on
#                       Rhythm Sync, Armory Siege, Co-op Puzzle with 3 seeds {42,68,81}; reuses existing
#                       Pixel-Partial checkpoints, so only 13 runs are trained (seeds via COMRAD_KD6J_CLEAN_SEEDS).
#   - `full_obs_global`: true full observability (own RGB + global state vector, `--full_obs_mode=global_state`)
#                        for MAPPO on the same 3 scenarios and seeds; 9 runs, Partial runs are reused.
#   - `state_entity`: SMAC-style partially observable state (`--state_encoding=entity`) vs Pixel-Partial on the 5
#                     cells of Table tab:po_state, seeds {42,68,81} (via COMRAD_STATE_ENTITY_SEEDS); 15 state runs
#                     plus Pixel-Partial seed 81 (5 runs), seeds 42/68 are reused from obs_modality_2x2.
#
# Minimal design (default `p0`): 64 + 8 + 9 = 81 runs at 50M env steps, 2 seeds {42,68}.
#
# Usage (from local repo root):
#   experiments/iclr2027_rebuttal/submit_rebuttal_kd6j_train.sh [p0|p1|obs_modality_2x2|blind_ablation|full_observability|full_obs_clean|full_obs_global|full_obs_global_v2|state_entity|all] [seeds]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

TARGET="${1:-p0}"
SEEDS="${2:-42,68}"

export COMRAD_KD6J_SEEDS="$SEEDS"

submit_profile() {
    local profile="$1"
    echo "=================================================================="
    echo "[$(date +%T)] Submitting profile=$profile with seeds=${COMRAD_KD6J_SEEDS:-${COMRAD_FULL_OBS_V2_SEEDS:-}} on gpu_mig"
    echo "=================================================================="
    COMRAD_EXPERIMENT_PROFILE="$profile" ./comrad/send.sh 3
}

case "$TARGET" in
    p0)
        submit_profile "obs_modality_2x2"
        submit_profile "blind_ablation"
        submit_profile "visibility_telemetry"
        ;;
    p1)
        submit_profile "full_observability"
        ;;
    obs_modality_2x2|blind_ablation|visibility_telemetry|full_observability)
        submit_profile "$TARGET"
        ;;
    full_obs_clean|full_obs_global)
        export COMRAD_KD6J_CLEAN_SEEDS="${2:-42,68,81}"
        submit_profile "$TARGET"
        ;;
    state_entity)
        export COMRAD_STATE_ENTITY_SEEDS="${2:-42,68,81}"
        submit_profile "$TARGET"
        ;;
    full_obs_global_v2)
        # fixed global-state encoding, Table 3 protocol: 5 seeds, 100M frames, layout pools (experiments/iclr2027_rebuttal/layouts/batch_<env>_layouts)
        export COMRAD_FULL_OBS_V2_SEEDS="${2:-42,68,81,97,154}"
        submit_profile "$TARGET"
        ;;
    all)
        submit_profile "obs_modality_2x2"
        submit_profile "blind_ablation"
        submit_profile "full_observability"
        ;;
    *)
        echo "Usage: $0 [p0|p1|obs_modality_2x2|blind_ablation|full_observability|full_obs_clean|full_obs_global|full_obs_global_v2|state_entity|all] [seeds]"
        exit 1
        ;;
esac
