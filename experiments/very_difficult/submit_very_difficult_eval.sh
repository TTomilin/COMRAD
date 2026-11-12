#!/bin/bash
# Submits the held-out evaluation of the very-difficult learnability references on Snellius (run from ~/ViZDoom).
# One SLURM job per (env, algo); each job evaluates the listed training seeds in parallel.
#   experiments/very_difficult/submit_very_difficult_eval.sh                 # all 12 env x algo combos, seeds from SEEDS_<...> overrides
#   experiments/very_difficult/submit_very_difficult_eval.sh env:ALGO:seeds  # explicit specs (e.g. reruns)
#   experiments/very_difficult/submit_very_difficult_eval.sh converse        # the 4 converse combos (scripted gunner / navigator)
set -eu
cd "$(dirname "$0")/../.."
mkdir -p train_dir/vd_eval_logs
ALL_SEEDS=42,68,81,97,154

if [ "${1:-}" = "converse" ]; then
    SPECS=(
        "stealth_labyrinth_scripted_gunner:IPPO:$ALL_SEEDS"
        "stealth_labyrinth_scripted_gunner:IDQN:$ALL_SEEDS"
        "lava_maze_scripted_navigator:IPPO:$ALL_SEEDS"
        "lava_maze_scripted_navigator:IDQN:$ALL_SEEDS"
    )
elif [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=(
        "stealth_labyrinth_scripted_scout:IPPO:$ALL_SEEDS"
        "stealth_labyrinth_scripted_scout:IDQN:${SEEDS_SCOUT_IDQN:-$ALL_SEEDS}"
        "lava_maze_scripted_observer:IPPO:$ALL_SEEDS"
        "lava_maze_scripted_observer:IDQN:$ALL_SEEDS"
        "lava_maze_oracle_msg:MAPPO:$ALL_SEEDS"
        "lava_maze_oracle_msg:QMIX:$ALL_SEEDS"
        "lava_maze_oracle_hint:MAPPO:$ALL_SEEDS"
        "lava_maze_oracle_hint:QMIX:$ALL_SEEDS"
        "stealth_labyrinth_lit:MAPPO:${SEEDS_LIT_MAPPO:-$ALL_SEEDS}"
        "stealth_labyrinth_lit:QMIX:$ALL_SEEDS"
        "stealth_labyrinth_shared_vision:MAPPO:$ALL_SEEDS"
        "stealth_labyrinth_shared_vision:QMIX:${SEEDS_SV_QMIX:-$ALL_SEEDS}"
    )
fi

for spec in "${SPECS[@]}"; do
    echo "$spec -> $(sbatch --parsable --export=ALL experiments/very_difficult/very_difficult_heldout_eval.sh "$spec")"
done
squeue -u "$USER" -h -o "%i %j %T %M" | head -n 30
