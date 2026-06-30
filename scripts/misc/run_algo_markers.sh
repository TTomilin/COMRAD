#!/bin/bash
#SBATCH --job-name=asdf
#SBATCH --output=results/schelling/slurm_algo_%j.out
#SBATCH --error=results/schelling/slurm_algo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

set -e
source .venv/bin/activate

BENCHMARK_DIR="train_dir/mode3_comrad_benchmark_20260503_134649"
OUTPUT_DIR="results/schelling"
mkdir -p "$OUTPUT_DIR"

SCENARIOS=("stag_hunt_arena" "foraging_commons" "armory_siege")
ALGOS=("IPPO" "MAPPO" "HAPPO")

for scenario in "${SCENARIOS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "[$(date)] $scenario / $algo"

        TRAIN_DIR="$BENCHMARK_DIR/$scenario"
        EXPERIMENT=$(ls -d "$TRAIN_DIR/00_${algo}_"* 2>/dev/null | head -1 | xargs basename)

        if [ -z "$EXPERIMENT" ]; then
            echo "No checkpoint found for $scenario/$algo"
            continue
        fi

        echo "  Experiment: $EXPERIMENT"
        echo "  Train dir:  $TRAIN_DIR"

        CSV_PATH="$TRAIN_DIR/$EXPERIMENT/cooperation_results.csv"
        rm -f "$CSV_PATH"

        time python scripts/analyze_cooperation.py \
            --env="$scenario" \
            --algo="$algo" \
            --experiment="$EXPERIMENT" \
            --train_dir="$TRAIN_DIR" \
            --num_workers=4 \
            --num_envs_per_worker=4

        if [ -f "$CSV_PATH" ]; then
            cp "$CSV_PATH" "$OUTPUT_DIR/${scenario}_${algo}_cooperation.csv"
            echo "Copied: $OUTPUT_DIR/${scenario}_${algo}_cooperation.csv"
        else
            echo "No CSV produced for $scenario/$algo"
        fi

        echo ""
    done
done

echo "[$(date)] All done."
echo ""
for f in "$OUTPUT_DIR"/*cooperation.csv; do
    echo "$(basename $f): $(wc -l < $f) rows"
done
