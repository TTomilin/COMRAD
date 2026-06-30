#!/bin/bash
#SBATCH --job-name=marker_${SCENARIO}
#SBATCH --output=results/schelling/slurm_marker_%j_${SCENARIO}.out
#SBATCH --error=results/schelling/slurm_marker_%j_${SCENARIO}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

set -e
source .venv/bin/activate

BENCHMARK_DIR="train_dir/mode3_comrad_benchmark_20260503_134649"
OUTPUT_DIR="results/schelling"
mkdir -p "$OUTPUT_DIR"

SCENARIO="${SCENARIO}"
ALGOS=("IPPO" "MAPPO" "HAPPO")

for algo in "${ALGOS[@]}"; do
    echo "[$(date)] $SCENARIO / $algo"

    TRAIN_DIR="$BENCHMARK_DIR/$SCENARIO"
    EXPERIMENT=$(ls -d "$TRAIN_DIR/00_${algo}_"* 2>/dev/null | head -1 | xargs basename)

    if [ -z "$EXPERIMENT" ]; then
        echo "No checkpoint found for $SCENARIO/$algo"
        continue
    fi

    echo "Experiment: $EXPERIMENT"
    echo "Train dir:$TRAIN_DIR"

    CSV_PATH="$TRAIN_DIR/$EXPERIMENT/cooperation_results.csv"
    rm -f "$CSV_PATH"

    time python scripts/analyze_cooperation.py \
        --env="$SCENARIO" \
        --algo="$algo" \
        --experiment="$EXPERIMENT" \
        --train_dir="$TRAIN_DIR" \
        --num_workers=4 \
        --num_envs_per_worker=4

    if [ -f "$CSV_PATH" ]; then
        cp "$CSV_PATH" "$OUTPUT_DIR/${SCENARIO}_${algo}_cooperation.csv"
        echo "Copied: $OUTPUT_DIR/${SCENARIO}_${algo}_cooperation.csv"
    else
        echo "No CSV produced for $SCENARIO/$algo"
    fi

    echo ""
done

echo "[$(date)] $SCENARIO done."
