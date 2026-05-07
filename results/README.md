# `results/`

This directory stores committed artifacts derived from benchmark runs.

## What is here

- paper-ready figures and plots such as `baseline_learning_curves.*` and `agent_scaling_*.*`
- generated LaTeX tables such as `baseline_results_table.tex`
- summary JSON files used by figure and table scripts
- selected benchmark result snapshots, for example `mode3_comrad_benchmark_20260503_134649/`
- media exports such as `videos/ammo_carrier_navigation_heatmap/`
- WAD-variant outputs and rendered images under `variants/`

## Subdirectories

- `scaling/`: scaling-study logs and TensorBoard event files
- `scenarios/`: scenario overview images
- `variants/`: generated WAD variants plus rendered previews
- `videos/`: rendered trajectory or heatmap media

## Relationship to scripts and paper

- scripts in [`../scripts/`](../scripts/README.md) generate or refresh many of these artifacts
- the paper consumes copies of selected outputs under `paper/figure/`

Do not assume every raw experiment is fully mirrored here. The canonical runtime outputs remain in `train_dir/` and any external HPC storage used during the experiments.
