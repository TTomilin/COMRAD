# `scripts/`

This directory contains the scripts used to turn benchmark outputs into paper artifacts or diagnostics.

## Main groups

- Figure and table generation:
  - `baseline_results.py`
  - `baseline_results_table.py`
  - `baseline_heatmap.py`
  - `baseline_policy_family_gap_bars.py`
  - `difficulty_axes_figure.py`
  - `scenario_overview_figure.py`
  - `agent_scaling_figure.py`
  - `platform_chain_curriculum_protocol.py`
- DoomGen and WAD visualization:
  - `make_doomgen_variants.py`
  - `render_doomgen_variant_images.py`
  - `wad2image/`
- Experiment helpers:
  - `download_baseline_results.sh`
  - `run_armory_siege.sh`
  - `record_ammo_carrier_nav_heatmaps.sh`
- Analysis or ablation utilities:
  - `qmix_lr_sensitivity.py`
  - `categories.py`
  - `categories_with_difficulty.py`

## Usage notes

- Many scripts assume the repo root is the current working directory
- Some scripts assume a co-located `DoomGen/` checkout and pre-existing result files under `results/` or `train_dir/`
- Generated outputs are typically written back into `results/` or `paper/figure/`

Treat this directory as the artifact-production layer above the benchmark runtime.
