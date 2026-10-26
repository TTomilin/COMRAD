# `scripts/`

Scripts that turn benchmark outputs into paper figures, tables and diagnostics. See also [Reproducing the paper](https://ttomilin.github.io/COMRAD/training/reproducing/).

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
- Documentation assets:
  - `build_docs_assets.sh`: regenerates `docs/assets/` (thumbnails, figures, hero GIF)
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
- The DoomGen scripts need the separate `DoomGen/` environment, because `doomgen` is not a COMRAD dependency
- Most scripts expect existing results under `results/` or `train_dir/`, and write into `results/` or `paper/figure/`
