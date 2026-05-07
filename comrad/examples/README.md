# `comrad/examples/`

This directory contains exploratory scripts for inspecting or poking COMRAD environments. It is not the main reproduction surface for the benchmark or the paper.

## Files

- `single_player.py`: small single-player interaction example
- `agent_run_env.py`
- `you_run_env.py`
- `inspect_obs.py`: observation inspection utility
- `WandB_Scraper/`: older helper code for W&B-based plotting or inspection

Use the main package entrypoints in `comrad/train.py`, `comrad/record_video.py`, and `comrad/record_topdown_heatmap.py` for actual benchmark workflows. Treat these scripts as local utilities or debugging aids.
