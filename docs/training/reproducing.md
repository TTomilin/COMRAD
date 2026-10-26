# Reproducing the paper

Run these commands from a [development install](../getting-started/installation.md#development-install).

## Benchmark sweeps

`comrad/train_all.py` defines every paper experiment as a launcher profile. Pick one with `COMRAD_TRAIN_ALL_PROFILE`:

| Profile | Experiments |
|---|---|
| `benchmark` (default) | All scenarios × all seven algorithms |
| `agent_scaling` | Armory Siege with 2, 3, 4, 6 and 8 agents |
| `platform_chain_curriculum` | Curriculum strategies on the Platform Chain pool |
| `qmix_lr_sensitivity_pilot` | Short QMIX learning-rate study |
| `qmix_lr_sensitivity_full` | Full QMIX learning-rate study |

```bash
COMRAD_TRAIN_ALL_PROFILE=benchmark uv run python -m sample_factory.launcher.run \
    --run=comrad.train_all --backend=processes --max_parallel=1 --pause_between=1
```

The launcher adds `--experiment` and `--train_dir` itself, so don't put them in the profile commands. Algorithm hyperparameters, step budgets and experiment names live at the top of `comrad/train_all.py`.

## Recording and heatmaps

```bash
# High-resolution video from a checkpoint
uv run python -m comrad.record_video --env=stag_hunt_arena --train_dir=train_dir \
    --experiment=<experiment> --load_checkpoint_kind=best --resolution=1280x720

# Top-down trajectory heatmap on the automap
uv run python -m comrad.record_topdown_heatmap --env=ammo_carrier --train_dir=<train_root> \
    --experiment=<experiment> --output_dir=results/videos/ammo_carrier_navigation_heatmap --device=cpu
```

Both need the full experiment directory with `config.json` and the checkpoint folder. Run recorders from the repository root, because ViZDoom writes a `_vizdoom/` folder into the working directory.

## WAD batches and curricula

Some scenarios ship pools of pre-generated maps under `comrad/scenarios/batch_*/`, each listed in a `batch_registry.json`:

- `batch_armory/`, `batch_armory_siege/`: Armory Siege layout variants
- `batch_lava_maze/`: Lava Maze pool
- `batch_platform_chain_curriculum/`: the Platform Chain curriculum used in the paper

```bash
uv run python -m comrad.train --env=platform_chain --algo=MAPPO \
    --wad_batch=comrad/scenarios/batch_platform_chain_curriculum \
    --curriculum=learning_progress --wad_swap_every=1
```

Available curricula: `uniform`, `learning_progress`, `plr`, `sequential` and `omni`. See `comrad/envs/doom_params.py` for their hyperparameters.

## Figures and tables

The scripts in `scripts/` rebuild the paper's figures and tables from the archived runs in `results/`. See `scripts/README.md` for the full list.
