# COMRAD

COMRAD is a cooperative multi-agent reinforcement learning benchmark built on ViZDoom for first-person visual coordination. This repository contains the benchmark runtime assets, COMRAD-specific environment and model code, a modified Sample Factory training stack, paper artifacts, and the scripts used to produce figures, tables, and diagnostic media.

## Repository map

| Path | Purpose |
| --- | --- |
| `comrad/` | Benchmark package: training entrypoints, environment registration, wrappers, models, shipped scenario assets, launchers, tests. |
| `sample_factory/` | Modified training engine used by COMRAD. |
| `scripts/` | Figure-generation, result aggregation, WAD rendering, and experiment helper scripts. |
| `results/` | Committed figures, tables, result summaries, and selected media artifacts. |
| `DoomGen/` | Procedural generator used to build and vary scenario WADs. |

Directory-level guides are provided in:

- [`comrad/README.md`](comrad/README.md)
- [`sample_factory/README.md`](sample_factory/README.md)
- [`scripts/README.md`](scripts/README.md)
- [`results/README.md`](results/README.md)

## Setup

COMRAD is packaged through `pyproject.toml` and is intended to be run with `uv`.

```bash
uv sync --all-extras
uv run pre-commit install
```

The project requires Python 3.10 or newer. The package metadata and CI workflow are the authoritative sources for dependency resolution.

## Single-run training

The main training entrypoint is `comrad.train`.

```bash
uv run python -m comrad.train \
  --env=stag_hunt_arena \
  --algo=IPPO \
  --train_for_env_steps=10000 \
  --num_workers=4 \
  --num_envs_per_worker=4 \
  --device=cpu
```

Training outputs are written under `train_dir/<experiment>/`. The main runtime log is:

```bash
tail -n 120 train_dir/<experiment>/sf_log.txt
```

For launcher-owned benchmark sweeps, use `comrad/train_all.py` instead of manually reproducing those long commands.

## Recording and evaluation

High-resolution checkpoint recording:

```bash
uv run python -m comrad.record_video \
  --env=stag_hunt_arena \
  --train_dir=train_dir \
  --experiment=<experiment_name> \
  --load_checkpoint_kind=best \
  --resolution=1280x720
```

Automap/top-down heatmap rendering:

```bash
uv run python -m comrad.record_topdown_heatmap \
  --env=ammo_carrier \
  --train_dir=<train_root> \
  --experiment=<experiment_name> \
  --output_dir=results/videos/ammo_carrier_navigation_heatmap \
  --device=cpu \
  --overwrite
```

Both commands expect a full experiment directory with at least `config.json` and the corresponding checkpoint directory.

## Benchmark launchers

COMRAD ships several paper-facing launcher profiles through `comrad/train_all.py`.

- `benchmark`: full benchmark suite
- `agent_scaling`: Armory Siege agent-count sweep
- `platform_chain_curriculum`: curriculum comparison on Platform Chain
- `qmix_lr_sensitivity_pilot`: Short sensitivity study
- `qmix_lr_sensitivity_full`: Full sensitivity study

Run a launcher profile locally with:

```bash
COMRAD_TRAIN_ALL_PROFILE=benchmark \
uv run python -m sample_factory.launcher.run \
  --run=comrad.train_all \
  --backend=processes \
  --max_parallel=1 \
  --pause_between=1
```

The exact benchmark definitions, algorithms, root directory names, and profile constants live at the top of [`comrad/train_all.py`](comrad/train_all.py).

## Scenario assets and DoomGen

Runtime scenario assets are stored in [`comrad/scenarios/`](comrad/scenarios/README.md) as `.cfg` and `.wad` pairs, plus pre-generated batch directories such as `batch_platform_chain_curriculum/`.

Generation workflows are in `DoomGen/`. The usual split is:

- `DoomGen/examples/benchmark`: procedural generators and geometry logic
- `comrad/scenarios/`: runtime WADs and configs consumed by training and evaluation

If a scenario generator changes, rebuild the corresponding WAD in `DoomGen/` and copy the regenerated artifact back into `comrad/scenarios/` before training or evaluation.

## Tests

Run COMRAD and Sample Factory tests with `uv`:

```bash
uv run pytest comrad/tests -v
uv run pytest sample_factory/tests -v
```

Tests in `comrad/tests/` focus on benchmark logic, launchers, and recorder behavior. Tests in `sample_factory/tests/` cover the modified training stack.

## Outputs and artifacts

- `train_dir/`: experiment directories, configs, checkpoints, logs
- `results/`: committed figures, tables, result summaries, selected media
- `wandb/`: local W&B run state and uploads

See [`results/README.md`](results/README.md) for the structure of the committed artifacts.

## License and upstream assets

The repository is released under the MIT license in [`LICENSE`](LICENSE). Upstream and third-party assets are documented in the repository itself, including the bundled `scripts/wad2image/` tool and the paper's citations to ViZDoom, Sample Factory, HARL, PyMARL, and QPLEX.
