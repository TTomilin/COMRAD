# `comrad/scenarios/`

Runtime assets for every scenario: one `.cfg` plus `.wad` pair per scenario, and `batch_*/` pools of pre-generated maps, each described by a `batch_registry.json`. For the scenario catalog, see the [scenario docs](https://ttomilin.github.io/COMRAD/scenarios/).

- `batch_armory/`, `batch_armory_siege/`, `batch_armory_siege_w2_eval/`: Armory Siege layout variants
- `batch_lava_maze/`: Lava Maze pool
- `batch_platform_chain_curriculum/`: Platform Chain curriculum pool (used by `comrad/train_all.py`)

Pools are consumed with `--wad_batch=<dir>` (see [Reproducing](https://ttomilin.github.io/COMRAD/training/reproducing/#wad-batches-and-curricula)).

The maps are produced by the separate generators in `DoomGen/`, which are not part of this package. After regenerating a map there, copy the new `.wad` here and keep the `.cfg` and `batch_registry.json` in sync. Training always reads from this directory.
