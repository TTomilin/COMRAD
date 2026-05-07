# `comrad/scenarios/`

This directory stores the runtime benchmark assets consumed by COMRAD training and evaluation.

## Contents

- One `.cfg` plus one `.wad` per shipped scenario, for example `armory_siege.cfg` and `armory_siege.wad`
- Pre-generated task pools such as `batch_armory/`, `batch_lava_maze/`, and `batch_platform_chain_curriculum/`
- `batch_registry.json` files that enumerate the WAD instances available to a pool
- Optional pool metadata such as interestingness graphs for curriculum experiments

## Shipped scenarios

The main benchmark scenarios live here directly, including:

- `stag_hunt_arena`
- `rhythm_sync`
- `foraging_commons`
- `coop_puzzle`
- `platform_chain`
- `armory_siege`
- `coop_health_gathering`
- `lavapit`
- `smart_enemies`
- `dumb_enemies`
- `stealth_labyrinth`
- `ammo_carrier`
- `lava_maze`

Additional assets such as `platform_chain_easy` and `lava_maze_simple` are retained for ablations, easier variants, or diagnostics.

## Relationship to DoomGen

`comrad/scenarios/` is the runtime asset store. Generation workflows assume a co-located `DoomGen/` checkout, which is responsible for producing or mutating the underlying WAD geometry. When a generator changes:

1. Regenerate the WAD in `DoomGen/`.
2. Copy the resulting `.wad` into this directory or the relevant batch subdirectory.
3. Keep the paired `.cfg` and any `batch_registry.json` metadata aligned with the new artifact set.

Training and evaluation always consume the files from `comrad/scenarios/`, not the generator sources directly.
