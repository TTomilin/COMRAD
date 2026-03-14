# Multi-WAD Batch Training

Allows training across a set of pre-generated WAD files rather than a single fixed map. Each environment worker resets to the next WAD in the batch according to a sampling strategy, exposing agents to many map layouts easily.

---

## Overview

The standard COMRAD setup binds one `DoomSpec` to one `.cfg` file, which points to one `.wad`. Multi-WAD extends this by:

1. Keeping the base `.cfg` identical across all WADs (action space, reward shaping, game settings).
2. Patching only the `doom_scenario_path` line in a temporary copy of that `.cfg` for each WAD.
3. Calling `swap_scenario()` on the underlying `VizdoomEnv` only when an episode ends and the inner env auto-resets.

No agent code changes are required. The observation and action spaces are identical across all maps in a batch.

---

## Data flow

```
Batch directory on disk
  -> WAD files + batch_registry.json

COMRAD WadBatch.from_dir(batch_dir)
  -> list of WadInfo(name, wad_path, metadata)

DoomBatchSpec(base=DoomSpec, batch_dir=..., swap_every=N, strategy=...)
  -> make_doom_env_from_batch()
      -> make_doom_env_from_spec(base)   - builds the full env stack
      -> MultiWADEnv(env, batch, ...)    - wraps it

MultiWADEnv.step()
  when episode is done and inner env auto-resets:
  every swap_every completed episodes:
    -> patch_wad_path(base_cfg, new_wad_path, tmp_cfg)
    -> env.swap_scenario(tmp_cfg)
      -> VizdoomEnv: game.close() + game.load_config(tmp_cfg) + game.init()
```

---


`batch_registry.json` schema (for batches):

```json
[
  {
    "id":       "armory_0",
    "filename": "armory_0.wad",
    "config":   { "distance": 600, "corridor_width": 4, ... }
  },
  ...
]
```

---

## Training with a Batch (COMRAD)

```bash
python -m comrad.train \
    --env armory_siege \
    --algo MAPPO \
    --wad_batch comrad/scenarios/batch_armory \
    --wad_swap_every 1 \
    --wad_strategy round_robin
```

This registers `armory_siege_batch` as the env name and sets `cfg.env` to it before the runner starts.

### CLI params

| Flag | Default | Description |
|---|---|---|
| `--wad_batch` | — | Path to the batch directory. Must contain `batch_registry.json` or `.wad` files. |
| `--wad_swap_every` | `1` | Number of episodes between WAD swaps. |
| `--wad_strategy` | `round_robin` | Sampling strategy: `round_robin`, `random`, `weighted`. |

---

## Relevant Classes

### `WadInfo` (`comrad/envs/wad_catalog.py`)
Simple dataclass holding `name`, `wad_path`, and `metadata` dict parsed from `batch_registry.json`.

### `WadBatch` (`comrad/envs/wad_catalog.py`)
Loads a batch directory. Supports `round_robin`, `random`, and `weighted` sampling. Weights can be updated externally (curriculum hook).

### `MultiWADEnv` (`comrad/envs/multi_wad_env.py`)
`gym.Wrapper` that sits on top of the full env stack. Calls `_apply_swap()` every `swap_every` episodes. Compatible with both single-agent `VizdoomEnv` and `MultiAgentEnv`.

### `DoomBatchSpec` / `make_doom_env_from_batch` (`comrad/utils/doom_utils.py`)
Factory for building a `MultiWADEnv`-wrapped environment registered with Sample Factory.

### `patch_wad_path` (`comrad/utils/wad_utils.py`)
Reads the base `.cfg`, replaces the `doom_scenario_path` line in-place, writes to a temp file. The original `.cfg` is never modified.

---

## Adding a New Scenario Batch

1. Generate a batch directory that contains `.wad` files and a `batch_registry.json` index.
2. Place it somewhere accessible to training (for example `comrad/scenarios/<batch_name>/`).
3. Add a `DoomSpec` entry for the scenario in `comrad/utils/doom_utils.py`.
4. Point `--wad_batch` at the generated directory.
