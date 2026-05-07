# `comrad/envs/`

This directory contains the environment construction layer that turns shipped WAD assets and benchmark configs into trainable multi-agent ViZDoom environments.

## Key files

- `doom_gym.py`: base Gymnasium-facing ViZDoom environment wrapper
- `multiagent/doom_multiagent.py`: synchronized per-agent multiplayer game management
- `multiagent/doom_multiagent_wrapper.py`: higher-level multi-agent environment wrapper
- `doom_params.py`: CLI arguments and defaults for COMRAD environment options
- `multi_wad_env.py`: episode-boundary swapping across pre-generated WAD pools
- `wad_catalog.py`: metadata loader for `batch_registry.json`
- `action_space.py`: factorized action-space definitions and utilities

## Runtime flow

1. `comrad.train` registers COMRAD environments from `comrad.utils.doom_utils`.
2. Scenario configs resolve to assets in `comrad/scenarios/`.
3. Optional `--wad_batch` and `--curriculum` settings wrap a base scenario with `multi_wad_env.py`.
4. The final environment is wrapped again by benchmark-specific reward and logging wrappers.

## When to edit this directory

- Add or change environment CLI flags in `doom_params.py`.
- Extend batch-WAD behavior in `multi_wad_env.py` and `wad_catalog.py`.
- Debug synchronization, reset, or step issues in `multiagent/`.
- Change the factorized action interface in `action_space.py`.

For shipped assets and batch registries, see [`../scenarios/README.md`](../scenarios/README.md).
