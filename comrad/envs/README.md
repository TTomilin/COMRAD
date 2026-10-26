# `comrad/envs/`

Environment layer: turns scenario assets into multi-agent ViZDoom environments.

- `doom_gym.py`: base Gymnasium ViZDoom env
- `multiagent/doom_multiagent.py`: per-player multiplayer game
- `multiagent/doom_multiagent_wrapper.py`: `MultiAgentEnv`, which steps all players in lockstep (`auto_reset=False` in the public API)
- `pettingzoo_env.py`: `ComradParallelEnv`, the PettingZoo adapter behind `comrad.parallel_env`
- `doom_params.py`: CLI options and `default_doom_cfg()`
- `multi_wad_env.py`, `wad_catalog.py`: episode-boundary swapping across `batch_registry.json` pools
- `action_space.py`: per-scenario tuple action spaces

Episode semantics are documented in the [API docs](https://ttomilin.github.io/COMRAD/api/#episodes).
