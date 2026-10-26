# `comrad/`

The `comrad-bench` package. User documentation is at **https://ttomilin.github.io/COMRAD/**.

| Path | Purpose |
|---|---|
| `__init__.py`, `api.py` | Public API: `make`, `parallel_env`, `list_scenarios`, Gymnasium ids ([API docs](https://ttomilin.github.io/COMRAD/api/)) |
| `utils/doom_utils.py` | Scenario registry (`DOOM_ENVS`, `SCENARIO_METADATA`) and the env factory used by both the API and training |
| `train.py`, `train_all.py` | Training entry point and paper launcher profiles ([training docs](https://ttomilin.github.io/COMRAD/training/sample-factory/)) |
| `record_video.py`, `record_topdown_heatmap.py`, `enjoy.py` | Checkpoint recording and evaluation |
| [`envs/`](envs/README.md) | ViZDoom env, multiplayer sync, PettingZoo adapter, WAD batches |
| [`scenarios/`](scenarios/README.md) | Shipped `.cfg`/`.wad` assets and batch pools |
| [`wrappers/`](wrappers/README.md) | Reward shaping, shared reward, measurements, recording |
| [`models/`](models/README.md) | Encoders, actor-critics and mixers for the bundled algorithms |
| [`curriculum/`](curriculum/README.md) | Task-pool curricula over WAD batches |
| [`tests/`](tests/README.md) | COMRAD tests |
| [`examples/`](examples/README.md) | Local debugging scripts |
| `play/`, `misc/` | Standalone ViZDoom binaries and legacy scenarios (not part of the benchmark) |
