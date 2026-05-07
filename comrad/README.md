# `comrad/`

This package contains the benchmark-specific code that sits on top of ViZDoom and the modified Sample Factory runner.

## Main entrypoints

- `train.py`: single-run training entrypoint used by `python -m comrad.train`
- `train_all.py`: launcher-owned benchmark, scaling, and curriculum profiles
- `record_video.py`: high-resolution RGB checkpoint recording
- `record_topdown_heatmap.py`: automap-aligned trajectory and density rendering
- `send.sh`: HPC sync and SLURM submission helper
- `enjoy.py`: interactive or evaluation-time rollout entrypoint

## Important subdirectories

- [`envs/`](envs/README.md): environment construction, multiplayer wrappers, batch WAD support
- [`scenarios/`](scenarios/README.md): shipped `.cfg` and `.wad` benchmark assets
- [`wrappers/`](wrappers/README.md): reward shaping, shared reward, logging, and observation wrappers
- [`models/`](models/README.md): policy backbones and mixers for the supported MARL algorithms
- [`curriculum/`](curriculum/README.md): task-pool curriculum strategies and persistence helpers
- [`templates/`](templates/README.md): SLURM templates rendered by `send.sh`
- [`tests/`](tests/README.md): unit and integration tests for COMRAD-specific behavior
- [`examples/`](examples/README.md): exploratory scripts, not the main reproduction surface

## Other contents

- `play/`: local standalone VizDoom binaries and launch helpers
- `misc/`: smaller legacy scenarios and auxiliary assets that are not part of the main benchmark suite
- `utils/`: environment registration, rendering helpers, video upload, WAD utilities

Use this package as the authoritative code surface for benchmark runtime behavior. The paper defines the benchmark claims; `comrad/` defines how those claims are executed in code.
