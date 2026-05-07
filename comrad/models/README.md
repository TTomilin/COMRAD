# `comrad/models/`

This directory defines the neural model families used by COMRAD's supported MARL baselines.

## Main files

- `doom_model.py`: shared visual encoder factory and ViZDoom-specific model registration
- `mappo_model.py`: IPPO and MAPPO actor-critic models
- `happo_model.py`: HAPPO-specific actor and critic structure
- `qmix_model.py`: value-decomposition actor-critic wrapper and QMIX-side model logic
- `qplex_mixer.py`: QPLEX mixer implementations

## Algorithm mapping

- IPPO and MAPPO use the actor-critic path registered through `mappo_model.py`
- HAPPO uses the agent-specific actor path in `happo_model.py`
- VDN, QMIX, and QPLEX use the decomposition path centered on `qmix_model.py` plus the relevant mixer

If you are changing algorithm-specific modeling assumptions, start here and then trace the corresponding learner in `sample_factory/algo/learning/`.
