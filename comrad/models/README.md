# `comrad/models/`

Model families for the bundled baselines ([training docs](https://ttomilin.github.io/COMRAD/training/sample-factory/)).

- `doom_model.py`: shared visual encoder
- `mappo_model.py`: IPPO and MAPPO actor-critics
- `happo_model.py`: HAPPO per-agent actors and critic
- `qmix_model.py`: value-decomposition path (VDN, QMIX, QPLEX agent networks and the QMIX mixer)
- `qplex_mixer.py`: QPLEX mixers

The matching learners live in `sample_factory/algo/learning/`.
