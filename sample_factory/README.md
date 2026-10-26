# `sample_factory/`

Modified [Sample Factory](https://github.com/alex-petrenko/sample-factory) that trains the COMRAD baselines (IPPO, MAPPO, HAPPO, IDQN, VDN, QMIX, QPLEX). It is shipped inside `comrad-bench`. Usage is documented in the [training docs](https://ttomilin.github.io/COMRAD/training/sample-factory/).

- `algo/`: learners, samplers, runners
- `cfg/`: arguments and config validation
- `launcher/`: local and SLURM launchers
- `model/`, `utils/`: model interfaces and helpers
- `docs/`: implementation notes
- `tests/`: engine regression tests (`uv run pytest sample_factory/tests -v`)

Benchmark semantics and model registration live in `comrad/`, and sampling, learning and checkpointing live here.
