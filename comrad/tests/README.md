# `comrad/tests/`

This directory contains tests for COMRAD-specific behavior rather than generic Sample Factory logic.

## Coverage areas

- launcher profile construction in `train_all.py`
- batch and scaling configuration regressions
- recorder and top-down heatmap behavior
- selected environment and wrapper invariants

## Running tests

Run the whole COMRAD test set:

```bash
uv run pytest comrad/tests -v
```

Run a focused file:

```bash
uv run pytest comrad/tests/test_train_all_scaling.py -v
```

If a change spans both benchmark code and the training engine, also run the relevant tests under `sample_factory/tests/`.
