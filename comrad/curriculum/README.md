# `comrad/curriculum/`

This directory implements curriculum strategies over fixed pools of pre-generated WAD tasks.

## Supported strategies

- `uniform.py`
- `sequential.py`
- `learning_progress.py`
- `omni.py`
- `plr.py`

`base.py` defines the common task-pool interface, and `observer.py` handles persistence and logging hooks.

## How it is activated

Curriculum is not enabled for ordinary single-WAD runs. It is activated through:

- `--wad_batch=<path-to-batch-dir>`
- `--curriculum=<strategy>`

The batch directory must contain a `batch_registry.json` file readable through `comrad.envs.wad_catalog`.

## Persistence

- learner checkpoints store curriculum state when curriculum is active
- `observer.py` also writes a `curriculum_state.json` file into the experiment directory

For the task pools consumed by these strategies, see [`../scenarios/README.md`](../scenarios/README.md).
