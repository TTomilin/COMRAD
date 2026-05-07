# `sample_factory/`

This subtree is the modified training engine COMRAD uses for asynchronous collection, inference, learning, launchers, and experiment utilities.

## What is here

- `algo/`: learners, samplers, runners, and algorithm utilities
- `cfg/`: argument parsing and configuration validation
- `launcher/`: local process and SLURM launch backends
- `model/`: generic model interfaces used by the COMRAD-specific factories
- `utils/`: logging, typing, GPU, networking, and experiment helpers
- `tests/`: regression tests for the modified engine

## COMRAD-specific role

COMRAD uses this subtree as an in-repo engine dependency rather than as an untouched upstream vendor drop. When debugging training behavior, trace across both:

- `comrad/` for benchmark semantics and model registration
- `sample_factory/` for sampling, learning, checkpointing, and launch behavior

## Notes

- `sample_factory/docs/*.md` are implementation notes, not polished end-user guides
- `sample_factory/tests/` is the main verification surface for engine-level changes

Run engine tests with:

```bash
uv run pytest sample_factory/tests -v
```
