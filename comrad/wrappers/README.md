# `comrad/wrappers/`

This directory contains the benchmark-specific wrapper stack applied around the base ViZDoom environments.

## Structure

- `scenario_wrappers/`: per-scenario reward shaping and benchmark score logic
- `shared_reward.py`: actor-critic reward sharing when a scenario is intended to be fully joint
- `additional_input.py`: optional auxiliary measurement branch inputs
- `agent_id_wrapper.py`: explicit agent-identity observations used by HAPPO-style models
- `multiplayer_stats.py`: logging helpers for synchronized multi-agent episodes
- `observation_space.py`: observation-space adjustments
- `video_recorder.py`: recorder wrapper used during training-time video logging

## What belongs here

- Benchmark semantics that must be applied at runtime without modifying the underlying WAD
- Wrapper-level bookkeeping such as `true_objective`, shared reward, or auxiliary info channels
- Logging and recording behavior that should stay outside the core environment class

## Scenario wrappers

Each file in `scenario_wrappers/` defines the shaping and terminal-score semantics for one benchmark scenario. This is the first place to inspect when reward curves, benchmark metrics, or episode diagnostics look wrong.

For the shipped runtime assets these wrappers operate on, see [`../scenarios/README.md`](../scenarios/README.md).
