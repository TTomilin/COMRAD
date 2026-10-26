# `comrad/wrappers/`

Runtime wrappers around the base ViZDoom env.

- `scenario_wrappers/`: per-scenario reward shaping and the `info["true_objective"]` score. Start here when rewards or metrics look wrong.
- `shared_reward.py`: `(1 - α)·r_i + α·Σ r_j` reward blending (the `shared_reward` option)
- `additional_input.py`: optional measurement vector (the `additional_input` option)
- `agent_id_wrapper.py`: agent-id observations for HAPPO
- `action_mask.py`, `observation_space.py`, `multiplayer_stats.py`, `video_recorder.py`: masks, resizing, stats and training-time video logging

What each scenario rewards is summarised on its [scenario page](https://ttomilin.github.io/COMRAD/scenarios/).
