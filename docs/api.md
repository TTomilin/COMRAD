# API reference

```python
import comrad

env = comrad.parallel_env("lavapit", shared_reward=1.0)  # PettingZoo
env = comrad.make("lavapit", shared_reward=1.0)          # Gymnasium, per-agent lists
```

## Functions

### `comrad.parallel_env(scenario, **options)`

Returns a PettingZoo [`ParallelEnv`](https://pettingzoo.farama.org/api/parallel/).

- `possible_agents`: `["player_0", ..., "player_{N-1}"]`
- `reset(seed=None, options=None) -> (observations, infos)`
- `step(actions) -> (observations, rewards, terminations, truncations, infos)`, all dicts keyed by agent
- `observation_space(agent)` and `action_space(agent)` are the same for every agent
- `render()` and `close()`

### `comrad.make(scenario, **options)`

Returns a Gymnasium env with the same options. `reset()` returns `(obs_list, info_list)`, and `step(action_list)` returns `(obs_list, reward_list, terminated_list, truncated_list, info_list)`, with one entry per agent. `observation_space` and `action_space` describe a single agent.

After `import comrad`, every scenario is also registered as `comrad/<scenario>-v0`, so `gym.make("comrad/lavapit-v0", **options)` works.

### `comrad.list_scenarios()`

Returns a list of `ScenarioInfo(name, num_agents, description, score)`, one for each scenario.

## Options

| Option | Default | Meaning |
|---|---|---|
| `num_agents` | scenario default (2) | Team size, must be ≥ 2. See each scenario page for supported sizes. |
| `frameskip` | `4` | Game tics per `step()`. The action is repeated for every tic. |
| `resolution` | `(128, 72)` | `(width, height)` of each observation. |
| `pixel_format` | `"HWC"` | `"HWC"` gives `(H, W, 3)` arrays, `"CHW"` gives `(3, H, W)`. |
| `shared_reward` | `None` | `None` uses the scenario default. Any value α in `[0, 1]` blends each agent's reward with the team reward: `(1 - α) · r_i + α · Σ_j r_j`. |
| `additional_input` | `False` | Adds a measurement vector (health, ammo, ...) for scenarios that define one. Observations become `{"obs": image, "measurements": vector}`. |
| `render_mode` | `None` | `"rgb_array"` makes `render()` return all agent views side by side, and `"human"` shows them in a window. |

The defaults match the configuration used for the paper baselines.

## Spaces

- **Observation**: `Box(0, 255, (72, 128, 3), uint8)` per agent, or a `Dict` with `measurements` when `additional_input=True`.
- **Action**: a `Tuple` of `Discrete` heads, for example move, strafe, turn and attack. Index `0` of every head is "do nothing". The heads for each scenario are listed on its page.

## Episodes

- The env **never resets itself**. Call `reset()` after every agent is terminated or truncated.
- In `parallel_env`, an agent that finishes early leaves `env.agents`. It no longer receives observations or rewards, and COMRAD sends it a no-op action until the episode ends.
- If the Doom engine crashes, COMRAD restarts the game transparently. The interrupted episode is then reported as truncated for every agent.

## Info keys

| Key | When | Meaning |
|---|---|---|
| `true_objective` | at episode end | The benchmark score for the scenario. Report this, not the shaped return. |
| `episode_extra_stats` | at episode end | Extra scenario statistics, as a dict. |
| `num_frames` | every step | Game tics simulated by the step (= `frameskip`). |
| `_hidden_reset_count` | after a crash recovery | Number of transparent restarts since the previous step. |
