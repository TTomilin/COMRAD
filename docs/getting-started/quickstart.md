# Quickstart

COMRAD exposes every scenario in two ways, and both build the exact same environment that the paper's baselines were trained on.

| Interface | Call | Per-agent data |
|---|---|---|
| PettingZoo `ParallelEnv` | `comrad.parallel_env(name)` | dicts keyed `player_0`, `player_1`, ... |
| Gymnasium `Env` | `comrad.make(name)` or `gym.make("comrad/<name>-v0")` | lists indexed by agent |

## PettingZoo

```python
import comrad

env = comrad.parallel_env("stag_hunt_arena")
observations, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print("score:", {agent: info.get("true_objective") for agent, info in infos.items()})
env.close()
```

An agent that is done leaves `env.agents`, and the episode is over when the list is empty.

## Gymnasium

```python
import gymnasium as gym

import comrad  # registers the comrad/<scenario>-v0 ids

env = gym.make("comrad/stag_hunt_arena-v0", num_agents=2)
observations, infos = env.reset(seed=42)  # one entry per agent

done = False
while not done:
    actions = [env.action_space.sample() for _ in observations]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    done = all(t or tr for t, tr in zip(terminated, truncated))

print("score:", [info.get("true_objective") for info in infos])
env.close()
```

`env.observation_space` and `env.action_space` describe a single agent.

Both scripts live in [`examples/`](https://github.com/TTomilin/COMRAD/tree/main/examples).

## Common options

```python
env = comrad.parallel_env(
    "armory_siege",
    num_agents=3,            # scenarios support different team sizes, see each scenario page
    shared_reward=1.0,       # 1.0 = everyone gets the team reward, 0.0 = individual rewards
    resolution=(128, 72),    # (width, height) of each observation
    render_mode="rgb_array", # env.render() returns all agent views side by side
)
```

The full list is in the [API reference](../api.md).

## Episodes and scores

- Episodes never reset on their own: call `reset()` once all agents are done.
- The benchmark score is `info["true_objective"]` at the end of an episode. Rewards are a shaped training signal and differ from the score.
- `seed` sets the game seed for every agent.
