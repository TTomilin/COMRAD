# Integrations

COMRAD environments are standard PettingZoo `ParallelEnv`s and Gymnasium `Env`s, so most MARL code can use them directly. This page covers the few things that are specific to COMRAD.

## A custom training loop

```python
import numpy as np
import comrad

env = comrad.parallel_env("lavapit", pixel_format="CHW")
obs, infos = env.reset(seed=0)
episode_return = {agent: 0.0 for agent in env.possible_agents}

while env.agents:
    batch = np.stack([obs[agent] for agent in env.agents])  # (n_live_agents, 3, 72, 128) uint8
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # replace with your policy
    obs, rewards, terms, truncs, infos = env.step(actions)
    for agent, r in rewards.items():
        episode_return[agent] += r

score = next(iter(infos.values()))["true_objective"]
```

## Tuple action spaces

Each agent's action space is a `Tuple` of `Discrete` heads. Libraries that only accept a single `Discrete` or `MultiDiscrete` space need a small adapter:

```python
import numpy as np
from gymnasium.spaces import MultiDiscrete

space = env.action_space("player_0")
multi = MultiDiscrete([head.n for head in space.spaces])  # e.g. [3, 3, 3, 2]
to_env = lambda a: tuple(int(x) for x in np.asarray(a))   # MultiDiscrete sample -> COMRAD action
```

Treat each head as an independent categorical distribution (factored policy). Flattening all heads into one `Discrete` of their product also works, because the product stays small (for example `3·3·3·2 = 54`).

## Running many environments

- Each environment runs one Doom process per agent, so parallelise across **processes** (for example `gymnasium.vector.AsyncVectorEnv`, or one environment per worker in RLlib or Sample Factory).
- Environments in the same machine pick UDP ports automatically, but very large sweeps can collide. Give each job its own `DOOM_DEFAULT_UDP_PORT` base, for example `12000`, `25000`, `38000`.
- Start environments a few at a time. Many Doom multiplayer games starting at once can time out during network synchronisation.

## Rewards for value-decomposition methods

QMIX, VDN and QPLEX sum per-agent rewards into a team reward themselves. Use `shared_reward=0.0` for them, so the team reward is not counted twice. Actor-critic methods (MAPPO, HAPPO) usually want the scenario default or `shared_reward=1.0`.
