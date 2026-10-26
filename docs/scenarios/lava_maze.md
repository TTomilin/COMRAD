# Lava Maze

![Lava Maze](../assets/scenarios/lava_maze.jpg)

A navigator crosses a lava field along a hidden safe path. An observer looks down on the maze and can only help by sending colour signals. The roles have masked action sets.

```python
import comrad

env = comrad.parallel_env("lava_maze")
```

| | |
|---|---|
| Gymnasium id | `comrad/lava_maze-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Emergent communication, asymmetric information |
| Episode limit | 5250 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector available with `additional_input=True` |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(2), Discrete(5))` |
| Default `shared_reward` | `1.0`: rewards are fully shared across the team |
| Score (`info["true_objective"]`) | Maze levels cleared |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. move: none / forward / backward
1. strafe: none / right / left
2. attack: none / attack
3. signal: none / colour 1 / 2 / 3 / 4 (weapon select)

The navigator and the observer have role-specific action masks.

## Reward

+10 per cleared level and a bonus for every step closer to the exit; penalties for each step, signals, burns and death.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- `lava_maze_simple`: smaller, easier maze.
- `comrad/scenarios/batch_lava_maze/`: pool of generated mazes.
