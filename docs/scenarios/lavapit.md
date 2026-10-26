# Lava Pit

![Lava Pit](../assets/scenarios/lavapit.jpg)

To cross each bridge, one agent must hold a support platform while the other crosses. Then the roles swap before the team can move on. Progress is monotone and joint.

```python
import comrad

env = comrad.parallel_env("lavapit")
```

| | |
|---|---|
| Gymnasium id | `comrad/lavapit-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Reciprocal assistance, temporary role asymmetry |
| Episode limit | 1000 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector available with `additional_input=True` |
| Action space | `Tuple(Discrete(3), Discrete(2), Discrete(3))` |
| Default `shared_reward` | `1.0`: rewards are fully shared across the team |
| Score (`info["true_objective"]`) | Joint bridges cleared |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. turn: none / left / right
1. move: none / forward
2. strafe: none / left / right

## Reward

One-time frontier milestones, +0.75 per joint platform advance and +2 on success; penalties for stalling, failure, timeout and unfinished bridges.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- None.
