# Foraging Commons

![Foraging Commons](../assets/scenarios/foraging_commons.jpg)

Harvesting health restores an agent but depletes a shared resource. A cleanup room restores the commons at a personal opportunity cost. Tests tragedy-of-the-commons dynamics and whether some agents take on the cleanup role.

```python
import comrad

env = comrad.parallel_env("foraging_commons")
```

| | |
|---|---|
| Gymnasium id | `comrad/foraging_commons-v0` |
| Agents | 2 by default (supports 2–4) |
| Capability tested | Commons dilemma, public-goods provision |
| Episode limit | 5250 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector available with `additional_input=True` |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(3))` |
| Default `shared_reward` | `0.0`: each agent receives its own reward |
| Score (`info["true_objective"]`) | Survival time (episode steps) |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. move: none / forward / backward
1. strafe: none / right / left
2. turn: none / left / right

## Reward

Per agent: a small bonus for each own harvest and each cleanup (cleanup requires leaving and re-entering the zone); -1 on death.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- None.
