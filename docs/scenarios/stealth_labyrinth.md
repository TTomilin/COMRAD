# Stealth Labyrinth

![Stealth Labyrinth](../assets/scenarios/stealth_labyrinth.jpg)

A scout that cannot attack carries the light, and a gunner that cannot see in the dark does the fighting. Combat rooms activate one after another, so the pair has to move and fight as a unit.

```python
import comrad

env = comrad.parallel_env("stealth_labyrinth")
```

| | |
|---|---|
| Gymnasium id | `comrad/stealth_labyrinth-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Tightly coupled movement, extreme role asymmetry |
| Episode limit | 5250 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector not available |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(3), Discrete(2))` |
| Default `shared_reward` | `1.0`: rewards are fully shared across the team |
| Score (`info["true_objective"]`) | Enemies destroyed / total enemies |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. move: none / forward / backward
1. strafe: none / right / left
2. turn: none / left / right
3. attack: none / attack

## Reward

Bonuses for milestones, room discoveries and kills (+2); penalties for team health loss and any unsuccessful termination.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- None.
