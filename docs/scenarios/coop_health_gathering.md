# Co-op Health Gathering

![Co-op Health Gathering](../assets/scenarios/coop_health_gathering.jpg)

Tethered players survive in a toxic maze that damages them continuously. Health kits are scarce, so the team has to explore, collect and manage chain slack together.

```python
import comrad

env = comrad.parallel_env("coop_health_gathering")
```

| | |
|---|---|
| Gymnasium id | `comrad/coop_health_gathering-v0` |
| Agents | 2 by default (supports 2–8) |
| Capability tested | Resource sharing under continuous damage |
| Episode limit | 2100 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector not available |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(3))` |
| Default `shared_reward` | `0.0`: each agent receives its own reward |
| Score (`info["true_objective"]`) | Survival time (episode steps) |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. move: none / forward / backward
1. strafe: none / right / left
2. turn: none / left / right

## Reward

Bonuses for health gained and kits picked up; penalties for health lost, death and stretching the chain.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- None.
