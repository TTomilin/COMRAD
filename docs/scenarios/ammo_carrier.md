# Ammo Carrier

![Ammo Carrier](../assets/scenarios/ammo_carrier.jpg)

Fixed asymmetric roles: an immobile defender holds the hub against enemies, and a runner keeps the defender supplied with ammunition from depots.

```python
import comrad

env = comrad.parallel_env("ammo_carrier")
```

| | |
|---|---|
| Gymnasium id | `comrad/ammo_carrier-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Role-specialised logistics, resupply timing |
| Episode limit | 2100 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector available with `additional_input=True` |
| Action space | `Tuple(Discrete(3), Discrete(2), Discrete(3), Discrete(2))` |
| Default `shared_reward` | `0.0`: each agent receives its own reward |
| Score (`info["true_objective"]`) | Survival time (episode steps) |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. turn: none / left / right
1. move: none / forward
2. strafe: none / left / right
3. attack: none / attack

## Reward

Role-specific, per agent: the defender is rewarded for kills and damage, the runner for collecting and delivering ammo; penalties for death. Rewards are not shared by default.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- None.
