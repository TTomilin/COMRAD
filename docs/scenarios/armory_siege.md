# Armory Siege

![Armory Siege](../assets/scenarios/armory_siege.jpg)

The team defends a central core against waves of enemies while rotating through gated weapon, ammo and medical rooms. Agents must split between holding the core and resupplying.

```python
import comrad

env = comrad.parallel_env("armory_siege")
```

| | |
|---|---|
| Gymnasium id | `comrad/armory_siege-v0` |
| Agents | 2 by default (supports 2–8) |
| Capability tested | Combat coordination, resource routing, rotating roles |
| Episode limit | 4500 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector available with `additional_input=True` |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(3), Discrete(2), Discrete(3))` |
| Default `shared_reward` | `0.0`: each agent receives its own reward |
| Score (`info["true_objective"]`) | Shaped episodic return |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. move: none / forward / backward
1. strafe: none / right / left
2. turn: none / left / right
3. attack: none / attack
4. weapon: none / weapon 1 / weapon 2

## Reward

Per agent: bonuses for kills, hits, new weapons, ammo, healing and surviving the timeout; penalties for core damage, core loss, death and firing with no ammo. `true_objective` is the episodic shaped return.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- `comrad/scenarios/batch_armory_siege/` and `batch_armory/`: layout variants for generalisation tests.
