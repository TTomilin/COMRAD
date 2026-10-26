# Rhythm Sync

![Rhythm Sync](../assets/scenarios/rhythm_sync.jpg)

Agents in separate rooms must press paired switches inside a shared timing window that is cued on screen. Pressing alone or out of time fails the stage. Some stages give the cue to only one agent, so the other has to infer the timing.

```python
import comrad

env = comrad.parallel_env("rhythm_sync")
```

| | |
|---|---|
| Gymnasium id | `comrad/rhythm_sync-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Temporal coordination, implicit synchronisation |
| Episode limit | 1750 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector available with `additional_input=True` |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(3), Discrete(2))` |
| Default `shared_reward` | `1.0`: rewards are fully shared across the team |
| Score (`info["true_objective"]`) | Stages completed |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. move: none / forward / backward
1. strafe: none / right / left
2. turn: none / left / right
3. use: none / use (press switch)

## Reward

+1 per newly completed stage and -1 per failed stage, split across agents.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- `rhythm_sync_dense`: same map with dense reward shaping.
