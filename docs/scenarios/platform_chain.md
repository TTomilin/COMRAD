# Platform Chain

![Platform Chain](../assets/scenarios/platform_chain.jpg)

Tethered players climb a long route of platforms over a hazard. Moving too far apart breaks the tether, so both must keep close while jumping. Checkpoints only advance in order.

```python
import comrad

env = comrad.parallel_env("platform_chain")
```

| | |
|---|---|
| Gymnasium id | `comrad/platform_chain-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Long-horizon joint movement under a tether |
| Episode limit | 5250 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector not available |
| Action space | `Tuple(Discrete(3), Discrete(3), Discrete(3), Discrete(2))` |
| Default `shared_reward` | `1.0`: rewards are fully shared across the team |
| Score (`info["true_objective"]`) | Joint checkpoint progress (max 47 on the default map) |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. turn: none / left / right
1. move: none / forward / backward
2. strafe: none / left / right
3. jump: none / jump

## Reward

+1 per level of best joint progress; penalties for chain breaks, dragging the chain and death.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- `platform_chain_easy`: shorter, easier route.
- `comrad/scenarios/batch_platform_chain_curriculum/`: curriculum pool used in the paper (see [Reproducing](../training/reproducing.md#wad-batches-and-curricula)).
