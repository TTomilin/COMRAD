# Co-op Puzzle

![Co-op Puzzle](../assets/scenarios/coop_puzzle.jpg)

Two players progress through paired lanes by stepping on plates that unlock the other player's path. Periodic stages require both players on their plates at the same time, and the exit is joint.

```python
import comrad

env = comrad.parallel_env("coop_puzzle")
```

| | |
|---|---|
| Gymnasium id | `comrad/coop_puzzle-v0` |
| Agents | 2 by default (supports 2) |
| Capability tested | Turn-taking, partner-specific affordances |
| Episode limit | 2000 steps |
| Observation | RGB image `(72, 128, 3)` per agent; measurement vector not available |
| Action space | `Tuple(Discrete(3), Discrete(2), Discrete(3))` |
| Default `shared_reward` | `1.0`: rewards are fully shared across the team |
| Score (`info["true_objective"]`) | Completed puzzle pairs |

## Actions

Each agent sends one choice per head; index `0` is always "do nothing".

0. turn: none / left / right
1. move: none / forward
2. strafe: none / left / right

## Reward

Bonuses for zone advances, completed pairs and success; a small per-step cost and a penalty for unsuccessful timeouts, split across agents.

The score in `info["true_objective"]` is what the benchmark reports. The shaped reward is only a training signal.

## Variants

- None.
