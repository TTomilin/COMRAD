# Parallel scenario

This is log for reward design iterations

## Static environment

Same environment, no randomization

### Attempt 1

Rewards:

```python
new_zone: float = 1.0,
old_zone: float = -1.0,
enter_plate: float = 5.0,
leave_plate: float = -3.0,
both_next_zone: float = 10.0,
finished: float = 50.0,
timeout: float = -5.0,
last_zone: int = 6,
```

WB: [Link](https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/runs/parallel_MAPPO_20251209_172820_20251209_172820_766941?nw=nwuserbaokhoi136)

Eval: Pretty much stuck at agent 2 going to new room, agent 1 gets stuck in first room as agent 2 doesn't learn to stand on the plate. And episode is always 34 seconds as there are no ways now for agents to die. This doesn't work, learning signal is sparse.

Next:
+ Deal with fixed timeout.
+ Randomize plates ACS.
+ Maybe merge 2 rooms, sequential and to go to zone 3, they must both stand on the plate.

### Attempt 2

+ More dense reward function, proper zone rewarding

WB: [Link](https://wandb.ai/khoi-eindhoven-university-of-technology/marl_vizdoom/runs/parallel_MAPPO_20251210_150116_20251210_150116_719767?nw=nwuserbaokhoi136)

Eval: This current reward design converges to agent 2 runs toward the plate of agent 1 and agent 1 runs to a corner

## Randomized plate placement

Non-static scenario

### Attempt 1
