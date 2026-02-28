https://wandb.ai/mitko-zh-eindhoven-university-of-technology/COMRAD/runs/armory_siege_aggressive_long_v2_20260208_194547_270609?nw=nwusermitkozh


## Dynamic environment (in-map)
positions of things constantly changed, but map layout is not (yet)

## Long horizon task
From my testing learning is most efficient with at least 128 rollout and recurrence 128. Otherwise, learning gets stuck

## Sparse vs dense
In my earlier testing, I tried very dense reward shaping, but it was yielding too much noise and instability when training (grad_norm too high, entropy stuck). Focusing on the kills seems to be the best strategy for agents learning to protect. Needs further testing!!
