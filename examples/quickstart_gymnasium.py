import gymnasium as gym

import comrad  # noqa: F401  (registers the comrad/<scenario>-v0 ids)

env = gym.make("comrad/stag_hunt_arena-v0", num_agents=2)
observations, infos = env.reset(seed=42)  # one entry per agent

done = False
while not done:
    actions = [env.action_space.sample() for _ in observations]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    done = all(t or tr for t, tr in zip(terminated, truncated))

print("score:", [info.get("true_objective") for info in infos])
env.close()
