import comrad

env = comrad.parallel_env("stag_hunt_arena")
observations, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print("score:", {agent: info.get("true_objective") for agent, info in infos.items()})
env.close()
