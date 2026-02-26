from __future__ import annotations

import gymnasium as gym
import numpy as np


class AgentIDWrapper(gym.ObservationWrapper):
    """
    Appends observations with agent_id for network routing per agent, in this case HAPPO as ordering is important
    If obs space is Box, wraps it into a dict with key 'obs' first
    """

    def __init__(self, env: gym.Env, agent_index: int, num_agents: int):
        super().__init__(env)
        assert 0 <= agent_index < num_agents, (
            f"agent_index={agent_index} out of range for num_agents={num_agents}")
        self.agent_index = agent_index
        self.num_agents = num_agents

        if isinstance(env.observation_space, gym.spaces.Dict):
            spaces = dict(env.observation_space.spaces)
        elif isinstance(env.observation_space, gym.spaces.Box):
            spaces = {"obs": env.observation_space}
        else:
            raise ValueError(f"Unsupported obs space type: {type(env.observation_space)}")

        spaces["agent_id"] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self._was_box = isinstance(env.observation_space, gym.spaces.Box)

    def observation(self, obs):
        if self._was_box:
            obs = {"obs": obs}
        agent_id_onehot = np.zeros(self.num_agents, dtype=np.float32)
        agent_id_onehot[self.agent_index] = 1.0
        obs["agent_id"] = agent_id_onehot
        return obs
