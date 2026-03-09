'''
Reference from https://openreview.net/pdf?id=rkltE0VKwH
Agents must all collect the same treasure. The first agent to collect a treasure during an episode determines the goal for the rest of the agents.
'''
import gymnasium as gym


class DoomMWHRewardShaping(gym.Wrapper):
    # goal for all agents in that env/episode
    # has type {id(env): type}, type is either 1 or 2
    goal = {}

    def __init__(self, env, task=1):
        super().__init__(env)
        self.task = task
        self.p_u1 = None
        self.p_u2 = None
        self.orig_env_reward = 0
        self.key = id(env)
        self._episode_shaped_return = 0.0

    def _reward_shaping(self, info, done):
        if info is None or done:
            return 0.0

        u1 = info.get("USER1", 0)
        u2 = info.get("USER2", 0)
        d1 = u1 - (self.p_u1 or 0)
        d2 = u2 - (self.p_u2 or 0)
        reward = 0

        # Determine goal type for all agents, type index corresponds to user
        # +10 for correct treasure collection
        # -0.2 every game tics
        # If no treasure collected in a step type stays at None
        type = None
        if d1 > 0:
            type = 1
        elif d2 > 0:
            type = 2

        if type is not None:
            key = DoomMWHRewardShaping.goal.get(self.key, None)
            if key is None:
                DoomMWHRewardShaping.goal[self.key] = type
                key = type
            if type == key:
                reward = 10

        self.p_u1 = u1
        self.p_u2 = u2
        return reward

    def reset(self, **kwargs):
        self.p_u1 = None
        self.p_u2 = None
        self.orig_env_reward = 0
        self._episode_shaped_return = 0.0
        if self.key in DoomMWHRewardShaping.goal:
            del DoomMWHRewardShaping.goal[self.key]
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if reward is None:
            return observation, reward, terminated, truncated, info

        reward = float(reward)
        self.orig_env_reward += reward

        done = terminated | truncated
        reward += self._reward_shaping(info, done)

        if done:
            true_objective = self.orig_env_reward
            info["true_objective"] = true_objective

        return observation, reward, terminated, truncated, info
