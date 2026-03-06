from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym


class SharedRewardWrapper(gym.Wrapper):
    def __init__(self, env, *, alpha: float = 1.0, scalarisation: str = "sum"):
        super().__init__(env)

        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"reward_alpha must be in [0, 1], not {alpha}")
        if scalarisation not in ("sum", "mean"):
            raise ValueError(f"reward scalarisation must be 'sum' or 'mean', not {scalarisation!r}")

        self.alpha = alpha
        self.scalarisation = scalarisation

    def agg(self, rewards: Sequence[float]) -> float:
        team_reward = float(sum(rewards))
        if self.scalarisation == "mean":
            team_reward /= len(rewards)
        return team_reward

    def step(self, action):
        obs, rewards, terminated, truncated, infos = self.env.step(action)

        if rewards is None:
            return obs, rewards, terminated, truncated, infos

        if not isinstance(rewards, Sequence):
            raise TypeError(f"Expected a reward sequence from the wrapped multi-agent env, got {type(rewards)!r}")
        if len(rewards) == 0:
            return obs, rewards, terminated, truncated, infos

        reward_list = [float(reward) for reward in rewards]
        team_reward = self.agg(reward_list)
        blended_rewards = [(1.0 - self.alpha) * reward + self.alpha * team_reward for reward in reward_list]

        return obs, blended_rewards, terminated, truncated, infos
