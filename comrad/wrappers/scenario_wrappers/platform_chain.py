import gymnasium as gym
import math


class PlatformChainRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        progress_reward_per_level=1.0,
        chain_break_penalty=-0.05,
        drag_penalty=-0.0025,
    ):
        super().__init__(env)
        self.progress_reward_per_level = float(progress_reward_per_level)
        self.chain_break_penalty = float(chain_break_penalty)
        self.drag_penalty = float(drag_penalty)

        self.prev_chain_break_events = 0
        self.best_joint_progress = 0.0
        self.orig_env_reward = 0.0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        return 1.0 / float(self._num_agents())

    @staticmethod
    def _int_stat(info, key: str, default: int = 0) -> int:
        return int(max(0, float(info.get(key, default))))

    def _joint_progress_levels(self, info) -> float:
        if info is None:
            return 0.0

        if "USER55" in info:
            return max(0.0, float(info.get("USER55", 0.0)) / 65536.0)

        return float(self._int_stat(info, "USER54"))

    @staticmethod
    def _checkpoint(progress: float) -> int:
        return max(0, int(math.floor(progress + 1e-6)))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0

        if info is None:
            self.prev_chain_break_events = 0
            self.best_joint_progress = 0.0
            return obs, info

        self.prev_chain_break_events = self._int_stat(info, "USER52")
        self.best_joint_progress = self._joint_progress_levels(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None:
            reward = 0.0
        reward = float(reward)
        self.orig_env_reward += reward

        if info is None:
            return obs, reward, terminated, truncated, info

        prev_best_joint_progress = self.best_joint_progress
        curr_joint_progress = self._joint_progress_levels(info)
        if curr_joint_progress > self.best_joint_progress:
            self.best_joint_progress = curr_joint_progress

        curr_break_events = self._int_stat(info, "USER52")
        curr_dragged_links = self._int_stat(info, "USER53")

        shaped_team_reward = 0.0

        delta_best_joint_progress = self.best_joint_progress - prev_best_joint_progress
        if delta_best_joint_progress > 0.0:
            shaped_team_reward += delta_best_joint_progress * self.progress_reward_per_level

        delta_chain_breaks = max(0, curr_break_events - self.prev_chain_break_events)
        if delta_chain_breaks > 0 and self.chain_break_penalty != 0.0:
            shaped_team_reward += delta_chain_breaks * self.chain_break_penalty

        if curr_dragged_links > 0 and self.drag_penalty != 0.0:
            shaped_team_reward += curr_dragged_links * self.drag_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        info["true_objective"] = float(self._checkpoint(self.best_joint_progress))
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self.prev_chain_break_events = curr_break_events
        return obs, total_reward, terminated, truncated, info
