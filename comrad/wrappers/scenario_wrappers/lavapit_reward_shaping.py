import gymnasium as gym


class LavapitRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        joint_progress_reward: float = 1.0,
        success_bonus: float = 2.0,
        failure_penalty: float = -1.0,
        no_progress_failure_penalty: float = -0.5,
        step_penalty: float = -0.001,
        frontier_stall_grace_steps: int = 12,
        frontier_stall_penalty: float = -0.03,
        timeout_penalty: float = -0.75,
    ):
        super().__init__(env)
        self.joint_progress_reward = float(joint_progress_reward)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.no_progress_failure_penalty = float(no_progress_failure_penalty)
        self.step_penalty = float(step_penalty)
        self.frontier_stall_grace_steps = max(0, int(frontier_stall_grace_steps))
        self.frontier_stall_penalty = float(frontier_stall_penalty)
        self.timeout_penalty = float(timeout_penalty)

        self.prev_joint_platform = 0
        self.frontier_stall_steps = 0
        self.orig_env_reward = 0.0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        return 1.0 / float(self._num_agents())

    @staticmethod
    def _int(info, key: str, default: int = 0) -> int:
        try:
            return int(info.get(key, default))
        except (TypeError, ValueError, OverflowError, AttributeError):
            return default

    def _has_progress_contract(self, info) -> bool:
        return self._int(info, "USER17", 0) > 0

    def _goal_platform(self, info) -> int:
        return max(0, self._int(info, "USER17", 0))

    def _player_best_platforms(self, info) -> list[int]:
        num_agents = self._num_agents()
        return [max(0, self._int(info, f"USER{15 + agent_idx}", 0)) for agent_idx in range(num_agents)]

    def _current_platforms(self, info) -> list[int]:
        num_agents = self._num_agents()
        return [max(0, self._int(info, f"USER{18 + agent_idx}", 0)) for agent_idx in range(num_agents)]

    def _frontier_entry_held(self, info) -> bool:
        return bool(self._int(info, "USER20", 0))

    def _frontier_bridge_occupied(self, info) -> bool:
        return bool(self._int(info, "USER21", 0))

    def _joint_platform(self, info) -> int:
        if info is None or not self._has_progress_contract(info):
            return 0

        platforms = self._player_best_platforms(info)
        if not platforms:
            return 0
        return min(platforms)

    def _true_objective(self, info) -> float:
        return float(max(0, self._joint_platform(info) - 1))

    def _frontier_engaged(self, info, joint_platform: int, goal_platform: int) -> bool:
        if not (0 < joint_platform < goal_platform):
            return False

        current_platforms = self._current_platforms(info)
        split_state = False
        if len(current_platforms) >= 2:
            sorted_platforms = sorted(current_platforms[:2])
            split_state = sorted_platforms == [joint_platform, joint_platform + 1]

        return self._frontier_entry_held(info) or self._frontier_bridge_occupied(info) or split_state

    def _reset_episode_state(self) -> None:
        self.prev_joint_platform = 0
        self.frontier_stall_steps = 0

    def _sync(self, info) -> None:
        self.prev_joint_platform = self._joint_platform(info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0
        self._reset_episode_state()
        self._sync(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None:
            reward = 0.0
        reward = float(reward)
        self.orig_env_reward += reward

        if info is None:
            return obs, reward, terminated, truncated, info

        if not self._has_progress_contract(info):
            info["true_objective"] = 0.0
            info["success"] = False
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            return obs, reward, terminated, truncated, info

        curr_joint_platform = self._joint_platform(info)
        goal_platform = self._goal_platform(info)

        shaped_team_reward = self.step_penalty

        joint_delta = curr_joint_platform - self.prev_joint_platform
        if joint_delta > 0:
            shaped_team_reward += joint_delta * self.joint_progress_reward
            self.frontier_stall_steps = 0
        elif self._frontier_engaged(info, curr_joint_platform, goal_platform):
            self.frontier_stall_steps += 1
            if self.frontier_stall_steps > self.frontier_stall_grace_steps:
                shaped_team_reward += self.frontier_stall_penalty
        else:
            self.frontier_stall_steps = 0

        success = bool(
            terminated
            and not truncated
            and goal_platform > 0
            and curr_joint_platform >= goal_platform
        )
        if success:
            shaped_team_reward += self.success_bonus

        if terminated and not success:
            shaped_team_reward += self.failure_penalty
            if curr_joint_platform <= 1:
                shaped_team_reward += self.no_progress_failure_penalty

        if truncated and not success:
            shaped_team_reward += self.timeout_penalty
            if curr_joint_platform <= 1:
                shaped_team_reward += self.no_progress_failure_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        info["true_objective"] = self._true_objective(info)
        info["success"] = success
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
