import gymnasium as gym


class LavapitRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        frontier_support_reward: float = 0.5,
        joint_progress_reward: float = 1.0,
        handoff_reward: float = 0.25,
        success_bonus: float = 2.0,
        failure_penalty: float = -1.5,
        step_penalty: float = -0.002,
        timeout_penalty: float = -0.5,
    ):
        super().__init__(env)
        self.frontier_support_reward = float(frontier_support_reward)
        self.joint_progress_reward = float(joint_progress_reward)
        self.handoff_reward = float(handoff_reward)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.step_penalty = float(step_penalty)
        self.timeout_penalty = float(timeout_penalty)

        self.prev_joint_platform = 0
        self.prev_active_bridge = 0
        self.rewarded_frontier_supports: set[int] = set()
        self.rewarded_handoffs: set[int] = set()
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

    def _active_bridge(self, info) -> int:
        return max(0, self._int(info, "USER13", 0))

    def _player_best_platforms(self, info) -> list[int]:
        num_agents = self._num_agents()
        return [max(0, self._int(info, f"USER{15 + agent_idx}", 0)) for agent_idx in range(num_agents)]

    def _joint_platform(self, info) -> int:
        if info is None or not self._has_progress_contract(info):
            return 0

        platforms = self._player_best_platforms(info)
        if not platforms:
            return 0
        return min(platforms)

    def _lead_platform(self, info) -> int:
        if info is None or not self._has_progress_contract(info):
            return 0

        platforms = self._player_best_platforms(info)
        if not platforms:
            return 0
        return max(platforms)

    def _true_objective(self, info) -> float:
        return float(max(0, self._joint_platform(info) - 1))

    def _sync(self, info) -> None:
        self.prev_joint_platform = self._joint_platform(info)
        self.prev_active_bridge = self._active_bridge(info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0
        self.rewarded_frontier_supports = set()
        self.rewarded_handoffs = set()
        self.prev_joint_platform = 0
        self.prev_active_bridge = 0
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

        shaped_team_reward = self.step_penalty
        contract_available = self._has_progress_contract(info)
        curr_joint_platform = self._joint_platform(info)
        curr_lead_platform = self._lead_platform(info)
        curr_active_bridge = self._active_bridge(info)
        goal_platform = self._goal_platform(info)

        if contract_available and curr_joint_platform == self.prev_joint_platform:
            frontier_bridge = max(1, curr_joint_platform)
            if (
                curr_active_bridge >= frontier_bridge
                and curr_active_bridge > self.prev_active_bridge
                and frontier_bridge not in self.rewarded_frontier_supports
            ):
                shaped_team_reward += self.frontier_support_reward
                self.rewarded_frontier_supports.add(frontier_bridge)

        if contract_available and curr_joint_platform > self.prev_joint_platform:
            shaped_team_reward += (
                curr_joint_platform - self.prev_joint_platform
            ) * self.joint_progress_reward

        if contract_available and curr_lead_platform == curr_joint_platform + 1:
            handoff_bridge = curr_lead_platform - 1
            if handoff_bridge > 0 and handoff_bridge not in self.rewarded_handoffs:
                shaped_team_reward += self.handoff_reward
                self.rewarded_handoffs.add(handoff_bridge)

        success = bool(
            contract_available
            and terminated
            and not truncated
            and goal_platform > 0
            and curr_joint_platform >= goal_platform
        )
        if success:
            shaped_team_reward += self.success_bonus

        if terminated and not success:
            shaped_team_reward += self.failure_penalty

        if truncated and not success:
            shaped_team_reward += self.timeout_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        info["true_objective"] = self._true_objective(info)
        info["success"] = success
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
