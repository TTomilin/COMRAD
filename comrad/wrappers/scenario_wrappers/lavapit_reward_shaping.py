import gymnasium as gym
import numpy as np


class LavapitAdditionalInput(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        low = np.array([0.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": current_obs_space,
                "measurements": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            }
        )
        self.measurements_vec = np.zeros(1, dtype=np.float32)

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}
        if info is None:
            self.measurements_vec.fill(0.0)
            return obs_dict

        self.measurements_vec[0] = float(bool(info.get("USER23", 0)))
        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.measurements_vec.fill(0.0)
        return self._parse_info(obs, info), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, rew, terminated, truncated, info
        return self._parse_info(obs, info), rew, terminated, truncated, info


class LavapitRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        entry_support_reward: float = 0.05,
        outbound_bridge_reward: float = 0.10,
        exit_support_reward: float = 0.10,
        return_bridge_reward: float = 0.10,
        joint_progress_reward: float = 0.75,
        success_bonus: float = 2.0,
        failure_penalty: float = -1.25,
        unfinished_bridge_penalty: float = -0.12,
        step_penalty: float = -0.002,
        frontier_stall_grace_steps: int = 16,
        frontier_stall_penalty: float = -0.02,
        timeout_penalty: float = -1.50,
    ):
        super().__init__(env)
        self.entry_support_reward = float(entry_support_reward)
        self.outbound_bridge_reward = float(outbound_bridge_reward)
        self.exit_support_reward = float(exit_support_reward)
        self.return_bridge_reward = float(return_bridge_reward)
        self.joint_progress_reward = float(joint_progress_reward)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.unfinished_bridge_penalty = float(unfinished_bridge_penalty)
        self.step_penalty = float(step_penalty)
        self.frontier_stall_grace_steps = max(0, int(frontier_stall_grace_steps))
        self.frontier_stall_penalty = float(frontier_stall_penalty)
        self.timeout_penalty = float(timeout_penalty)

        self.prev_joint_platform = 0
        self.frontier_stage_claimed = 0
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

    def _frontier_entry_held(self, info) -> bool:
        return bool(self._int(info, "USER20", 0))

    def _frontier_bridge_occupied(self, info) -> bool:
        return bool(self._int(info, "USER21", 0))

    def _frontier_exit_held(self, info) -> bool:
        return bool(self._int(info, "USER22", 0))

    def _joint_platform(self, info) -> int:
        if info is None or not self._has_progress_contract(info):
            return 0

        platforms = self._player_best_platforms(info)
        if not platforms:
            return 0
        return min(platforms)

    def _true_objective(self, info) -> float:
        return float(max(0, self._joint_platform(info) - 1))

    def _remaining_bridges(self, joint_platform: int, goal_platform: int) -> int:
        return max(0, goal_platform - max(1, joint_platform))

    def _frontier_stage(self, info, joint_platform: int, goal_platform: int) -> int:
        if not (0 < joint_platform < goal_platform):
            return 0

        entry_held = self._frontier_entry_held(info)
        bridge_occupied = self._frontier_bridge_occupied(info)
        exit_held = self._frontier_exit_held(info)

        if exit_held:
            return 4 if bridge_occupied else 3
        if entry_held:
            return 2 if bridge_occupied else 1
        return 0

    def _frontier_engaged(self, info, joint_platform: int, goal_platform: int) -> bool:
        return self._frontier_stage(info, joint_platform, goal_platform) > 0

    def _frontier_stage_reward(self, stage: int) -> float:
        if stage == 1:
            return self.entry_support_reward
        if stage == 2:
            return self.outbound_bridge_reward
        if stage == 3:
            return self.exit_support_reward
        if stage == 4:
            return self.return_bridge_reward
        return 0.0

    def _reset_episode_state(self) -> None:
        self.prev_joint_platform = 0
        self.frontier_stage_claimed = 0
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
            self.frontier_stage_claimed = 0
            self.frontier_stall_steps = 0
        else:
            frontier_stage = self._frontier_stage(info, curr_joint_platform, goal_platform)
            if frontier_stage > self.frontier_stage_claimed:
                for stage in range(self.frontier_stage_claimed + 1, frontier_stage + 1):
                    shaped_team_reward += self._frontier_stage_reward(stage)
                self.frontier_stage_claimed = frontier_stage
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

        if (terminated or truncated) and not success:
            shaped_team_reward += (
                self.failure_penalty if terminated else self.timeout_penalty
            )
            shaped_team_reward += self._remaining_bridges(
                curr_joint_platform, goal_platform
            ) * self.unfinished_bridge_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        info["true_objective"] = self._true_objective(info)
        info["success"] = success
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
