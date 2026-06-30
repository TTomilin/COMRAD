import gymnasium as gym
import math


class PlatformChainRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        progress_reward_per_level=1.0,
        chain_break_penalty=-0.05,
        drag_penalty=-0.0025,
        death_penalty=-1.0,
        track_coop=True,
    ):
        super().__init__(env)
        self.progress_reward_per_level = float(progress_reward_per_level)
        self.chain_break_penalty = float(chain_break_penalty)
        self.drag_penalty = float(drag_penalty)
        self.death_penalty = float(death_penalty)

        self.prev_chain_break_events = 0
        self.best_joint_progress = 0.0
        self.prev_health = 0.0
        self.orig_env_reward = 0.0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop
        self._prev_level = 0

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

    @staticmethod
    def _health(info) -> float:
        if info is None:
            return 0.0
        return float(info.get("HEALTH", 0.0))

    def _player_id(self) -> int:
        return int(max(0, getattr(self.env.unwrapped, "player_id", 0)))

    def _own_checkpoint(self, info) -> int:
        return self._int_stat(info, f"USER{60 + self._player_id()}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self._prev_level = 0
        self._prev_own_checkpoint = 0

        if info is None:
            self.prev_chain_break_events = 0
            self.best_joint_progress = 0.0
            self.prev_health = 0.0
            return obs, info

        self.prev_chain_break_events = self._int_stat(info, "USER52")
        self.best_joint_progress = self._joint_progress_levels(info)
        self.prev_health = self._health(info)
        self._prev_level = self._int_stat(info, "USER54")
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

        shaped_reward = shaped_team_reward * self._reward_share()

        curr_health = self._health(info)
        if curr_health <= 0.0 and self.prev_health > 0.0 and self.death_penalty != 0.0:
            shaped_reward += self.death_penalty

        total_reward = reward + shaped_reward

        info["true_objective"] = float(self._checkpoint(self.best_joint_progress))
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        curr_level = self._int_stat(info, "USER54")
        curr_own_checkpoint = self._own_checkpoint(info)
        delta_progress = max(0.0, curr_joint_progress - prev_best_joint_progress)
        delta_breaks = max(0, curr_break_events - self.prev_chain_break_events)
        delta_own_checkpoint = curr_own_checkpoint - self._prev_own_checkpoint

        if self.track_coop:
            coop = 1.0 if delta_own_checkpoint > 0 else 0.0
            defect = 1.0 if delta_breaks > 0 else 0.0

            info["coop_step_signal"] = coop
            info["defect_step_signal"] = defect

            self.episode_coop_steps += coop
            self.episode_defect_steps += defect

            if terminated or truncated:
                self._record_episode_stats(info)

        self.prev_chain_break_events = curr_break_events
        self._prev_level = curr_level
        self._prev_own_checkpoint = curr_own_checkpoint
        return obs, total_reward, terminated, truncated, info

    def _record_episode_stats(self, info):
        extra = info.setdefault("episode_extra_stats", {})
        total = self.episode_coop_steps + self.episode_defect_steps
        extra["coop_steps"] = self.episode_coop_steps
        extra["defect_steps"] = self.episode_defect_steps
        extra["total_coop_defect_steps"] = total
        if total > 0:
            extra["cooperation_index"] = self.episode_coop_steps / total
            extra["defector_index"] = self.episode_defect_steps / total
