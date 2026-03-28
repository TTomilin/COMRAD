import gymnasium as gym


class RhythmSyncRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        stage_completion_reward=1.0,
        failure_penalty=-1.0,
        completion_bonus=0.0,
    ):
        super().__init__(env)
        self.stage_completion_reward = float(stage_completion_reward)
        self.failure_penalty = float(failure_penalty)
        self.completion_bonus = float(completion_bonus)

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.episode_shaped_return = 0.0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        """Reards are duplicated per player (in parallel corridor), so should be divided.
        Off policy will sum them up again later. On policy will sum with alpha=1 or use bare reward if alpha=0"""
        return 1.0 / float(self._num_agents())

    @staticmethod
    def _flag(info, key: str) -> float:
        return 1.0 if float(info.get(key, 0.0)) > 0.0 else 0.0

    def _completed_stages(self, info) -> float:
        if info is None:
            return 0.0

        if "USER25" in info:
            return max(0.0, float(info.get("USER25", 0.0)))

        current_stage = max(1.0, float(info.get("USER24", 1.0)))
        finished = self._flag(info, "USER30")
        return max(0.0, current_stage - 1.0 + finished)

    def _sync(self, info):
        if info is None:
            self.prev_vars = {}
            return

        self.prev_vars = {
            "completed_stages": self._completed_stages(info),
            "failed": self._flag(info, "USER29"),
            "finished": self._flag(info, "USER30"),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.episode_shaped_return = 0.0
        self._sync(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None:
            reward = 0.0
        reward = float(reward)
        self.orig_env_reward += reward

        if info is None:
            self.episode_shaped_return += reward
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self._sync(info)
            if terminated or truncated:
                info["true_objective"] = self._completed_stages(info)
                info["orig_env_reward"] = self.orig_env_reward
            self.episode_shaped_return += reward
            return obs, reward, terminated, truncated, info

        curr_completed = self._completed_stages(info)
        prev_completed = self.prev_vars.get("completed_stages", 0.0)
        curr_failed = self._flag(info, "USER29")
        prev_failed = self.prev_vars.get("failed", 0.0)
        curr_finished = self._flag(info, "USER30")
        prev_finished = self.prev_vars.get("finished", 0.0)

        shaped_reward = 0.0
        share = self._reward_share()

        delta_completed = curr_completed - prev_completed
        if delta_completed > 0.0:
            shaped_reward += delta_completed * self.stage_completion_reward * share

        if curr_finished > prev_finished and self.completion_bonus != 0.0:
            shaped_reward += self.completion_bonus * share

        if curr_failed > prev_failed and self.failure_penalty != 0.0:
            shaped_reward += self.failure_penalty * share

        total_reward = reward + shaped_reward
        self.episode_shaped_return += total_reward

        if terminated or truncated:
            info["true_objective"] = curr_completed
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
