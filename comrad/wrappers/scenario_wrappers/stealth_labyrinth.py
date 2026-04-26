import gymnasium as gym


class StealthLabyrinthRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        relay_completion_reward=1.0,
        sync_progress_reward=0.01,
        extraction_reward=1.5,
        death_penalty=-1.0,
    ):
        super().__init__(env)
        self.relay_completion_reward = float(relay_completion_reward)
        self.sync_progress_reward = float(sync_progress_reward)
        self.extraction_reward = float(extraction_reward)
        self.death_penalty = float(death_penalty)

        self.prev_relays_done = None
        self.prev_p1_alive = None
        self.prev_p2_alive = None
        self.best_progress_by_objective = {}
        self.orig_env_reward = 0.0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        return 1.0 / float(self._num_agents())

    @staticmethod
    def _int(info, key: str, default: int = 0) -> int:
        try:
            return int(info.get(key, default))
        except (TypeError, ValueError, OverflowError):
            return default

    def _sync(self, info):
        if info is None:
            self.prev_relays_done = None
            self.prev_p1_alive = None
            self.prev_p2_alive = None
            return

        self.prev_relays_done = max(0, self._int(info, "USER44"))
        active_objective = self._int(info, "USER45", -1)
        objective_progress = max(0, self._int(info, "USER46"))
        self.prev_p1_alive = max(0, self._int(info, "USER41"))
        self.prev_p2_alive = max(0, self._int(info, "USER42"))

        if active_objective >= 0 and objective_progress > 0:
            best_progress = max(0, int(self.best_progress_by_objective.get(active_objective, 0)))
            if objective_progress > best_progress:
                self.best_progress_by_objective[active_objective] = objective_progress

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_relays_done = None
        self.prev_p1_alive = None
        self.prev_p2_alive = None
        self.best_progress_by_objective = {}
        self.orig_env_reward = 0.0
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

        curr_relays_done = max(0, self._int(info, "USER44"))
        curr_active_objective = self._int(info, "USER45", -1)
        curr_objective_progress = max(0, self._int(info, "USER46"))
        curr_extraction_unlocked = max(0, self._int(info, "USER47"))
        curr_sync_state = max(0, self._int(info, "USER48"))
        curr_p1_alive = max(0, self._int(info, "USER41"))
        curr_p2_alive = max(0, self._int(info, "USER42"))

        if (
            self.prev_relays_done is None
            or self.prev_p1_alive is None
            or self.prev_p2_alive is None
        ):
            info["true_objective"] = float(curr_relays_done)
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            self._sync(info)
            return obs, reward, terminated, truncated, info

        shaped_team_reward = 0.0

        delta_relays = curr_relays_done - self.prev_relays_done
        if delta_relays > 0:
            shaped_team_reward += delta_relays * self.relay_completion_reward

        # ACS progress resets to zero when agents leave the valid room, so rewarding raw
        # deltas would let policies farm shaping by repeatedly partial-holding and resetting.
        # Reward only new best progress achieved on each objective within the episode.
        if curr_sync_state > 0 and curr_active_objective >= 0:
            best_progress = max(0, int(self.best_progress_by_objective.get(curr_active_objective, 0)))
            if curr_objective_progress > best_progress:
                progress_delta = curr_objective_progress - best_progress
                shaped_team_reward += progress_delta * self.sync_progress_reward
                self.best_progress_by_objective[curr_active_objective] = curr_objective_progress

        prev_team_alive = self.prev_p1_alive > 0 and self.prev_p2_alive > 0
        curr_team_alive = curr_p1_alive > 0 and curr_p2_alive > 0
        if prev_team_alive and not curr_team_alive:
            shaped_team_reward += self.death_penalty

        success = bool(
            terminated
            and not truncated
            and curr_team_alive
            and curr_extraction_unlocked > 0
        )
        if success:
            shaped_team_reward += self.extraction_reward

        total_reward = reward + shaped_team_reward * self._reward_share()

        info["true_objective"] = float(curr_relays_done + (1 if success else 0))
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
