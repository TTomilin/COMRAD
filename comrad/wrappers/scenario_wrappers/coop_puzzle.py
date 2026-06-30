import gymnasium as gym


class CoopPuzzleRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        zone_advance_reward: float = 0.75,
        pair_completion_reward: float = 1.0,
        success_bonus: float = 2.0,
        timeout_penalty: float = -1.0,
        step_penalty: float = -0.002,
        last_zone: int | None = None,
        track_coop=True,
    ):
        super().__init__(env)
        self.zone_advance_reward = float(zone_advance_reward)
        self.pair_completion_reward = float(pair_completion_reward)
        self.success_bonus = float(success_bonus)
        self.timeout_penalty = float(timeout_penalty)
        self.step_penalty = float(step_penalty)
        self.last_zone_override = None if last_zone is None else int(last_zone)

        self.prev_team_max_zone = 0
        self.prev_completed_pairs = 0
        self.orig_env_reward = 0.0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        """Team rewards are surfaced once per player env, so divide before optional shared aggregation."""
        return 1.0 / float(self._num_agents())

    @staticmethod
    def _safe_int(info, key: str, default: int = 0) -> int:
        try:
            return int(info.get(key, default))
        except (TypeError, ValueError, OverflowError, AttributeError):
            return default

    def _has_per_player_progress(self, info) -> bool:
        # USER6 is exported only by the fixed WAD and acts as the contract
        # version gate for USER4/USER5 and environment-derived last_zone.
        return self._safe_int(info, "USER6", 0) > 0

    def _zones(self, info) -> tuple[int, int]:
        if not self._has_per_player_progress(info):
            return 0, 0

        zone_a = self._safe_int(info, "USER4", 0)
        zone_b = self._safe_int(info, "USER5", 0)
        return zone_a, zone_b

    def _team_max_zone(self, info, zone_a: int, zone_b: int) -> int:
        if not self._has_per_player_progress(info):
            return 0
        return max(zone_a, zone_b)

    def _last_zone(self, info) -> int | None:
        if self._has_per_player_progress(info):
            return self._safe_int(info, "USER6", 0)
        return self.last_zone_override

    def _completed_pairs(self, zone_a: int, zone_b: int) -> int:
        return max(0, min(zone_a, zone_b) // 2)

    def _sync_progress(self, info) -> None:
        zone_a, zone_b = self._zones(info)
        self._prev_zone_a = zone_a
        self._prev_zone_b = zone_b
        self.prev_team_max_zone = self._team_max_zone(info, zone_a, zone_b)
        if self._has_per_player_progress(info):
            self.prev_completed_pairs = self._completed_pairs(zone_a, zone_b)
        else:
            self.prev_completed_pairs = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if info is not None and self.track_coop:
            self._record_episode_stats(info)
        self.orig_env_reward = 0.0
        self.prev_team_max_zone = 0
        self.prev_completed_pairs = 0
        self._prev_zone_a = 0
        self._prev_zone_b = 0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        if info is not None:
            self._sync_progress(info)
        return obs, info

    def _record_episode_stats(self, info):
        extra = info.setdefault("episode_extra_stats", {})
        total = self.episode_coop_steps + self.episode_defect_steps
        extra["coop_steps"] = self.episode_coop_steps
        extra["defect_steps"] = self.episode_defect_steps
        extra["total_coop_defect_steps"] = total
        if total > 0:
            extra["cooperation_index"] = self.episode_coop_steps / total
            extra["defector_index"] = self.episode_defect_steps / total

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward is None:
            return obs, reward, terminated, truncated, info

        env_reward = float(reward)
        self.orig_env_reward += env_reward

        if info is None:
            return obs, env_reward, terminated, truncated, info

        share = self._reward_share()
        zone_a, zone_b = self._zones(info)
        team_max_zone = self._team_max_zone(info, zone_a, zone_b)
        completed_pairs = self._completed_pairs(zone_a, zone_b) if self._has_per_player_progress(info) else 0
        last_zone = self._last_zone(info)

        shaped_reward = self.step_penalty * share

        if team_max_zone > self.prev_team_max_zone:
            shaped_reward += (team_max_zone - self.prev_team_max_zone) * self.zone_advance_reward * share

        if completed_pairs > self.prev_completed_pairs:
            shaped_reward += (completed_pairs - self.prev_completed_pairs) * self.pair_completion_reward * share

        success = bool(
            terminated
            and not truncated
            and last_zone is not None
            and zone_a >= last_zone
            and zone_b >= last_zone
        )
        if success:
            shaped_reward += self.success_bonus * share

        if truncated and not success:
            shaped_reward += self.timeout_penalty * share

        info["true_objective"] = float(completed_pairs)
        info["success"] = success

        if self.track_coop:
            delta_zone_a = zone_a - self._prev_zone_a
            delta_zone_b = zone_b - self._prev_zone_b
            player_id = int(max(0, getattr(self.env.unwrapped, "player_id", 0)))
            if player_id == 0:
                my_progress = delta_zone_a > 0
            else:
                my_progress = delta_zone_b > 0
            coop = 1.0 if my_progress else 0.0
            defect = 1.0 if not my_progress and not success else 0.0
            info["coop_step_signal"] = coop
            info["defect_step_signal"] = defect

            self.episode_coop_steps += coop
            self.episode_defect_steps += defect

            if coop > 0.0 or defect > 0.0:
                net_coop = coop - defect
                info.setdefault("episode_extra_stats", {})["cooperation_index"] = max(0.0, net_coop)
                info["episode_extra_stats"]["defector_index"] = max(0.0, -net_coop)

            if terminated or truncated:
                self._record_episode_stats(info)

        self._sync_progress(info)

        return obs, env_reward + shaped_reward, terminated, truncated, info
