import gymnasium as gym


class StealthLabyrinthRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        room_discovery_reward=0.3,
        first_room_discovery_reward=0.8,
        progress_milestone_reward=0.4,
        kill_reward=2.0,
        damage_taken_penalty_per_hp=-0.02,
        death_penalty=-1.0,
        track_coop=True,
    ):
        super().__init__(env)
        self.room_discovery_reward = float(room_discovery_reward)
        self.first_room_discovery_reward = float(first_room_discovery_reward)
        self.progress_milestone_reward = float(progress_milestone_reward)
        self.kill_reward = float(kill_reward)
        self.damage_taken_penalty_per_hp = float(damage_taken_penalty_per_hp)
        self.death_penalty = float(death_penalty)

        self.best_destroyed = None
        self.best_rooms_seen = None
        self.best_progress_milestones = None
        self.prev_team_hp = None
        self.orig_env_reward = 0.0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop

        self._prev_destroyed = 0
        self._prev_alive = True

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

    @staticmethod
    def _float(info, key: str, default: float = 0.0) -> float:
        try:
            return float(info.get(key, default))
        except (TypeError, ValueError, OverflowError):
            return default

    def _true_objective(self, info) -> float:
        total = max(0, self._int(info, "USER47"))
        destroyed = max(0, self._int(info, "USER46"))
        if total <= 0:
            return 0.0
        return min(1.0, destroyed / float(total))

    def _sync(self, info) -> None:
        if info is None:
            self.best_destroyed = None
            self.best_rooms_seen = None
            self.best_progress_milestones = None
            self.prev_team_hp = None
            return
        self.best_destroyed = max(0, self._int(info, "USER46"))
        self.best_rooms_seen = max(0, self._int(info, "USER48"))
        self.best_progress_milestones = max(0, self._int(info, "USER51"))
        self.prev_team_hp = max(0.0, self._float(info, "USER50"))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if info is not None and self.track_coop:
            self._record_episode_stats(info)
        self.best_destroyed = None
        self.best_rooms_seen = None
        self.best_progress_milestones = None
        self.prev_team_hp = None
        self.orig_env_reward = 0.0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0

        self._prev_destroyed = 0
        self._prev_alive = True
        self._sync(info)
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
            reward = 0.0
        reward = float(reward)
        self.orig_env_reward += reward

        if info is None:
            return obs, reward, terminated, truncated, info

        destroyed = max(0, self._int(info, "USER46"))
        rooms_seen = max(0, self._int(info, "USER48"))
        progress_milestones = max(0, self._int(info, "USER51"))
        remaining = max(0, self._int(info, "USER44"))
        total = max(0, self._int(info, "USER47"))
        team_hp = max(0.0, self._float(info, "USER50"))
        p1_alive = max(0, self._int(info, "USER41"))
        p2_alive = max(0, self._int(info, "USER42"))

        if (
            self.best_destroyed is None
            or self.best_rooms_seen is None
            or self.best_progress_milestones is None
            or self.prev_team_hp is None
        ):
            self._sync(info)
            info["true_objective"] = self._true_objective(info)
            info["success"] = False
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            return obs, reward, terminated, truncated, info

        delta_destroyed = max(0, destroyed - self.best_destroyed)
        delta_rooms_seen = max(0, rooms_seen - self.best_rooms_seen)
        delta_progress_milestones = max(0, progress_milestones - self.best_progress_milestones)
        delta_team_damage = max(0.0, self.prev_team_hp - team_hp)

        success = bool(
            terminated
            and not truncated
            and total > 0
            and remaining == 0
            and p1_alive > 0
            and p2_alive > 0
        )

        shaped_team_reward = 0.0
        shaped_team_reward += delta_progress_milestones * self.progress_milestone_reward
        if self.best_rooms_seen < 1 and rooms_seen >= 1:
            shaped_team_reward += self.first_room_discovery_reward
            delta_rooms_seen = max(0, delta_rooms_seen - 1)
        shaped_team_reward += delta_rooms_seen * self.room_discovery_reward
        shaped_team_reward += delta_destroyed * self.kill_reward
        shaped_team_reward += delta_team_damage * self.damage_taken_penalty_per_hp

        if terminated and not success:
            shaped_team_reward += self.death_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        if (
            destroyed < self.best_destroyed
            or rooms_seen < self.best_rooms_seen
            or progress_milestones < self.best_progress_milestones
        ):
            info.setdefault("episode_extra_stats", {})["counter_regression"] = 1

        self.best_destroyed = max(self.best_destroyed, destroyed)
        self.best_rooms_seen = max(self.best_rooms_seen, rooms_seen)
        self.best_progress_milestones = max(self.best_progress_milestones, progress_milestones)
        self.prev_team_hp = team_hp

        info["true_objective"] = self._true_objective(info)
        info["success"] = success
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        player_id = getattr(self.env.unwrapped, "player_id", -1)
        is_alive = (player_id == 0 and p1_alive > 0) or (player_id == 1 and p2_alive > 0)
        both_alive = p1_alive > 0 and p2_alive > 0
        delta_destroyed = max(0, destroyed - self._prev_destroyed)
        delta_rooms_seen = max(0, rooms_seen - self.best_rooms_seen) if self.best_rooms_seen is not None else 0
        delta_progress = max(0, progress_milestones - self.best_progress_milestones) if self.best_progress_milestones is not None else 0
        delta_team_damage = max(0.0, self.prev_team_hp - team_hp) if self.prev_team_hp is not None else 0.0

        if self.track_coop:
            made_contribution = delta_destroyed > 0 or delta_rooms_seen > 0 or delta_progress > 0
            took_damage_no_kill = delta_team_damage > 0 and delta_destroyed == 0

            coop = 1.0 if made_contribution and both_alive else 0.0
            defect = 1.0 if not is_alive or took_damage_no_kill else 0.0
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

        self._prev_destroyed = destroyed
        self._prev_alive = is_alive

        return obs, total_reward, terminated, truncated, info
