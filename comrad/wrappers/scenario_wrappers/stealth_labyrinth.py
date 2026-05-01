import gymnasium as gym


class StealthLabyrinthRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        room_discovery_reward=0.3,
        first_room_discovery_reward=0.8,
        kill_reward=2.0,
        damage_taken_penalty_per_hp=-0.02,
        death_penalty=-1.0,
    ):
        super().__init__(env)
        self.room_discovery_reward = float(room_discovery_reward)
        self.first_room_discovery_reward = float(first_room_discovery_reward)
        self.kill_reward = float(kill_reward)
        self.damage_taken_penalty_per_hp = float(damage_taken_penalty_per_hp)
        self.death_penalty = float(death_penalty)

        self.best_destroyed = None
        self.best_rooms_seen = None
        self.prev_team_hp = None
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
            self.prev_team_hp = None
            return
        self.best_destroyed = max(0, self._int(info, "USER46"))
        self.best_rooms_seen = max(0, self._int(info, "USER48"))
        self.prev_team_hp = max(0.0, self._float(info, "USER50"))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.best_destroyed = None
        self.best_rooms_seen = None
        self.prev_team_hp = None
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

        destroyed = max(0, self._int(info, "USER46"))
        rooms_seen = max(0, self._int(info, "USER48"))
        remaining = max(0, self._int(info, "USER44"))
        total = max(0, self._int(info, "USER47"))
        team_hp = max(0.0, self._float(info, "USER50"))
        p1_alive = max(0, self._int(info, "USER41"))
        p2_alive = max(0, self._int(info, "USER42"))

        if self.best_destroyed is None or self.best_rooms_seen is None or self.prev_team_hp is None:
            self._sync(info)
            info["true_objective"] = self._true_objective(info)
            info["success"] = False
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            return obs, reward, terminated, truncated, info

        delta_destroyed = max(0, destroyed - self.best_destroyed)
        delta_rooms_seen = max(0, rooms_seen - self.best_rooms_seen)
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
        if self.best_rooms_seen < 1 and rooms_seen >= 1:
            shaped_team_reward += self.first_room_discovery_reward
            delta_rooms_seen = max(0, delta_rooms_seen - 1)
        shaped_team_reward += delta_rooms_seen * self.room_discovery_reward
        shaped_team_reward += delta_destroyed * self.kill_reward
        shaped_team_reward += delta_team_damage * self.damage_taken_penalty_per_hp

        if terminated and not success:
            shaped_team_reward += self.death_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        if destroyed < self.best_destroyed or rooms_seen < self.best_rooms_seen:
            info.setdefault("episode_extra_stats", {})["counter_regression"] = 1

        self.best_destroyed = max(self.best_destroyed, destroyed)
        self.best_rooms_seen = max(self.best_rooms_seen, rooms_seen)
        self.prev_team_hp = team_hp

        info["true_objective"] = self._true_objective(info)
        info["success"] = success
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        return obs, total_reward, terminated, truncated, info
