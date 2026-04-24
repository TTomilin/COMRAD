import gymnasium as gym


class StagHuntArenaRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        rabbit_reward=3.0,
        stag_damage_reward_per_hp=0.005, # 0.01 means average 0.7 reward per shotgun blast
        stag_reward=50.0,
        damage_taken_penalty_per_hp=-0.01,
        death_penalty=-2.0,
        timeout_survival_bonus=1.0,
    ):
        super().__init__(env)
        self.rabbit_reward = float(rabbit_reward)
        self.stag_damage_reward_per_hp = float(stag_damage_reward_per_hp)
        self.stag_reward = float(stag_reward)
        self.damage_taken_penalty_per_hp = float(damage_taken_penalty_per_hp)
        self.death_penalty = float(death_penalty)
        self.timeout_survival_bonus = float(timeout_survival_bonus)

        self.best_stag_kills = None
        self.best_own_rabbit_kills = None
        self.stag_alive = False
        self.lowest_stag_health = None
        self.orig_env_reward = 0.0
        self.prev_health = None

    def _player_id(self) -> int:
        return int(max(0, getattr(self.env.unwrapped, "player_id", 0)))

    def _own_rabbit_key(self) -> str:
        return f"USER{56 + self._player_id()}"

    @staticmethod
    def _float(info, key: str, default: float = 0.0) -> float:
        return float(info.get(key, default))

    def _sync(self, info):
        if info is None:
            self.best_stag_kills = None
            self.best_own_rabbit_kills = None
            self.stag_alive = False
            self.lowest_stag_health = None
            self.prev_health = None
            return

        self.best_stag_kills = self._float(info, "USER54")
        self.best_own_rabbit_kills = self._float(info, self._own_rabbit_key())
        self.stag_alive = self._float(info, "USER55") > 0.0
        self.lowest_stag_health = self._float(info, "USER51") if self.stag_alive else None
        self.prev_health = self._float(info, "HEALTH")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
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
            return obs, 0.0, terminated, truncated, info

        curr_stag_health = self._float(info, "USER51")
        curr_stag_kills = self._float(info, "USER54")
        curr_stag_alive = self._float(info, "USER55") > 0.0
        curr_own_rabbit_kills = self._float(info, self._own_rabbit_key())
        curr_health = self._float(info, "HEALTH")

        if self.best_stag_kills is None or self.best_own_rabbit_kills is None:
            self._sync(info)
            info["true_objective"] = curr_stag_kills
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0

        prev_health = curr_health if self.prev_health is None else self.prev_health
        damage_taken = max(0.0, prev_health - curr_health)
        if damage_taken > 0.0:
            shaped_reward += damage_taken * self.damage_taken_penalty_per_hp
        if curr_health <= 0.0 < prev_health:
            shaped_reward += self.death_penalty

        delta_own_rabbit_kills = curr_own_rabbit_kills - self.best_own_rabbit_kills
        if delta_own_rabbit_kills > 0.0:
            shaped_reward += delta_own_rabbit_kills * self.rabbit_reward

        delta_stag_kills = curr_stag_kills - self.best_stag_kills
        if delta_stag_kills > 0.0:
            shaped_reward += delta_stag_kills * self.stag_reward

        if curr_stag_alive:
            if not self.stag_alive or self.lowest_stag_health is None:
                self.lowest_stag_health = curr_stag_health
            elif curr_stag_health < self.lowest_stag_health:
                unique_damage = self.lowest_stag_health - curr_stag_health
                shaped_reward += unique_damage * self.stag_damage_reward_per_hp
                self.lowest_stag_health = curr_stag_health
        else:
            self.lowest_stag_health = None

        if curr_stag_kills < self.best_stag_kills or curr_own_rabbit_kills < self.best_own_rabbit_kills:
            extra_stats = info.setdefault("episode_extra_stats", {})
            extra_stats["counter_regression"] = 1

        self.best_stag_kills = max(self.best_stag_kills, curr_stag_kills)
        self.best_own_rabbit_kills = max(self.best_own_rabbit_kills, curr_own_rabbit_kills)
        self.stag_alive = curr_stag_alive
        self.prev_health = curr_health

        if truncated and not terminated and curr_health > 0.0 and self.timeout_survival_bonus != 0.0:
            shaped_reward += self.timeout_survival_bonus

        info["true_objective"] = self.best_stag_kills
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        return obs, shaped_reward, terminated, truncated, info
