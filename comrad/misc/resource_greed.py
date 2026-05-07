import gymnasium as gym


class CommonHarvestRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        kill_reward=1.0,
        ammo_pickup_reward=0.05,
        death_penalty=-1.0,
        low_ammo_threshold=20,
        low_shared_ammo_pen=-0.02,
        enemy_count_bonus=0.002,
    ):
        super().__init__(env)
        self.kill_reward = kill_reward
        self.ammo_pickup_reward = ammo_pickup_reward
        self.death_penalty = death_penalty
        self.low_ammo_threshold = low_ammo_threshold
        self.low_shared_ammo_pen = low_shared_ammo_pen
        self.enemy_count_bonus = enemy_count_bonus

        self.prev_vars = {}
        self.orig_env_reward = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self._sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped = 0.0

        curr_health = info.get("HEALTH", 0)
        curr_ammo   = info.get("AMMO1", 0)
        curr_kills  = info.get("KILLCOUNT", 0)
        shared_pool = info.get("USER53", 0)
        enemy_count = info.get("USER55", 0)

        prev_health = self.prev_vars.get("HEALTH", 100)
        prev_ammo   = self.prev_vars.get("AMMO1", 0)
        prev_kills  = self.prev_vars.get("KILLCOUNT", 0)

        # Death penalty
        if curr_health <= 0 and prev_health > 0:
            shaped += self.death_penalty

        # Kill reward
        delta_kills = curr_kills - prev_kills
        if delta_kills > 0:
            shaped += delta_kills * self.kill_reward

        # Ammo delta
        delta_ammo = curr_ammo - prev_ammo
        if delta_ammo > 0:
            # Picked up ammo (was in ammo room)
            shaped += delta_ammo * self.ammo_pickup_reward

        # Penalty when shared pool is dangerously low
        if shared_pool < self.low_ammo_threshold:
            shaped += self.low_shared_ammo_pen

        # Small per-step bonus proportional to enemy pressure (keeps agents engaged)
        if curr_health > 0 and enemy_count > 0:
            shaped += self.enemy_count_bonus * enemy_count

        self.orig_env_reward += reward

        if terminated or truncated:
            info["true_objective"] = self.orig_env_reward

        self._sync_vars(info)
        return obs, reward + shaped, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sync_vars(info)
        self.orig_env_reward = 0.0
        return obs, info

    def _sync_vars(self, info):
        self.prev_vars = {
            "HEALTH":    info.get("HEALTH", 100),
            "AMMO1":     info.get("AMMO1", 0),
            "KILLCOUNT": info.get("KILLCOUNT", 0),
        }
