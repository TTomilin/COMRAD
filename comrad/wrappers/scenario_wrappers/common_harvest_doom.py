"""
Reward shaping for Common Harvest Doom.

Social dilemma structure
------------------------
Two agents share an ammo pool. Ammo only spawns inside one of four side rooms
when a player is physically present there — so at least one agent must
periodically leave the arena to restock, sacrificing combat time.

Reward signals
--------------
kill_reward          : +N per kill (per-player KILLCOUNT delta)
ammo_pickup_reward   : +N per ammo unit picked up (AMMO1 delta > 0); set high
                       enough (~0.05) that a full ammo-room trip (~10 clips = +0.5)
                       competes with the opportunity cost of 1 missed kill (+1.0)
death_penalty        : -N on dying (HEALTH drops to 0)
low_shared_ammo_pen  : -N each step when the shared pool (USER53) < low_ammo_threshold
enemy_count_bonus    : +N each step proportional to live enemy count (encourages
                       fighting when enemies are present rather than hiding)

Global observation variables exposed through cfg (available to the agent):
  USER52  ch_kills_global       - total kills by both agents this episode
  USER53  ch_shared_ammo_global - total ammo held by P1+P2 (equalized every 4 tics)
  USER54  ch_world_ammo_global  - ammo items currently on the ground
  USER55  ch_enemy_count_global - live enemy count
"""

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
