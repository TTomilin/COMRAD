import gymnasium as gym

class ArmorySiegeRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        core_alive_reward=0,
        core_damage_penalty=-0.01,
        death_penalty=-2.0,
        weapon_pickup_reward=0.3,
        first_weapon_reward=1,
        ammo_pickup_reward=0.012,
        core_death_penalty=-7.0,
        kill_reward=3,
        hit_reward=0.2,
        ammo_use_penalty=0,
        no_ammo_penalty=-0.02,
        weapon_keys=["WEAPON1", "WEAPON2"],
        ammo_keys=["AMMO1", "AMMO2"],
        health_pickup_reward=0.01,
        survival_bonus=10.0,
    ):
        super().__init__(env)
        self.core_alive_reward = core_alive_reward
        self.core_damage_penalty = core_damage_penalty
        self.death_penalty = death_penalty
        self.weapon_pickup_reward = weapon_pickup_reward
        self.ammo_pickup_reward = ammo_pickup_reward
        self.kill_reward = kill_reward
        self.hit_reward = hit_reward
        self.ammo_use_penalty = ammo_use_penalty
        self.no_ammo_penalty = no_ammo_penalty
        self.weapon_keys = weapon_keys
        self.ammo_keys = ammo_keys
        self.first_weapon_reward = first_weapon_reward
        self.core_death_penalty = core_death_penalty
        self.health_pickup_reward = health_pickup_reward
        self.survival_bonus = survival_bonus

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.max_core_hp = None

    def _get_core_health(self, info):
        core_raw = info.get("USER1")
        if core_raw is not None and core_raw > 0:
            if self.max_core_hp is None:
                self.max_core_hp = core_raw
            return core_raw
        return self.prev_vars.get("USER1") if self.prev_vars else self.max_core_hp

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        current_core = self._get_core_health(info)
        reward = float(reward)

        if not self.prev_vars:
            self.sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0

        prev_core = self.prev_vars.get("USER1")

        if current_core is not None and current_core > 0:
            shaped_reward += self.core_alive_reward

        if current_core is not None and prev_core is not None:
            diff_core = current_core - prev_core
            if diff_core < 0:
                shaped_reward += self.core_damage_penalty * abs(diff_core)

        if current_core is not None and current_core <= 0:
            terminated = True
            if prev_core is not None and prev_core > 0:
                shaped_reward += self.core_death_penalty


        prev_weapon_count = sum(1 for k in self.weapon_keys if self.prev_vars.get(k, 0) > 0)

        newly_acquired_weapons = []
        for wk in self.weapon_keys:
            if info.get(wk, 0) > 0 and self.prev_vars.get(wk, 0) == 0:
                newly_acquired_weapons.append(wk)

        if newly_acquired_weapons:
            if prev_weapon_count == 0:
                shaped_reward += self.first_weapon_reward
                if len(newly_acquired_weapons) > 1:
                    shaped_reward += self.weapon_pickup_reward * (len(newly_acquired_weapons) - 1)

            else:
                shaped_reward += self.weapon_pickup_reward * len(newly_acquired_weapons)


        for ak in self.ammo_keys:
            diff_ammo = info.get(ak, 0) - self.prev_vars.get(ak, 0)
            if diff_ammo > 0:
                shaped_reward += diff_ammo * self.ammo_pickup_reward
            elif diff_ammo < 0:
                shaped_reward += abs(diff_ammo) * -self.ammo_use_penalty

        current_hp = info.get("HEALTH", 0)
        prev_hp = self.prev_vars.get("HEALTH", 0)
        if current_hp <= 0 and prev_hp > 0:
            shaped_reward += self.death_penalty

        # Health pickup
        if current_hp > prev_hp and prev_hp > 0: # prev_hp > 0 to exclude respawn
            shaped_reward += self.health_pickup_reward * (current_hp - prev_hp)

        # Kills with corridor interception bonus
        diff_kills = info.get("KILLCOUNT", 0) - self.prev_vars.get("KILLCOUNT", 0)
        if diff_kills > 0:
            shaped_reward += self.kill_reward * diff_kills

        diff_hits = info.get("HITCOUNT", 0) - self.prev_vars.get("HITCOUNT", 0)
        if diff_hits > 0:
            shaped_reward += self.hit_reward * diff_hits

        has_weapon = any(info.get(k, 0) > 0 for k in self.weapon_keys)
        total_ammo = sum(info.get(k, 0) for k in self.ammo_keys)
        if has_weapon and total_ammo <= 0:
            shaped_reward += self.no_ammo_penalty

        if truncated and not terminated and current_core is not None and current_core > 0:
            shaped_reward += self.survival_bonus

        reward += shaped_reward
        self.orig_env_reward += reward

        if terminated or truncated:
            info["true_objective"] = self.orig_env_reward

        self.sync_vars(info)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.max_core_hp = None

        if info is not None and "USER1" in info:
            self.sync_vars(info)
            core_init = info.get("USER1")
            if core_init is not None and core_init > 0:
                self.max_core_hp = core_init

        return obs, info

    def sync_vars(self, info):
        core_val = self._get_core_health(info)
        self.prev_vars = {
            "USER1": core_val,
            "KILLCOUNT": info.get("KILLCOUNT", 0),
            "HITCOUNT": info.get("HITCOUNT", 0),
            "HEALTH": info.get("HEALTH", 100),
        }
        for k in self.weapon_keys:
            self.prev_vars[k] = info.get(k, 0)
        for k in self.ammo_keys:
            self.prev_vars[k] = info.get(k, 0)
