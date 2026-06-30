import gymnasium as gym
import numpy as np


class ArmorySiegeAdditionalInput(gym.Wrapper):
    """
    health
    ammo1: pistol
    ammo2: shotgun
    weapon1: has pistol (0/1)
    weapon2: has shotgun (0/1)
    core_hp (0-1)
    selected_weapon: 0=none, 1=pistol, 2=shotugn
    killcount
    attack_ready: 0/1
    """

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([200.0, 200.0, 100.0, 1.0, 1.0, 1.0, 3.0, 500.0, 1.0], dtype=np.float32)
        self.low = low
        self.high = high

        self.observation_space = gym.spaces.Dict({
            "obs": current_obs_space,
            "measurements": gym.spaces.Box(low=low, high=high, dtype=np.float32),
        })

        self.measurements_vec = np.zeros(9, dtype=np.float32)
        self._observed_max_core_hp = None

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}

        if info is None:
            self.measurements_vec.fill(0.0)
            return obs_dict

        health = max(0.0, info.get("HEALTH", 0.0))
        self.measurements_vec[0] = health

        ammo1 = max(0.0, info.get("AMMO1", 0.0))
        self.measurements_vec[1] = ammo1

        ammo2 = max(0.0, info.get("AMMO2", 0.0))
        self.measurements_vec[2] = ammo2

        self.measurements_vec[3] = float(info.get("WEAPON1", 0) > 0)

        self.measurements_vec[4] = float(info.get("WEAPON2", 0) > 0)

        # already 0-1 normalized
        core_hp = info.get("USER1", None)
        if core_hp is not None and core_hp > 0:
            if self._observed_max_core_hp is None:
                self._observed_max_core_hp = core_hp
            self.measurements_vec[5] = core_hp / self._observed_max_core_hp
        elif self._observed_max_core_hp is not None:
            self.measurements_vec[5] = 0.0 # Core destroyed
        else:
            self.measurements_vec[5] = 1.0

        selected = max(0, int(info.get("SELECTED_WEAPON", 0)))
        self.measurements_vec[6] = float(selected)

        killcount = max(0.0, info.get("KILLCOUNT", 0.0))
        self.measurements_vec[7] = killcount # TODO: might scale to smaller range

        self.measurements_vec[8] = float(info.get("ATTACK_READY", 0) > 0)

        np.clip(self.measurements_vec, self.low, self.high, out=self.measurements_vec)

        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._observed_max_core_hp = None
        self.measurements_vec.fill(0.0)

        # Get initial info from unwrapped env if needed
        if info is None:
            info = self.env.unwrapped.get_info()

        obs_dict = self._parse_info(obs, info)
        return obs_dict, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        if obs is None:
            return obs, rew, terminated, truncated, info

        obs_dict = self._parse_info(obs, info)
        return obs_dict, rew, terminated, truncated, info


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
        track_coop=True,
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
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop

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


        if self.track_coop:
            curr_kills = info.get("KILLCOUNT", 0)
            prev_kills = self.prev_vars.get("KILLCOUNT", 0) if self.prev_vars else 0
            delta_kills_now = curr_kills - prev_kills
            curr_hits = info.get("HITCOUNT", 0)
            prev_hits = self.prev_vars.get("HITCOUNT", 0) if self.prev_vars else 0
            delta_hits_now = curr_hits - prev_hits

            curr_core = self._get_core_health(info)
            prev_core_val = self.prev_vars.get("USER1") if self.prev_vars else None
            
            engaged = delta_kills_now > 0 or delta_hits_now > 0
            core_decreasing = (curr_core is not None and prev_core_val is not None and curr_core < prev_core_val)
            enemies_present = core_decreasing
            coop = 1.0 if engaged else 0.0
            defect = 1.0 if not engaged and enemies_present else 0.0

            info["coop_step_signal"] = coop
            info["defect_step_signal"] = defect

            self.episode_coop_steps += coop
            self.episode_defect_steps += defect

            if terminated or truncated:
                self._record_episode_stats(info)

        self.sync_vars(info)

        return obs, reward, terminated, truncated, info

    def _record_episode_stats(self, info):
        extra = info.setdefault("episode_extra_stats", {})
        total = self.episode_coop_steps + self.episode_defect_steps
        extra["coop_steps"] = self.episode_coop_steps
        extra["defect_steps"] = self.episode_defect_steps
        extra["total_coop_defect_steps"] = total
        if total > 0:
            extra["cooperation_index"] = self.episode_coop_steps / total
            extra["defector_index"] = self.episode_defect_steps / total

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.max_core_hp = None
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0

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
