import gymnasium as gym
import numpy as np


class AmmoCarrierAdditionalInput(gym.Wrapper):
    def __init__(self, env, reserve_target=60.0, home_scale=2048.0):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.reserve_target = max(float(reserve_target), 1.0)
        self.home_scale = max(float(home_scale), 1.0)

        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([200.0, 200.0, 1.0, 200.0, 1.0, 1.0, 32.0, 32.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": current_obs_space,
                "measurements": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            }
        )
        self.measurements_vec = np.zeros(10, dtype=np.float32)

    def _player_id(self) -> int:
        return int(max(0, getattr(self.env.unwrapped, "player_id", 0)))

    def _shortage_ratio(self, shooter_ammo: float, low_alert: float) -> float:
        shortage = max(0.0, self.reserve_target - shooter_ammo)
        if shortage <= 0.0 and low_alert > 0.0:
            shortage = min(self.reserve_target, 5.0)
        return min(1.0, shortage / self.reserve_target)

    def _hub_beacon(self, pos_x: float, pos_y: float, angle_deg: float) -> tuple[float, float]:
        dx = np.clip(-pos_x / self.home_scale, -1.0, 1.0)
        dy = np.clip(-pos_y / self.home_scale, -1.0, 1.0)

        theta = np.deg2rad(angle_deg)
        forward = dx * np.cos(theta) + dy * np.sin(theta)
        right = -dx * np.sin(theta) + dy * np.cos(theta)
        return float(np.clip(forward, -1.0, 1.0)), float(np.clip(right, -1.0, 1.0))

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}
        if info is None:
            self.measurements_vec.fill(0.0)
            return obs_dict

        shooter_ammo = float(max(0.0, info.get("USER41", info.get("AMMO1", 0.0))))
        low_alert = float(max(0.0, info.get("USER45", 0.0)))
        pos_x = float(info.get("POSITION_X", 0.0))
        pos_y = float(info.get("POSITION_Y", 0.0))
        angle_deg = float(info.get("ANGLE", 0.0))
        goal_forward, goal_right = self._hub_beacon(pos_x, pos_y, angle_deg)

        self.measurements_vec[0] = float(max(0.0, info.get("HEALTH", 0.0)))
        self.measurements_vec[1] = float(max(0.0, info.get("AMMO1", 0.0)))
        self.measurements_vec[2] = float(self._player_id() == 0)
        self.measurements_vec[3] = shooter_ammo
        self.measurements_vec[4] = self._shortage_ratio(shooter_ammo, low_alert)
        self.measurements_vec[5] = float(low_alert > 0.0)
        self.measurements_vec[6] = float(max(0.0, info.get("USER44", 0.0)))
        self.measurements_vec[7] = float(max(0.0, info.get("USER43", 0.0)))
        self.measurements_vec[8] = goal_forward
        self.measurements_vec[9] = goal_right

        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.measurements_vec.fill(0.0)
        return self._parse_info(obs, info), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, rew, terminated, truncated, info
        return self._parse_info(obs, info), rew, terminated, truncated, info


class AmmoCarrierRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        kill_reward=1.0,
        damage_reward=0.001,
        supply_delivery_reward=0.5,
        death_penalty=-1.0,
        timeout_survival_bonus=0.25,
        reserve_target=60.0,
        alert_shortage_floor=5.0,
        pressure_bonus_per_enemy=0.02,
        contextual_pickup_reward=0.02,
        runner_damage_taken_penalty=-0.005,
    ):
        super().__init__(env)
        self.kill_reward = float(kill_reward)
        self.damage_reward = float(damage_reward)
        self.supply_delivery_reward = float(supply_delivery_reward)
        self.death_penalty = float(death_penalty)
        self.timeout_survival_bonus = float(timeout_survival_bonus)
        self.reserve_target = float(reserve_target)
        self.alert_shortage_floor = float(alert_shortage_floor)
        self.pressure_bonus_per_enemy = float(pressure_bonus_per_enemy)
        self.contextual_pickup_reward = float(contextual_pickup_reward)
        self.runner_damage_taken_penalty = float(runner_damage_taken_penalty)

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.episode_steps = 0

    def _player_id(self) -> int:
        return int(max(0, getattr(self.env.unwrapped, "player_id", 0)))

    def _is_defender(self) -> bool:
        return self._player_id() == 0

    @staticmethod
    def _float(info, key: str, default: float = 0.0) -> float:
        return float(info.get(key, default))

    def _ammo_shortage(self, shooter_ammo: float, low_alert: float) -> float:
        shortage = max(0.0, self.reserve_target - shooter_ammo)
        if shortage > 0.0:
            return shortage
        if low_alert > 0.0:
            return self.alert_shortage_floor
        return 0.0

    def _sync(self, info):
        if info is None:
            self.prev_vars = {}
            return

        self.prev_vars = {
            "HEALTH": self._float(info, "HEALTH"),
            "AMMO1": self._float(info, "AMMO1"),
            "KILLCOUNT": self._float(info, "KILLCOUNT"),
            "DAMAGECOUNT": self._float(info, "DAMAGECOUNT"),
            "USER41": self._float(info, "USER41"),
            "USER42": self._float(info, "USER42"),
            "USER43": self._float(info, "USER43"),
            "USER44": self._float(info, "USER44"),
            "USER45": self._float(info, "USER45"),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.episode_steps = 0
        self._sync(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None:
            reward = 0.0
        reward = float(reward)
        self.orig_env_reward += reward
        self.episode_steps += 1

        if info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self._sync(info)
            if terminated or truncated:
                info["true_objective"] = float(self.episode_steps)
                info["orig_env_reward"] = self.orig_env_reward
            return obs, reward, terminated, truncated, info

        shaped_reward = 0.0

        curr_health = self._float(info, "HEALTH")
        prev_health = self.prev_vars.get("HEALTH", 0.0)
        if curr_health <= 0.0 < prev_health:
            shaped_reward += self.death_penalty

        if self._is_defender():
            curr_kills = self._float(info, "KILLCOUNT")
            prev_kills = self.prev_vars.get("KILLCOUNT", 0.0)
            delta_kills = curr_kills - prev_kills
            if delta_kills > 0.0:
                shaped_reward += delta_kills * self.kill_reward

            # DAMAGECOUNT already provides dense combat credit; rewarding HITCOUNT too
            # overweights the same progress signal and biases pellet-heavy contact spam.
            curr_damage = self._float(info, "DAMAGECOUNT")
            prev_damage = self.prev_vars.get("DAMAGECOUNT", 0.0)
            delta_damage = min(max(0.0, curr_damage - prev_damage), 100.0)
            if delta_damage > 0.0:
                shaped_reward += delta_damage * self.damage_reward
        else:
            curr_ammo = self._float(info, "AMMO1")
            prev_ammo = self.prev_vars.get("AMMO1", 0.0)
            prev_shooter_ammo = self.prev_vars.get("USER41", 0.0)
            curr_deliveries = self._float(info, "USER42")
            prev_deliveries = self.prev_vars.get("USER42", 0.0)
            prev_low_alert = self.prev_vars.get("USER45", 0.0)
            prev_enemy_count = self.prev_vars.get("USER44", 0.0)

            delta_ammo = curr_ammo - prev_ammo
            delta_deliveries = curr_deliveries - prev_deliveries
            damage_taken = max(0.0, prev_health - curr_health)
            shortage = self._ammo_shortage(prev_shooter_ammo, prev_low_alert)

            if self.contextual_pickup_reward > 0.0 and delta_ammo > 0.0:
                pickup_units = min(delta_ammo, 50.0)
                shaped_reward += pickup_units * self.contextual_pickup_reward

            if damage_taken > 0.0 and self.runner_damage_taken_penalty != 0.0:
                shaped_reward += damage_taken * self.runner_damage_taken_penalty

            if delta_deliveries > 0.0 and self.supply_delivery_reward > 0.0:
                # USER42 increments exactly when Script 2 hands ammo to the shooter.
                pressure_scale = 1.0 + self.pressure_bonus_per_enemy * max(0.0, prev_enemy_count)
                shortage_scale = 1.0 + min(1.0, shortage / max(self.reserve_target, 1.0))
                shaped_reward += delta_deliveries * self.supply_delivery_reward * pressure_scale * shortage_scale

        if truncated and not terminated and curr_health > 0.0 and self.timeout_survival_bonus != 0.0:
            shaped_reward += self.timeout_survival_bonus

        total_reward = reward + shaped_reward

        if terminated or truncated:
            info["true_objective"] = float(self.episode_steps)
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
