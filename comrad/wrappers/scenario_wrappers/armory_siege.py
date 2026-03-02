"""
Agent learns to pick up ammo, shoot, enemies in first few waves, but after the worm enemy appears and much more enemies spawn it stops shooting.
Also one agent tends to stares at the core. Eventually it runs to the room to pick up weapon and shoot 1/2 enemies, but then it idles staring at the core again. Then the other agent also stops shooting.

Old:
core_alive_reward=0,
core_damage_penalty=-0.01,
death_penalty=-1,
weapon_pickup_reward=0.3,
first_weapon_reward=1,
ammo_pickup_reward=0.012,
core_death_penalty=-7.0,
kill_reward=3,
hit_reward=0.1,
ammo_use_penalty=0,
no_ammo_penalty=0,
weapon_keys=["WEAPON1", "WEAPON2"],
ammo_keys=["AMMO1", "AMMO2"]
"""

import gymnasium as gym
import math

class ArmorySiegeRewardShaping(gym.Wrapper):
    shared = {} # keyed by (worker_index, vector_index)

    def __init__(
        self,
        env,
        core_alive_reward=0,
        core_damage_penalty=-0.03,
        death_penalty=-1,
        weapon_pickup_reward=0.3,
        first_weapon_reward=1,
        ammo_pickup_reward=0.012,
        core_death_penalty=-10.0,
        kill_reward=3,
        hit_reward=0.1,
        ammo_use_penalty=0.001,
        no_ammo_penalty=0,
        weapon_keys=["WEAPON1", "WEAPON2"],
        ammo_keys=["AMMO1", "AMMO2"],
        health_pickup_reward=0.01,
        common_reward=0.0,
        core_proximity_reward=0.02,
        away_penalty=-0.01,
        defend_radius=350,
        interception_bonus=0.5,
        survival_bonus=5.0,
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
        self.common_reward = common_reward
        self.core_proximity_reward = core_proximity_reward
        self.away_penalty = away_penalty
        self.defend_radius = defend_radius
        self.interception_bonus = interception_bonus
        self.survival_bonus = survival_bonus

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.steps_away = 0
        self.max_core_hp = None

        self._ek = None
        self.pid = getattr(env.unwrapped, 'player_id', None)

    @property
    def ek(self):
        if self._ek is None:
            worker_index = getattr(self.env.unwrapped, 'worker_index', 0)
            vector_index = getattr(self.env.unwrapped, 'vector_index', 0)
            self._ek = (worker_index, vector_index)
        return self._ek

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self.sync_vars(info)
            # Track max core HP from first observation
            current_core_init = info.get("USER1", 0)
            if self.max_core_hp is None and current_core_init > 0:
                self.max_core_hp = current_core_init
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0

        current_core = info.get("USER1", 0)
        if current_core > 0:
            shaped_reward += self.core_alive_reward

        prev_core = self.prev_vars.get("USER1", 1000)
        diff_core = current_core - prev_core
        if diff_core < 0:
            shaped_reward += self.core_damage_penalty * abs(diff_core)
            # if diff_core < -100:
            #     print(f"HUGE DROP: {prev_core} -> {current_core}")

        if current_core <= 0:
            terminated = True
            if prev_core > 0:
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
            base_kill = self.kill_reward * diff_kills
            # Interception bonus: kills further from core get up to interception_bonus extra
            if current_hp > 0 and self.defend_radius > 0 and self.interception_bonus > 0:
                px = info.get("POSITION_X", 0)
                py = info.get("POSITION_Y", 0)
                cx = info.get("USER2", 0)
                cy = info.get("USER3", 0)
                kill_dist = math.sqrt((px - cx)**2 + (py - cy)**2)
                bonus_frac = min(1.0, kill_dist / (self.defend_radius * 2))
                shaped_reward += base_kill + self.interception_bonus * bonus_frac * diff_kills
            else:
                shaped_reward += base_kill

        diff_hits = info.get("HITCOUNT", 0) - self.prev_vars.get("HITCOUNT", 0)
        if diff_hits > 0:
            shaped_reward += self.hit_reward * diff_hits

        has_weapon = any(info.get(k, 0) > 0 for k in self.weapon_keys)
        total_ammo = sum(info.get(k, 0) for k in self.ammo_keys)
        if has_weapon and total_ammo <= 0:
            shaped_reward += self.no_ammo_penalty

        # Spatial rewards: gradient based on distance from actual core position (USER2/USER3)
        # Use manhattan distance
        # If player runs too far away from the core they get punished
        if current_hp > 0 and self.defend_radius > 0:
            px = info.get("POSITION_X", 0)
            py = info.get("POSITION_Y", 0)
            cx = info.get("USER2", 0)
            cy = info.get("USER3", 0)
            dx = px - cx
            dy = py - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= self.defend_radius:
                shaped_reward += self.core_proximity_reward * (1.0 - dist / self.defend_radius)
                self.steps_away = 0
            else:
                self.steps_away += 1
                escalation = 1.0 + min(self.steps_away / 50.0, 3.0)
                shaped_reward += self.away_penalty * min(1.0, (dist - self.defend_radius) / self.defend_radius) * escalation

        # Terminal bonus: reward for keeping the core alive
        if terminated or truncated:
            if current_core > 0 and self.survival_bonus > 0:
                max_hp = self.max_core_hp if self.max_core_hp else 1000
                shaped_reward += (current_core / max_hp) * self.survival_bonus

        individual_reward = reward + shaped_reward
        self.orig_env_reward += reward


        # https://github.com/uoe-agents/epymarl/blob/cbc38c09588064eab978501d0f12c2cf58fa7fc2/src/envs/gymma.py#L63
        if self.common_reward > 0:
            self._post_reward(individual_reward)
            team_rewards = self._get_team_rewards()
            if team_rewards is not None:
                # How to aggregate rewards to single common reward
                # epymarl's default is sum so ig I will also use sum here
                # From what I understand, sum is the standard, mean is only used when scaling up to 100 agents or sth but still need to be stable
                # TODO: Decide and pass this to wrapper class, and probably cfg.py
                summ = True
                if not summ:
                    reward_agg = sum(team_rewards) / len(team_rewards)
                else:
                    reward_agg = sum(team_rewards)
                final_reward = (1.0 - self.common_reward) * individual_reward + self.common_reward * reward_agg
            else:
                final_reward = individual_reward
        else:
            final_reward = individual_reward

        if terminated or truncated:
            info["true_objective"] = self.orig_env_reward

        self.sync_vars(info)
        return obs, final_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.sync_vars(info)
        self.orig_env_reward = 0.0
        self.steps_away = 0
        self.max_core_hp = None
        self._clear_shared()
        return obs, info

    def _post_reward(self, reward):
        ArmorySiegeRewardShaping.shared.setdefault(self.ek, {})[self.pid] = reward

    def _get_team_rewards(self):
        if self.ek not in ArmorySiegeRewardShaping.shared:
            return None
        rewards = ArmorySiegeRewardShaping.shared[self.ek]
        if len(rewards) <= 1:
            return None
        return list(rewards.values())

    def _clear_shared(self):
        if self.ek in ArmorySiegeRewardShaping.shared:
            shared = ArmorySiegeRewardShaping.shared[self.ek]
            shared.pop(self.pid, None)
            if not shared:
                ArmorySiegeRewardShaping.shared.pop(self.ek, None)

    def sync_vars(self, info):
        self.prev_vars = {
            "USER1": info.get("USER1", 1000),
            "KILLCOUNT": info.get("KILLCOUNT", 0),
            "HITCOUNT": info.get("HITCOUNT", 0),
            "HEALTH": info.get("HEALTH", 100),
        }
        for k in self.weapon_keys:
            self.prev_vars[k] = info.get(k, 0)
        for k in self.ammo_keys:
            self.prev_vars[k] = info.get(k, 0)
