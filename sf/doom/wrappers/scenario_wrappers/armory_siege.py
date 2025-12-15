import gymnasium as gym
import math

class ArmorySiegeRewardShaping(gym.Wrapper):
    def __init__(
        self, 
        env, 
        core_alive_reward=0.015,
        
        core_damage_penalty=-0.05,
    
        death_penalty=-1.0,         
        
        weapon_pickup_reward=1.0,    
        ammo_pickup_reward=0.02,     
        
        kill_reward=0.1,            
        
        weapon_keys=["WEAPON2", "WEAPON3", "WEAPON5"],
        ammo_keys=["AMMO2", "AMMO3", "AMMO5"]
    ):
        super().__init__(env)
        self.core_alive_reward = core_alive_reward
        self.core_damage_penalty = core_damage_penalty
        self.death_penalty = death_penalty
        self.weapon_pickup_reward = weapon_pickup_reward
        self.ammo_pickup_reward = ammo_pickup_reward
        self.kill_reward = kill_reward
        self.weapon_keys = weapon_keys
        self.ammo_keys = ammo_keys
        
        self.prev_vars = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info
        
        if not self.prev_vars:
            self.sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0
        
        current_core = info.get("USER1", 0)
        if current_core > 0:
            shaped_reward += self.core_alive_reward

        prev_core = self.prev_vars.get("USER1", 800)
        diff_core = current_core - prev_core
        if diff_core < 0:
            shaped_reward += self.core_damage_penalty * abs(diff_core)

        for wk in self.weapon_keys:
            if info.get(wk, 0) > 0 and self.prev_vars.get(wk, 0) == 0:
                shaped_reward += self.weapon_pickup_reward
        
        for ak in self.ammo_keys:
            diff_ammo = info.get(ak, 0) - self.prev_vars.get(ak, 0)
            if diff_ammo > 0:
                shaped_reward += diff_ammo * self.ammo_pickup_reward

        current_hp = info.get("HEALTH", 0)
        prev_hp = self.prev_vars.get("HEALTH", 0)
        if current_hp <= 0 and prev_hp > 0:
            shaped_reward += self.death_penalty

        diff_kills = info.get("KILLCOUNT", 0) - self.prev_vars.get("KILLCOUNT", 0)
        if diff_kills > 0:
            shaped_reward += self.kill_reward * diff_kills

        self.sync_vars(info)
        return obs, reward + shaped_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.sync_vars(info)
        return obs, info

    def sync_vars(self, info):
        self.prev_vars = {
            "USER1": info.get("USER1", 800),
            "KILLCOUNT": info.get("KILLCOUNT", 0),
            "HEALTH": info.get("HEALTH", 100),
        }
        for k in self.weapon_keys:
            self.prev_vars[k] = info.get(k, 0)
        for k in self.ammo_keys:
            self.prev_vars[k] = info.get(k, 0)