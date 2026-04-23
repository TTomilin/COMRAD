import gymnasium as gym

class DumbEnemiesRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        kill_reward=5.0,
        hit_reward=0.1,
        health_gain_reward=0.1, 
        health_loss_penalty=-0.1,
        slow_event_reward=0.2,
        continuous_slow_reward=0.05,
        ammo_use_penalty=-0.02,
        death_penalty=-10.0,
    ):
        super().__init__(env)
        self.kill_reward = kill_reward
        self.hit_reward = hit_reward
        self.health_gain_reward = health_gain_reward
        self.health_loss_penalty = health_loss_penalty
        self.slow_event_reward = slow_event_reward
        self.continuous_slow_reward = continuous_slow_reward
        self.ammo_use_penalty = ammo_use_penalty
        self.death_penalty = death_penalty

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.ticks = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        self.ticks += 1
        reward = float(reward)

        if not self.prev_vars:
            self.sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0

        # Health reward and penalty
        curr_health = info.get("HEALTH", 0.0)
        prev_health = self.prev_vars.get("HEALTH", 0.0)
        
        delta_health = curr_health - prev_health
        if delta_health > 0:
            shaped_reward += self.health_gain_reward * delta_health
        elif delta_health < 0:
            shaped_reward += self.health_loss_penalty * abs(delta_health)
            
        # Death penalty
        if curr_health <= 0 and prev_health > 0:
            shaped_reward += self.death_penalty

        # Kill reward
        curr_kills = info.get("KILLCOUNT", 0)
        prev_kills = self.prev_vars.get("KILLCOUNT", 0)
        delta_kills = curr_kills - prev_kills
        if delta_kills > 0:
            shaped_reward += self.kill_reward * delta_kills

        # Hit reward
        curr_hits = info.get("HITCOUNT", 0)
        prev_hits = self.prev_vars.get("HITCOUNT", 0)
        delta_hits = curr_hits - prev_hits
        if delta_hits > 0:
            shaped_reward += self.hit_reward * delta_hits

        # Slow events reward (when new enemies become slow)
        curr_slow_events = info.get("USER33", 0)
        prev_slow_events = self.prev_vars.get("USER33", 0)
        delta_slow_events = curr_slow_events - prev_slow_events
        
        if delta_slow_events > 0:
            shaped_reward += self.slow_event_reward * delta_slow_events
        
        # Continuous slow reward
        curr_slow_enemies = info.get("USER32", 0)
        if curr_slow_enemies > 0:
            shaped_reward += self.continuous_slow_reward * curr_slow_enemies

        curr_ammo = info.get("AMMO2", 0)
        prev_ammo = self.prev_vars.get("AMMO2", 0)
        delta_ammo = curr_ammo - prev_ammo
        if delta_ammo < 0:
            shaped_reward += self.ammo_use_penalty * abs(delta_ammo)

        final_reward = reward + shaped_reward
        self.orig_env_reward += float(reward)

        if terminated or truncated:
            info["true_objective"] = self.ticks

        self.sync_vars(info)
        return obs, final_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.ticks = 0
        
        if info is not None:
            self.sync_vars(info)

        return obs, info

    def sync_vars(self, info):
        self.prev_vars = {
            "HEALTH": info.get("HEALTH", 100.0),
            "KILLCOUNT": info.get("KILLCOUNT", 0),
            "HITCOUNT": info.get("HITCOUNT", 0),
            "USER31": info.get("USER31", 0), # de_alive_enemies_global
            "USER32": info.get("USER32", 0), # de_slow_enemies_global
            "USER33": info.get("USER33", 0), # de_slow_events_global
            "USER34": info.get("USER34", 0), # de_spawn_events_global
            "AMMO2": info.get("AMMO2", 0),
        }
