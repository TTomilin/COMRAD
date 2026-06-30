import gymnasium as gym

class SmartEnemiesRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        kill_reward=5.0,
        hit_reward=0.3,
        health_gain_reward=0.1,
        health_loss_penalty=-0.1,
        fast_event_penalty=-0.15,
        continuous_fast_penalty=-0.05,
        ammo_use_penalty=-0.02,
        death_penalty=-10.0,
        track_coop=True,
    ):
        super().__init__(env)
        self.kill_reward = kill_reward
        self.hit_reward = hit_reward
        self.health_gain_reward = health_gain_reward
        self.health_loss_penalty = health_loss_penalty
        self.fast_event_penalty = fast_event_penalty
        self.continuous_fast_penalty = continuous_fast_penalty
        self.ammo_use_penalty = ammo_use_penalty
        self.death_penalty = death_penalty

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.ticks = 0
        self.episode_start_kills = None
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop

    def _true_objective(self, info) -> float:
        curr_kills = info.get("KILLCOUNT", 0)
        start_kills = 0 if self.episode_start_kills is None else int(self.episode_start_kills)
        return float(max(0, curr_kills - start_kills))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        if self.episode_start_kills is None:
            self.episode_start_kills = info.get("KILLCOUNT", 0)

        self.ticks += 1
        reward = float(reward)

        if not self.prev_vars:
            self.sync_vars(info)
            if terminated or truncated:
                info["true_objective"] = self._true_objective(info)
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

        # Fast events penalty (when new enemies become fast)
        curr_fast_events = info.get("USER33", 0)
        prev_fast_events = self.prev_vars.get("USER33", 0)
        delta_fast_events = curr_fast_events - prev_fast_events
        if delta_fast_events > 0:
            shaped_reward += self.fast_event_penalty * delta_fast_events

        # Continuous fast penalty
        curr_fast_enemies = info.get("USER32", 0)
        if curr_fast_enemies > 0:
            shaped_reward += self.continuous_fast_penalty * curr_fast_enemies

        curr_ammo = info.get("AMMO2", 0)
        prev_ammo = self.prev_vars.get("AMMO2", 0)
        delta_ammo = curr_ammo - prev_ammo
        if delta_ammo < 0:
            shaped_reward += self.ammo_use_penalty * abs(delta_ammo)

        final_reward = reward + shaped_reward
        self.orig_env_reward += float(reward)

        if terminated or truncated:
            info["true_objective"] = self._true_objective(info)

        curr_kills = info.get("KILLCOUNT", 0)
        prev_kills_val = self.prev_vars.get("KILLCOUNT", 0) if self.prev_vars else 0
        delta_kills_now = curr_kills - prev_kills_val
        curr_fast = info.get("USER32", 0)
        prev_fast = self.prev_vars.get("USER32", 0) if self.prev_vars else 0
        delta_fast = curr_fast - prev_fast

        if self.track_coop:
            coop = 1.0 if delta_kills_now > 0 and curr_fast == 0 else 0.0
            defect = 1.0 if delta_fast > 0 else 0.0

            info["coop_step_signal"] = coop
            info["defect_step_signal"] = defect

            self.episode_coop_steps += coop
            self.episode_defect_steps += defect

            if terminated or truncated:
                self._record_episode_stats(info)

        self.sync_vars(info)

        return obs, final_reward, terminated, truncated, info

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
        self.ticks = 0
        self.episode_start_kills = None
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0

        if info is not None:
            self.episode_start_kills = info.get("KILLCOUNT", 0)
            self.sync_vars(info)

        return obs, info

    def sync_vars(self, info):
        self.prev_vars = {
            "HEALTH": info.get("HEALTH", 100.0),
            "KILLCOUNT": info.get("KILLCOUNT", 0),
            "HITCOUNT": info.get("HITCOUNT", 0),
            "USER32": info.get("USER32", 0),
            "USER33": info.get("USER33", 0),
            "AMMO2": info.get("AMMO2", 0),
        }
