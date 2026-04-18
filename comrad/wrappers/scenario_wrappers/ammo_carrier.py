import gymnasium as gym
import math


class AmmoCarrierRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        # Shooter rewards (WEAPON=4)
        kill_reward=1.0,              # Kill an enemy
        death_penalty=-1.0,           # Penalty for dying
        health_loss_penalty=0.01,     # Penalty per health point lost
        survival_bonus=0.005,         # Small reward per step alive
        # Carrier rewards (WEAPON=-1)
        exploration_bonus=0.002,      # Reward for moving (exploration)
    ):
        super().__init__(env)
        self.kill_reward = kill_reward
        self.death_penalty = death_penalty
        self.health_loss_penalty = health_loss_penalty
        self.survival_bonus = survival_bonus
        self.exploration_bonus = exploration_bonus

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.episode_shaped_return = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info is None:
            return obs, 0.0, terminated, truncated, {}
        if reward is None:
            reward = 0.0

        reward = float(reward)
        self.orig_env_reward += reward

        if not self.prev_vars:
            self.sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0

        curr_health = info.get("HEALTH", 0)
        curr_kills  = info.get("KILLCOUNT", 0)
        curr_weapon = info.get("SELECTED_WEAPON", 0)
        curr_x      = info.get("POSITION_X", 0.0)
        curr_y      = info.get("POSITION_Y", 0.0)

        prev_health = self.prev_vars.get("HEALTH", 0)
        prev_kills  = self.prev_vars.get("KILLCOUNT", 0)
        prev_x      = self.prev_vars.get("POSITION_X", 0.0)
        prev_y      = self.prev_vars.get("POSITION_Y", 0.0)

        # ── 1. Death penalty (both agents) ────────
        if curr_health <= 0:
            terminated = True
            if prev_health > 0:
                shaped_reward += self.death_penalty
            return self._finalize(obs, reward, shaped_reward, terminated, truncated, info)

        # ── 2. Survival bonus (both agents) ───────
        shaped_reward += self.survival_bonus

        # ── 3. Health loss penalty (both agents) ──
        health_delta = curr_health - prev_health
        if health_delta < 0:
            shaped_reward += health_delta * self.health_loss_penalty

        # ── 4. Kill reward (Shooter only) ─────────
        # Carrier has WEAPON=-1 so will never get kills
        delta_kills = curr_kills - prev_kills
        if delta_kills > 0:
            shaped_reward += delta_kills * self.kill_reward

        # ── 5. Exploration bonus (Carrier only) ───
        # Encourage Carrier to keep moving and explore
        # Shooter is stationary so this naturally only affects Carrier
        displacement = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        if curr_weapon == -1 and displacement > 2.0:
            shaped_reward += self.exploration_bonus

        return self._finalize(obs, reward, shaped_reward, terminated, truncated, info)

    def _finalize(self, obs, env_reward, shaped_reward, terminated, truncated, info):
        self.prev_vars = info.copy()
        total_reward = env_reward + shaped_reward
        self.episode_shaped_return += total_reward

        if terminated or truncated:
            info["true_objective"] = self.episode_shaped_return
            info["orig_env_reward"] = self.orig_env_reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self.episode_shaped_return = 0.0
        self.sync_vars(info)
        return obs, info

    def sync_vars(self, info):
        self.prev_vars = info.copy()
# class AmmoCarrierRewardShaping(gym.Wrapper):
#     def __init__(
#         self,
#         env,
#         ammo_pickup_reward=1.0,       # Reward for picking up ammo (Carrier only)
#         ammo_give_reward=3.0,         # Reward for giving ammo to Shooter
#         idle_penalty=-0.02,           # Penalty for staying idle (Carrier only)
#         idle_steps_threshold=60,      # Steps before idle penalty triggers
#         idle_distance_threshold=2.0,  # Displacement below this is considered idle
#         kill_reward=3.0,              # Reward for killing an enemy (Shooter only)
#         no_ammo_penalty=-0.05,        # Penalty for Shooter having no ammo
#         death_penalty=-2.0,           # Penalty for dying (both agents)
#         health_loss_penalty=0.02,     # Penalty coefficient per health point lost
#         survival_bonus=0.01,          # Small reward per step for staying alive
#     ):
#         super().__init__(env)
#         self.ammo_pickup_reward = ammo_pickup_reward
#         self.ammo_give_reward = ammo_give_reward
#         self.idle_penalty = idle_penalty
#         self.idle_steps_threshold = idle_steps_threshold
#         self.idle_distance_threshold = idle_distance_threshold
#         self.kill_reward = kill_reward
#         self.no_ammo_penalty = no_ammo_penalty
#         self.death_penalty = death_penalty
#         self.health_loss_penalty = health_loss_penalty
#         self.survival_bonus = survival_bonus

#         self.prev_vars = {}
#         self.orig_env_reward = 0.0
#         self.episode_shaped_return = 0.0
#         self._idle_steps = 0

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)

#         if info is None:
#             return obs, 0.0, terminated, truncated, {}
#         if reward is None:
#             reward = 0.0

#         reward = float(reward)
#         self.orig_env_reward += reward

#         if not self.prev_vars:
#             self.sync_vars(info)
#             return obs, 0.0, terminated, truncated, info

#         shaped_reward = 0.0

#         curr_health = info.get("HEALTH", 0)
#         curr_ammo   = info.get("AMMO1", 0)
#         curr_weapon = info.get("SELECTED_WEAPON", 0)
#         curr_kills  = info.get("KILLCOUNT", 0)
#         curr_x      = info.get("POSITION_X", 0.0)
#         curr_y      = info.get("POSITION_Y", 0.0)

#         prev_health = self.prev_vars.get("HEALTH", 0)
#         prev_ammo   = self.prev_vars.get("AMMO1", 0)
#         prev_kills  = self.prev_vars.get("KILLCOUNT", 0)
#         prev_x      = self.prev_vars.get("POSITION_X", 0.0)
#         prev_y      = self.prev_vars.get("POSITION_Y", 0.0)

#         # ── 1. Death penalty ──────────────────────
#         if curr_health <= 0:
#             terminated = True
#             if prev_health > 0:
#                 shaped_reward += self.death_penalty
#             return self._finalize(obs, reward, shaped_reward, terminated, truncated, info)

#         # ── 2. Survival bonus ─────────────────────
#         shaped_reward += self.survival_bonus

#         # ── 3. Health loss penalty ────────────────
#         health_delta = curr_health - prev_health
#         if health_delta < 0:
#             shaped_reward += health_delta * self.health_loss_penalty

#         # ── 4. Ammo pickup / give (Carrier) ───────
#         delta_ammo = curr_ammo - prev_ammo
#         if delta_ammo > 0:
#             # Ammo increased: Carrier picked up ammo
#             shaped_reward += delta_ammo * self.ammo_pickup_reward
#         elif delta_ammo < 0:
#             # Ammo decreased: if not Shooter firing (WEAPON=4), treat as Carrier giving ammo
#             if curr_weapon != 4:
#                 shaped_reward += abs(delta_ammo) * self.ammo_give_reward

#         # ── 5. Kill reward (Shooter) ──────────────
#         delta_kills = curr_kills - prev_kills
#         if delta_kills > 0:
#             shaped_reward += delta_kills * self.kill_reward

#         # ── 6. No-ammo penalty (Shooter) ─────────
#         # Carrier has WEAPON=-1, so this condition never triggers for Carrier
#         if curr_weapon == 4 and curr_ammo <= 0:
#             shaped_reward += self.no_ammo_penalty

#         # ── 7. Idle penalty (Carrier) ─────────────
#         # Only penalise when carrying ammo to avoid penalising Shooter's natural stillness
#         displacement = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
#         if displacement < self.idle_distance_threshold:
#             self._idle_steps += 1
#         else:
#             self._idle_steps = 0

#         if self._idle_steps > self.idle_steps_threshold and curr_ammo > 0:
#             shaped_reward += self.idle_penalty

#         return self._finalize(obs, reward, shaped_reward, terminated, truncated, info)

#     def _finalize(self, obs, env_reward, shaped_reward, terminated, truncated, info):
#         self.prev_vars = info.copy()
#         total_reward = env_reward + shaped_reward
#         self.episode_shaped_return += total_reward

#         if terminated or truncated:
#             info["true_objective"] = self.episode_shaped_return
#             info["orig_env_reward"] = self.orig_env_reward

#         return obs, total_reward, terminated, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.prev_vars = {}
#         self.orig_env_reward = 0.0
#         self.episode_shaped_return = 0.0
#         self._idle_steps = 0
#         self.sync_vars(info)
#         return obs, info

#     def sync_vars(self, info):
#         self.prev_vars = info.copy()

# Old version
# import gymnasium as gym

# class AmmoCarrierRewardShaping(gym.Wrapper):
#     def __init__(
#         self,
#         env,
#         ammo_pickup_reward=0.5,
#         ammo_give_reward=2.0,
#         kill_reward=1.0,
#         death_penalty=-0.5,
#     ):
#         super().__init__(env)
#         self.ammo_pickup_reward = ammo_pickup_reward
#         self.ammo_give_reward = ammo_give_reward
#         self.kill_reward = kill_reward
#         self.death_penalty = death_penalty

#         self.prev_vars = {}
#         self.orig_env_reward = 0.0
#         self.episode_shaped_return = 0.0

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)

#         if reward is None or info is None:
#             return obs, reward, terminated, truncated, info

#         reward = float(reward)
#         self.orig_env_reward += reward

#         if not self.prev_vars:
#             self.sync_vars(info)
#             return obs, 0.0, terminated, truncated, info

#         shaped_reward = 0.0

#         curr_health = info.get("HEALTH", 0)
#         curr_ammo = info.get("AMMO1", 0)
#         curr_kills = info.get("KILLCOUNT", 0)
#         curr_weapon = info.get("SELECTED_WEAPON", 0)

#         prev_health = self.prev_vars.get("HEALTH", 0)
#         prev_ammo = self.prev_vars.get("AMMO1", 0)
#         prev_kills = self.prev_vars.get("KILLCOUNT", 0)

#         if curr_health <= 0:
#             terminated = True
#             if prev_health > 0:
#                 shaped_reward += self.death_penalty
#         else:
#             delta_ammo = curr_ammo - prev_ammo

#             if delta_ammo > 0:
#                 shaped_reward += delta_ammo * self.ammo_pickup_reward
#             elif delta_ammo < 0:
#                 is_armed_with_ammo1 = (curr_weapon == 4)

#                 if not is_armed_with_ammo1:
#                      shaped_reward += abs(delta_ammo) * self.ammo_give_reward

#         delta_kills = curr_kills - prev_kills
#         if delta_kills > 0:
#            shaped_reward += delta_kills * self.kill_reward

#         self.prev_vars = info.copy()

#         total_reward = reward + shaped_reward
#         self.episode_shaped_return += total_reward

#         done = terminated | truncated
#         if done:
#             info["true_objective"] = self.episode_shaped_return

#         return obs, total_reward, terminated, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.prev_vars = {}
#         self.orig_env_reward = 0.0
#         self.episode_shaped_return = 0.0
#         self.sync_vars(info)
#         return obs, info

#     def sync_vars(self, info):
#         self.prev_vars = info.copy()
