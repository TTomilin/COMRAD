import gymnasium as gym

class AmmoCarrierRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        ammo_pickup_reward=0.02,
        ammo_give_reward=0.05,
        kill_reward=1.0,
        death_penalty=-1.0,
    ):
        super().__init__(env)
        self.ammo_pickup_reward = ammo_pickup_reward
        self.ammo_give_reward = ammo_give_reward
        self.kill_reward = kill_reward
        self.death_penalty = death_penalty

        self.prev_vars = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self.sync_vars(info)
            return obs, 0.0, terminated, truncated, info

        shaped_reward = 0.0

        curr_health = info.get("HEALTH", 0)
        curr_ammo = info.get("AMMO1", 0)
        curr_kills = info.get("KILLCOUNT", 0)
        curr_weapon = info.get("SELECTED_WEAPON", 0)

        prev_health = self.prev_vars.get("HEALTH", 0)
        prev_ammo = self.prev_vars.get("AMMO1", 0)
        prev_kills = self.prev_vars.get("KILLCOUNT", 0)

        if curr_health <= 0:
            terminated = True
            if prev_health > 0:
                shaped_reward += self.death_penalty
        else:
            delta_ammo = curr_ammo - prev_ammo

            if delta_ammo > 0:
                shaped_reward += delta_ammo * self.ammo_pickup_reward
            elif delta_ammo < 0:
                is_armed_with_ammo1 = (curr_weapon == 4)

                if not is_armed_with_ammo1:
                     shaped_reward += abs(delta_ammo) * self.ammo_give_reward

        delta_kills = curr_kills - prev_kills
        if delta_kills > 0:
           shaped_reward += delta_kills * self.kill_reward

        self.prev_vars = info.copy()

        return obs, reward + shaped_reward, terminated, truncated, info

    def sync_vars(self, info):
        self.prev_vars = info.copy()
