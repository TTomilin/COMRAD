import gymnasium as gym

class LavapitRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        scaler: float = 0.01,
        death_penalty: float = -1.0,
        bridge_positions: list[float] = [256.0, 1024.0, 1792.0],
        bridge_reward: float = 5.0,
        pos_key: str = "POSITION_X",
        dead_key: str = "DEAD",
        x_start: float = 32.0,
    ):
        super().__init__(env)
        self.scaler = float(scaler)
        self.death_penalty = float(death_penalty)
        self.bridge_positions = sorted([float(b) for b in bridge_positions])
        self.bridge_reward = float(bridge_reward)
        self.pos_key = pos_key
        self.dead_key = dead_key
        self.x_start = float(x_start)

        self._prev_x: float | None = None
        self._prev_dead: bool = True
        self._bridges_crossed: set[float] = set()
        self._episode_shaped_return: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x = info.get(self.pos_key, None)
        self._prev_dead = bool(info.get(self.dead_key, 0))
        self._bridges_crossed = set()
        self._episode_shaped_return = 0.0
        return obs, info

    def step(self, action):
        obs, based_rewards, terminations, truncations, infos = self.env.step(action)

        if based_rewards is None:
            return obs, based_rewards, terminations, truncations, infos

        r = float(based_rewards)
        x = infos.get(self.pos_key, None)
        xp = self._prev_x

        # Forward movement reward
        if xp is not None and x is not None:
            forward_progress = max(0.0, x - xp)
            if forward_progress > 0:
                r += self.scaler * forward_progress

        # Bridge crossing rewards
        if xp is not None and x is not None:
            for bridge_x in self.bridge_positions:
                if bridge_x not in self._bridges_crossed and xp < bridge_x <= x:
                    r += self.bridge_reward
                    self._bridges_crossed.add(bridge_x)

        # Death penalty
        dead_now = bool(infos.get(self.dead_key, 0))
        just_died = not self._prev_dead and dead_now

        if just_died:
            r += self.death_penalty
            self._bridges_crossed = set()  # Reset bridge tracking on death

        self._prev_x = x
        self._prev_dead = dead_now

        self._episode_shaped_return += r

        done = terminations | truncations
        if done:
            infos["true_objective"] = self._episode_shaped_return

        return obs, r, terminations, truncations, infos
