import gymnasium as gym


class DoomSafeGround2RewardShaping(gym.Wrapper):
    """
    Safe Ground 2 reward shaping (team reward).

    Core signal:
    - Reward when Shooter's ammo increases (interpreted as successful transfer from Mover).
    Penalties:
    - Penalty when an agent dies (especially Shooter).
    """

    def __init__(
        self,
        env,
        *,
        ammo_key: str = "AMMO2",
        dead_key: str = "DEAD",
        shooter_idx: int = 0,
        mover_idx: int = 1,
        shooter_ammo_gain_scale: float = 0.3,
        mover_ammo_gain_scale: float = 0.1, ##distinguish ammo gain reward between shooter and mover
        shooter_death_penalty: float = -3.0,
        mover_death_penalty: float = -5.0, ##this case shooter fails to protect mover, more punishment added
        step_penalty: float = 0.0, 
    ):
        super().__init__(env)
        self.ammo_key = ammo_key
        self.dead_key = dead_key
        self.shooter_idx = int(shooter_idx)
        self.mover_idx = int(mover_idx)
        self.shooter_ammo_gain_scale = float(shooter_ammo_gain_scale)
        self.mover_ammo_gain_scale = float(mover_ammo_gain_scale)
        self.shooter_death_penalty = float(shooter_death_penalty)
        self.mover_death_penalty = float(mover_death_penalty)
        self.step_penalty = float(step_penalty)

        self._prev_shooter_ammo = None
        self._prev_dead_shooter = True
        self._prev_dead_mover = True
        self._episode_return = 0.0

    @staticmethod
    def _get_agent_value(infos, key: str, idx: int):
        """Robustly read per-agent values when infos[key] is scalar or a list/tuple."""
        if infos is None:
            return None
        v = infos.get(key, None)
        if isinstance(v, (list, tuple)):
            if 0 <= idx < len(v):
                return v[idx]
            return None
        return v

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0

        shooter_ammo = self._get_agent_value(info, self.ammo_key, self.shooter_idx)
        self._prev_shooter_ammo = shooter_ammo
        mover_ammo = self._get_agent_value(info, self.ammo_key, self.mover_idx)
        self._prev_mover_ammo = mover_ammo

        self._prev_dead_shooter = bool(self._get_agent_value(info, self.dead_key, self.shooter_idx) or 0)
        self._prev_dead_mover = bool(self._get_agent_value(info, self.dead_key, self.mover_idx) or 0)

        return obs, info

    def step(self, action):
        obs, base_rewards, terminations, truncations, infos = self.env.step(action)

        # If upstream returns None rewards (some wrappers do during init), preserve behavior
        if base_rewards is None:
            return obs, base_rewards, terminations, truncations, infos

        r = float(base_rewards)

        # Step penalty (optional)
        if self.step_penalty != 0.0:
            r += self.step_penalty

        ## Reward for ammo gain of Shooter (transfer success proxy)
        shooter_ammo = self._get_agent_value(infos, self.ammo_key, self.shooter_idx)
        if shooter_ammo is not None and self._prev_shooter_ammo is not None:
            delta = float(shooter_ammo) - float(self._prev_shooter_ammo)
            if delta > 0:
                r += self.shooter_ammo_gain_scale * delta

        ## Reward for ammo gain of Mover
        mover_ammo = self._get_agent_value(infos, self.ammo_key, self.mover_idx)
        if mover_ammo is not None and self._prev_mover_ammo is not None:
            delta = float(mover_ammo) - float(self._prev_mover_ammo)
            if delta > 0:
                r += self.mover_ammo_gain_scale * delta

        # Death penalties (use DEAD flag provided by env info)
        dead_shooter = bool(self._get_agent_value(infos, self.dead_key, self.shooter_idx) or 0)
        dead_mover = bool(self._get_agent_value(infos, self.dead_key, self.mover_idx) or 0)

        just_died_shooter = (not self._prev_dead_shooter) and dead_shooter
        just_died_mover = (not self._prev_dead_mover) and dead_mover

        if just_died_shooter:
            r += self.shooter_death_penalty
        if just_died_mover:
            r += self.mover_death_penalty

        # Episode bookkeeping
        self._episode_return += r
        self._prev_shooter_ammo = shooter_ammo
        self._prev_mover_ammo = mover_ammo
        self._prev_dead_shooter = dead_shooter
        self._prev_dead_mover = dead_mover

        done = terminations | truncations
        if done:
            # Keep the same convention as pitfall wrapper
            infos["true_objective"] = self._episode_return

        return obs, r, terminations, truncations, infos
