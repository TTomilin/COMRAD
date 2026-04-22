import gymnasium as gym


class StagHuntArenaRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        rabbit_reward=0.1,
        stag_team_reward=2.0,
    ):
        super().__init__(env)
        self.rabbit_reward = float(rabbit_reward)
        self.stag_team_reward = float(stag_team_reward)

        self.prev_stag_kills = None
        self.prev_own_rabbit_kills = None
        self.orig_env_reward = 0.0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _player_id(self) -> int:
        return int(max(0, getattr(self.env.unwrapped, "player_id", 0)))

    def _own_rabbit_key(self) -> str:
        return f"USER{56 + self._player_id()}"

    @staticmethod
    def _float(info, key: str, default: float = 0.0) -> float:
        return float(info.get(key, default))

    def _sync(self, info):
        if info is None:
            self.prev_stag_kills = None
            self.prev_own_rabbit_kills = None
            return

        self.prev_stag_kills = self._float(info, "USER54")
        self.prev_own_rabbit_kills = self._float(info, self._own_rabbit_key())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0
        self._sync(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None:
            reward = 0.0
        reward = float(reward)
        self.orig_env_reward += reward

        if info is None:
            return obs, reward, terminated, truncated, info

        curr_stag_kills = self._float(info, "USER54")
        curr_own_rabbit_kills = self._float(info, self._own_rabbit_key())

        if self.prev_stag_kills is None or self.prev_own_rabbit_kills is None:
            self._sync(info)
            info["true_objective"] = curr_stag_kills
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            return obs, reward, terminated, truncated, info

        shaped_reward = 0.0

        delta_own_rabbit_kills = curr_own_rabbit_kills - self.prev_own_rabbit_kills
        if delta_own_rabbit_kills > 0.0:
            shaped_reward += delta_own_rabbit_kills * self.rabbit_reward

        delta_stag_kills = curr_stag_kills - self.prev_stag_kills
        if delta_stag_kills > 0.0:
            shaped_reward += delta_stag_kills * (self.stag_team_reward / float(self._num_agents()))

        total_reward = reward + shaped_reward

        info["true_objective"] = curr_stag_kills
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
