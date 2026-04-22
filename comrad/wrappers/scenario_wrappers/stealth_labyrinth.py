import gymnasium as gym


class StealthLabyrinthRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        kill_reward=1.0,
        room_discovery_reward=0.04,
        coordinated_discovery_bonus=0.02,
        all_demons_cleared_bonus=0.5,
        death_penalty=-1.0,
        soft_distance_limit=960.0,
        separation_penalty_scale=0.00005,
        max_separation_penalty=0.03,
    ):
        super().__init__(env)
        self.kill_reward = float(kill_reward)
        self.room_discovery_reward = float(room_discovery_reward)
        self.coordinated_discovery_bonus = float(coordinated_discovery_bonus)
        self.all_demons_cleared_bonus = float(all_demons_cleared_bonus)
        self.death_penalty = float(death_penalty)
        self.soft_distance_limit = max(float(soft_distance_limit), 0.0)
        self.separation_penalty_scale = max(float(separation_penalty_scale), 0.0)
        self.max_separation_penalty = max(float(max_separation_penalty), 0.0)

        self.prev_team_kills = None
        self.prev_demons_alive = None
        self.prev_torch_room = None
        self.prev_distance = None
        self.prev_p1_alive = None
        self.prev_p2_alive = None
        self.seen_rooms = set()
        self.orig_env_reward = 0.0

    def _num_agents(self) -> int:
        return int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))

    def _reward_share(self) -> float:
        return 1.0 / float(self._num_agents())

    @staticmethod
    def _int(info, key: str, default: int = 0) -> int:
        try:
            return int(info.get(key, default))
        except (TypeError, ValueError, OverflowError):
            return default

    def _sync(self, info):
        if info is None:
            self.prev_team_kills = None
            self.prev_demons_alive = None
            self.prev_torch_room = None
            self.prev_distance = None
            self.prev_p1_alive = None
            self.prev_p2_alive = None
            return

        self.prev_team_kills = max(0, self._int(info, "USER46"))
        self.prev_demons_alive = max(0, self._int(info, "USER44"))
        self.prev_torch_room = self._int(info, "USER45", -1)
        self.prev_distance = max(0, self._int(info, "USER43"))
        self.prev_p1_alive = max(0, self._int(info, "USER41"))
        self.prev_p2_alive = max(0, self._int(info, "USER42"))

        if self.prev_torch_room >= 0:
            self.seen_rooms.add(self.prev_torch_room)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_team_kills = None
        self.prev_demons_alive = None
        self.prev_torch_room = None
        self.prev_distance = None
        self.prev_p1_alive = None
        self.prev_p2_alive = None
        self.seen_rooms = set()
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

        curr_team_kills = max(0, self._int(info, "USER46"))
        curr_demons_alive = max(0, self._int(info, "USER44"))
        curr_torch_room = self._int(info, "USER45", -1)
        curr_distance = max(0, self._int(info, "USER43"))
        curr_p1_alive = max(0, self._int(info, "USER41"))
        curr_p2_alive = max(0, self._int(info, "USER42"))

        if (
            self.prev_team_kills is None
            or self.prev_demons_alive is None
            or self.prev_distance is None
            or self.prev_p1_alive is None
            or self.prev_p2_alive is None
        ):
            info["true_objective"] = float(curr_team_kills)
            if terminated or truncated:
                info["orig_env_reward"] = self.orig_env_reward
            self._sync(info)
            return obs, reward, terminated, truncated, info

        shaped_team_reward = 0.0

        delta_team_kills = curr_team_kills - self.prev_team_kills
        if delta_team_kills > 0:
            shaped_team_reward += delta_team_kills * self.kill_reward

        if curr_torch_room >= 0 and curr_torch_room not in self.seen_rooms:
            self.seen_rooms.add(curr_torch_room)
            shaped_team_reward += self.room_discovery_reward
            if curr_p1_alive > 0 and curr_p2_alive > 0 and curr_distance <= self.soft_distance_limit:
                shaped_team_reward += self.coordinated_discovery_bonus

        if curr_p1_alive > 0 and curr_p2_alive > 0 and self.separation_penalty_scale > 0.0:
            excess_distance = max(0.0, float(curr_distance) - self.soft_distance_limit)
            if excess_distance > 0.0:
                shaped_team_reward -= min(
                    self.max_separation_penalty,
                    excess_distance * self.separation_penalty_scale,
                )

        if curr_demons_alive == 0 and self.prev_demons_alive > 0:
            shaped_team_reward += self.all_demons_cleared_bonus

        prev_team_alive = self.prev_p1_alive > 0 and self.prev_p2_alive > 0
        curr_team_alive = curr_p1_alive > 0 and curr_p2_alive > 0
        if prev_team_alive and not curr_team_alive:
            shaped_team_reward += self.death_penalty

        total_reward = reward + shaped_team_reward * self._reward_share()

        info["true_objective"] = float(curr_team_kills)
        if terminated or truncated:
            info["orig_env_reward"] = self.orig_env_reward

        self._sync(info)
        return obs, total_reward, terminated, truncated, info
