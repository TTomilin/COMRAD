import gymnasium as gym

class DoomPitfallRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        scaler: float = 0.01,
        death_penalty: float = -1.0,
        keep_lb: bool = True,
        goal_x: float | None = None,
        goal_reward: float = 10.0,
        pos_key: str = "POSITION_X", # This is quite pitfall specific
        dead_key: str = "DEAD",
        x_start: float = 32.0,
        team_score_adjust: float = 5.0,
    ):
        super().__init__(env)
        self.scaler = float(scaler)
        self.death_penalty = float(death_penalty)
        self.keep_lb = bool(keep_lb)
        self.goal_x = goal_x
        self.goal_reward = float(goal_reward)
        self.pos_key = pos_key
        self.dead_key = dead_key
        self.x_start = float(x_start)
        self.team_score_adjust = float(team_score_adjust)

        self._prev_x: float | None = None
        self._best_x: float = self.x_start
        self._goal_given: bool = False

        self._prev_dead: bool = True
        self.orig_env_reward: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0
        self._goal_given = False
        self._best_x = self.x_start
        self._prev_x = info.get(self.pos_key, None)
        self._prev_dead = bool(info.get(self.dead_key, 0))
        return obs, info

    def step(self, action):
        obs, based_rewards, terminations, truncations, infos = self.env.step(action)

        if based_rewards is None: return obs, based_rewards, terminations, truncations, infos

        r = float(based_rewards)
        self.orig_env_reward += r
        x = infos.get(self.pos_key, None)
        xp = self._prev_x

        if xp is not None and x is not None:
            if self.keep_lb:
                t = max(0.0, x - max(self._best_x, xp))
                if t > 0:
                    r += self.scaler * t
                    self._best_x = max(self._best_x, x)
            else:
                d = max(0.0, x - xp)
                if d > 0:
                    r += self.scaler * d
                    self._best_x = max(self._best_x, x)

        dead_now = bool(infos.get(self.dead_key, 0))
        just_died = not self._prev_dead and dead_now

        if just_died:
            r += self.death_penalty
            self._best_x = self.x_start

        if self.goal_x is not None and not self._goal_given and x is not None and x > float(self.goal_x):
            r += self.goal_reward
            self._goal_given = True
        self._prev_x = x
        self._prev_dead = dead_now

        done = terminations | truncations
        if done:
            infos["true_objective"] = self.orig_env_reward
            if self._goal_given:
                r += self.team_score_adjust
            # elif "one_agent_died" in infos.get("episode_extra_stats", {}):
            #     r -= self.team_score_adjust

        return obs, r, terminations, truncations, infos
