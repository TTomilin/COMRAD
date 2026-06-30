import gymnasium as gym
import numpy as np


class ForagingCommonsAdditionalInput(gym.Wrapper):
    def __init__(self, env, cleanup_time=72.0):
        super().__init__(env)
        current_obs_space = self.observation_space
        self.cleanup_time = max(float(cleanup_time), 1.0)

        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([200.0, 100.0, 500.0, 200.0, 200.0, 500.0, 200.0, 1.0], dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": current_obs_space,
                "measurements": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            }
        )

        self.measurements_vec = np.zeros(8, dtype=np.float32)

    def _player_index(self) -> int:
        return int(max(0, min(3, getattr(self.env.unwrapped, "player_id", 0))))

    def _key(self, base: int) -> str:
        return f"USER{base + self._player_index()}"

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}
        if info is None:
            self.measurements_vec.fill(0.0)
            return obs_dict

        own_cleanups = max(0.0, info.get(self._key(8), 0.0))
        total_cleanups = max(0.0, info.get("USER3", 0.0))

        self.measurements_vec[0] = max(0.0, info.get("HEALTH", 0.0))
        self.measurements_vec[1] = max(0.0, info.get("USER1", 0.0))
        self.measurements_vec[2] = max(0.0, info.get(self._key(4), 0.0))
        self.measurements_vec[3] = own_cleanups
        self.measurements_vec[4] = max(0.0, total_cleanups - own_cleanups)
        self.measurements_vec[5] = max(0.0, info.get("USER2", 0.0))
        self.measurements_vec[6] = total_cleanups
        self.measurements_vec[7] = min(1.0, max(0.0, info.get(self._key(12), 0.0) / self.cleanup_time))

        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.measurements_vec.fill(0.0)
        return self._parse_info(obs, info), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, rew, terminated, truncated, info
        return self._parse_info(obs, info), rew, terminated, truncated, info


class ForagingCommonsRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        harvest_reward=0.03,
        # cleanup_reward=0.25,
        cleanup_reward=0.03,
        death_penalty=-1.0,
        track_coop=True,
    ):
        super().__init__(env)
        self.harvest_reward = harvest_reward
        self.cleanup_reward = cleanup_reward
        self.death_penalty = death_penalty

        self.prev_vars = {}
        self._step_count = 0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self.track_coop = track_coop

    def _player_index(self) -> int:
        return int(max(0, min(3, getattr(self.env.unwrapped, "player_id", 0))))

    def _key(self, base: int) -> str:
        return f"USER{base + self._player_index()}"

    def _sync(self, info):
        if info is None:
            self.prev_vars = {}
            return

        self.prev_vars = {
            "HEALTH": info.get("HEALTH", 0.0),
            "USER1": info.get("USER1", 0.0),
            "OWN_HARVESTS": info.get(self._key(4), 0.0),
            "OWN_CLEANUPS": info.get(self._key(8), 0.0),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if info is not None and self.track_coop:
            self._record_episode_stats(info)
        self.prev_vars = {}
        self._step_count = 0
        self.episode_coop_steps = 0
        self.episode_defect_steps = 0
        self._sync(info)
        return obs, info

    def _record_episode_stats(self, info):
        extra = info.setdefault("episode_extra_stats", {})
        total = self.episode_coop_steps + self.episode_defect_steps
        extra["coop_steps"] = self.episode_coop_steps
        extra["defect_steps"] = self.episode_defect_steps
        extra["total_coop_defect_steps"] = total
        if total > 0:
            extra["cooperation_index"] = self.episode_coop_steps / total
            extra["defector_index"] = self.episode_defect_steps / total

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward is None:
            reward = 0.0
        reward = float(reward)

        self._step_count += 1

        if info is None:
            return obs, reward, terminated, truncated, info

        if not self.prev_vars:
            self._sync(info)
            if terminated or truncated:
                info["true_objective"] = float(self._step_count)
            return obs, reward, terminated, truncated, info

        shaped_reward = 0.0

        curr_health = info.get("HEALTH", 0.0)
        curr_harvests = info.get(self._key(4), 0.0)
        curr_cleanups = info.get(self._key(8), 0.0)

        prev_health = self.prev_vars.get("HEALTH", 0.0)
        prev_harvests = self.prev_vars.get("OWN_HARVESTS", 0.0)
        prev_cleanups = self.prev_vars.get("OWN_CLEANUPS", 0.0)

        delta_harvests = curr_harvests - prev_harvests
        if delta_harvests > 0:
            shaped_reward += delta_harvests * self.harvest_reward

        delta_cleanups = curr_cleanups - prev_cleanups
        if delta_cleanups > 0:
            shaped_reward += delta_cleanups * self.cleanup_reward

        if curr_health <= 0 < prev_health:
            shaped_reward += self.death_penalty

        reward += shaped_reward

        if self.track_coop:
            delta_harvests = curr_harvests - prev_harvests
            delta_cleanups = curr_cleanups - prev_cleanups
            coop = 1.0 if delta_cleanups > 0 and delta_harvests == 0 else 0.0
            defect = 1.0 if delta_harvests > 0 and delta_cleanups == 0 else 0.0
            info["coop_step_signal"] = coop
            info["defect_step_signal"] = defect

            self.episode_coop_steps += coop
            self.episode_defect_steps += defect

            if coop > 0.0 or defect > 0.0:
                net_coop = coop - defect
                info.setdefault("episode_extra_stats", {})["cooperation_index"] = max(0.0, net_coop)
                info["episode_extra_stats"]["defector_index"] = max(0.0, -net_coop)

        if terminated or truncated:
            info["true_objective"] = float(self._step_count)
            if self.track_coop:
                self._record_episode_stats(info)

        self._sync(info)
        return obs, reward, terminated, truncated, info
