import gymnasium as gym
import math
import numpy as np


class RhythmSyncAdditionalInput(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        max_rel_x=512.0,
        max_rel_y=256.0,
        max_distance=768.0,
        switch_use_range=64.0,
    ):
        '''
        current stage fraction
        completed fraction
        stage type / 2
        cue visible self
        pending self
        pending other
        cue owner (-1 regular/jitter, 0 A, 1 B)
        relative dx to own switch
        relative dy to own switch
        normalized Euclidean distance to own switch
        in switch zone
        '''

        super().__init__(env)
        current_obs_space = self.observation_space
        self.max_rel_x = float(max_rel_x)
        self.max_rel_y = float(max_rel_y)
        self.max_distance = float(max_distance)
        self.switch_use_range = float(switch_use_range)

        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": current_obs_space,
                "measurements": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            }
        )
        self.measurements_vec = np.zeros(low.shape[0], dtype=np.float32)

    @property
    def player_id(self):
        pid = getattr(self.env.unwrapped, "player_id", 0)
        return 0 if pid is None or pid < 0 else int(pid)

    def _pending_keys(self):
        if self.player_id == 0:
            return "USER27", "USER28"
        return "USER28", "USER27"

    def _target_keys(self):
        if self.player_id == 0:
            return "USER33", "USER34"
        return "USER35", "USER36"

    def _cue_visible_key(self):
        return "USER37" if self.player_id == 0 else "USER38"

    def _parse_info(self, obs, info):
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}
        measurements = obs_dict["measurements"]
        measurements.fill(0.0)

        if info is None:
            return obs_dict

        current_stage = max(0.0, float(info.get("USER24", 0.0)))
        completed = max(0.0, float(info.get("USER25", 0.0)))
        num_sections = max(1.0, float(info.get("USER32", 1.0)))
        stage_type = max(0.0, min(2.0, float(info.get("USER26", 0.0))))
        cue_owner = float(info.get("USER31", -1.0))

        pending_self_key, pending_other_key = self._pending_keys()
        target_x_key, target_y_key = self._target_keys()
        target_x = float(info.get(target_x_key, 0.0))
        target_y = float(info.get(target_y_key, 0.0))
        pos_x = float(info.get("POSITION_X", 0.0))
        pos_y = float(info.get("POSITION_Y", 0.0))

        rel_x = np.clip((target_x - pos_x) / self.max_rel_x, -1.0, 1.0)
        rel_y = np.clip((target_y - pos_y) / self.max_rel_y, -1.0, 1.0)
        dx = target_x - pos_x
        dy = target_y - pos_y
        dist = math.hypot(dx, dy)
        distance = min(1.0, dist / self.max_distance)
        in_switch_zone = float(abs(dx) <= self.switch_use_range and abs(dy) <= self.switch_use_range)

        measurements[0] = current_stage / num_sections
        measurements[1] = completed / num_sections
        measurements[2] = stage_type / 2.0
        measurements[3] = float(info.get(self._cue_visible_key(), 0.0) > 0.0)
        measurements[4] = float(info.get(pending_self_key, 0.0) > 0.0)
        measurements[5] = float(info.get(pending_other_key, 0.0) > 0.0)
        measurements[6] = np.clip(cue_owner, -1.0, 1.0)
        measurements[7] = rel_x
        measurements[8] = rel_y
        measurements[9] = distance
        measurements[10] = in_switch_zone
        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._parse_info(obs, info), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, rew, terminated, truncated, info
        return self._parse_info(obs, info), rew, terminated, truncated, info


class RhythmSyncRewardShapingDense(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        stage_completion_reward=1.0,
        success_bonus=1.0,
        failure_penalty=-1.0,
        step_penalty=-0.001,
        approach_switch_reward_scale=0.0005,
        max_approach_delta=128.0,
        switch_zone_entry_reward=0.05,
        switch_use_range=64.0,
    ):
        super().__init__(env)
        self.num_agents = int(max(1, getattr(self.env.unwrapped, "num_agents", 2)))
        self.stage_completion_reward = stage_completion_reward
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.step_penalty = step_penalty
        self.approach_switch_reward_scale = approach_switch_reward_scale
        self.max_approach_delta = max_approach_delta
        self.switch_zone_entry_reward = switch_zone_entry_reward
        self.switch_use_range = switch_use_range

        self.prev_vars = {}
        self.orig_env_reward = 0.0
        self._switch_zone_rewarded_stage = None
        self._best_distance_this_stage = None

    @property
    def player_id(self):
        pid = getattr(self.env.unwrapped, "player_id", 0)
        return 0 if pid is None or pid < 0 else int(pid)

    def _team_share(self, reward):
        """Reards are duplicated per player (in parallel corridor), so should be divided.
        Off policy will sum them up again later. On policy will sum with alpha=1 or use bare reward if alpha=0"""
        return float(reward) / float(max(1, self.num_agents))

    def _pending_key(self):
        return "USER27" if self.player_id == 0 else "USER28"

    def _cue_visible_key(self):
        return "USER37" if self.player_id == 0 else "USER38"

    def _target_coords(self, info):
        x_key, y_key = ("USER33", "USER34") if self.player_id == 0 else ("USER35", "USER36")
        try:
            x = float(info.get(x_key, 0.0))
            y = float(info.get(y_key, 0.0))
        except (TypeError, ValueError):
            return None

        if x == 0.0 or y == 0.0:
            return None

        return x, y

    def _distance_to_switch(self, info):
        stage = int(info.get("USER24", 0))
        num_sections = int(info.get("USER32", 0))
        if stage <= 0 or num_sections <= 0 or stage > num_sections:
            return None

        target_coords = self._target_coords(info)
        if target_coords is None:
            return None

        px = float(info.get("POSITION_X", 0.0))
        py = float(info.get("POSITION_Y", 0.0))
        target_x, target_y = target_coords
        return math.hypot(px - target_x, py - target_y)

    def _in_switch_zone(self, info):
        target_coords = self._target_coords(info)
        if target_coords is None:
            return False

        px = float(info.get("POSITION_X", 0.0))
        py = float(info.get("POSITION_Y", 0.0))
        target_x, target_y = target_coords
        return abs(px - target_x) <= self.switch_use_range and abs(py - target_y) <= self.switch_use_range

    def _sync(self, info):
        cue_key = self._cue_visible_key()
        self.prev_vars = {
            "USER24": int(info.get("USER24", 0)),
            "USER25": int(info.get("USER25", 0)),
            "USER27": int(info.get("USER27", 0)),
            "USER28": int(info.get("USER28", 0)),
            "USER29": int(info.get("USER29", 0)),
            "USER30": int(info.get("USER30", 0)),
            "USER31": int(info.get("USER31", 0)),
            "USER32": int(info.get("USER32", 0)),
            "USER33": float(info.get("USER33", 0.0)),
            "USER34": float(info.get("USER34", 0.0)),
            "USER35": float(info.get("USER35", 0.0)),
            "USER36": float(info.get("USER36", 0.0)),
            "POSITION_X": float(info.get("POSITION_X", 0.0)),
            "POSITION_Y": float(info.get("POSITION_Y", 0.0)),
            "cue_visible": int(info.get(cue_key, 0)),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.orig_env_reward = 0.0
        self._switch_zone_rewarded_stage = None
        self._best_distance_this_stage = self._distance_to_switch(info)
        self._sync(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward is None or info is None:
            return obs, reward, terminated, truncated, info

        reward = float(reward)
        self.orig_env_reward += reward

        if not self.prev_vars:
            self._sync(info)
            return obs, reward, terminated, truncated, info

        shaped_reward = self.step_penalty

        prev_stage = self.prev_vars.get("USER24", 0)
        curr_stage = int(info.get("USER24", 0))
        stage_changed = curr_stage != prev_stage
        if stage_changed:
            self._switch_zone_rewarded_stage = None
            self._best_distance_this_stage = self._distance_to_switch(info)

        prev_completed = self.prev_vars.get("USER25", 0)
        curr_completed = int(info.get("USER25", 0))
        completed_delta = max(0, curr_completed - prev_completed)
        if completed_delta > 0:
            shaped_reward += self._team_share(completed_delta * self.stage_completion_reward)
            self._switch_zone_rewarded_stage = None

        prev_failed = self.prev_vars.get("USER29", 0)
        curr_failed = int(info.get("USER29", 0))
        if curr_failed > prev_failed:
            shaped_reward += self._team_share(self.failure_penalty)

        prev_finished = self.prev_vars.get("USER30", 0)
        curr_finished = int(info.get("USER30", 0))
        if curr_finished > prev_finished:
            shaped_reward += self._team_share(self.success_bonus)

        pending_self = int(info.get(self._pending_key(), 0))
        if not stage_changed and not curr_failed and not curr_finished and pending_self == 0:
            curr_distance = self._distance_to_switch(info)
            if curr_distance is not None:
                if self._best_distance_this_stage is None:
                    self._best_distance_this_stage = curr_distance
                elif curr_distance < self._best_distance_this_stage:
                    distance_delta = min(self._best_distance_this_stage - curr_distance, self.max_approach_delta)
                    shaped_reward += distance_delta * self.approach_switch_reward_scale
                    self._best_distance_this_stage = curr_distance

            in_switch_zone = self._in_switch_zone(info)
            if in_switch_zone and self._switch_zone_rewarded_stage != curr_stage:
                shaped_reward += self.switch_zone_entry_reward
                self._switch_zone_rewarded_stage = curr_stage

        done = bool(terminated) or bool(truncated)
        if done:
            num_sections = max(1, int(info.get("USER32", 0)))
            info["true_objective"] = float(info.get("USER25", 0)) / float(num_sections)
            info["success"] = bool(info.get("USER30", 0))

        total_reward = reward + shaped_reward
        self._sync(info)
        return obs, total_reward, terminated, truncated, info
