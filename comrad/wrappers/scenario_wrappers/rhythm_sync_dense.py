import gymnasium as gym


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
        switch_use_range=96.0,
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
        return abs(px - target_x) + abs(py - target_y)

    def _in_switch_zone(self, info):
        distance = self._distance_to_switch(info)
        return distance is not None and distance <= (2.0 * self.switch_use_range)

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
            return obs, 0.0, terminated, truncated, info

        shaped_reward = self.step_penalty

        prev_stage = self.prev_vars.get("USER24", 0)
        curr_stage = int(info.get("USER24", 0))
        if curr_stage != prev_stage:
            self._switch_zone_rewarded_stage = None

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
        if not curr_failed and not curr_finished and pending_self == 0:
            prev_distance = self._distance_to_switch(self.prev_vars)
            curr_distance = self._distance_to_switch(info)
            if prev_distance is not None and curr_distance is not None:
                distance_delta = prev_distance - curr_distance
                if distance_delta > 0:
                    distance_delta = min(distance_delta, self.max_approach_delta)
                    shaped_reward += distance_delta * self.approach_switch_reward_scale

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
