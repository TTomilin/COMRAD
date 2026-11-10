from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


class SharedVisionWrapper(gym.Wrapper):
    """
    Oracle "shared vision" / full observability for a 2-agent MultiAgentEnv: every agent observes its own frame
    (or 1D state vector) stacked with its partner's along the channel/feature axis ([own, partner]).
    When `stack_measurements=True` or `include_relative_pose=True`, also stacks `[own_measurements, partner_measurements]`
    and appends the 5D normalized relative pose `[(dx)/scale, (dy)/scale, dist/scale, cos(d_yaw), sin(d_yaw)]`.
    """

    _REL_POSE_LOW = np.array([-1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    _REL_POSE_HIGH = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        env: gym.Env,
        obs_key: str = "obs",
        stack_measurements: bool = False,
        include_relative_pose: bool = False,
        map_scale: float = 2048.0,
    ) -> None:
        super().__init__(env)
        if getattr(env, "num_agents", 1) != 2:
            raise ValueError(f"SharedVisionWrapper expects a 2-agent env, got num_agents={getattr(env, 'num_agents', 1)}")
        self.num_agents = 2
        self.is_multiagent = True
        self.obs_key = obs_key
        self.stack_measurements = bool(stack_measurements)
        self.include_relative_pose = bool(include_relative_pose)
        self.map_scale = max(1.0, float(map_scale))

        space = env.observation_space
        self._input_is_dict = isinstance(space, gym.spaces.Dict)
        img_space = space[obs_key] if self._input_is_dict else space
        shape = img_space.shape
        if len(shape) == 1:
            self.channel_axis = 0
        else:
            self.channel_axis = 0 if shape[0] <= 4 else len(shape) - 1
        new_shape = list(shape)
        new_shape[self.channel_axis] *= 2
        low = np.concatenate([img_space.low, img_space.low], axis=self.channel_axis)
        high = np.concatenate([img_space.high, img_space.high], axis=self.channel_axis)
        stacked = gym.spaces.Box(low=low, high=high, shape=tuple(new_shape), dtype=img_space.dtype)

        has_meas = self._input_is_dict and "measurements" in space.spaces
        self._has_input_measurements = has_meas
        # Attach/extend measurements when input already has measurements or when relative pose is requested on 3D obs
        self._emit_measurements = False
        meas_box = None
        if has_meas and (self.stack_measurements or self.include_relative_pose):
            self._emit_measurements = True
            orig_m = space["measurements"]
            m_lows = [orig_m.low.astype(np.float32)]
            m_highs = [orig_m.high.astype(np.float32)]
            if self.stack_measurements:
                m_lows.append(orig_m.low.astype(np.float32))
                m_highs.append(orig_m.high.astype(np.float32))
            if self.include_relative_pose:
                m_lows.append(self._REL_POSE_LOW)
                m_highs.append(self._REL_POSE_HIGH)
            m_low = np.concatenate(m_lows, axis=0)
            m_high = np.concatenate(m_highs, axis=0)
            meas_box = gym.spaces.Box(low=m_low, high=m_high, shape=m_low.shape, dtype=np.float32)
        elif not has_meas and self.include_relative_pose and len(shape) > 1:
            self._emit_measurements = True
            meas_box = gym.spaces.Box(
                low=self._REL_POSE_LOW.copy(),
                high=self._REL_POSE_HIGH.copy(),
                shape=(5,),
                dtype=np.float32,
            )

        if self._input_is_dict or self._emit_measurements:
            spaces = dict(space.spaces) if self._input_is_dict else {}
            spaces[obs_key] = stacked
            if self._emit_measurements and meas_box is not None:
                spaces["measurements"] = meas_box
            self.observation_space = gym.spaces.Dict(spaces)
            self._output_is_dict = True
        else:
            self.observation_space = stacked
            self._output_is_dict = False

    def _img(self, obs):
        return obs[self.obs_key] if isinstance(obs, dict) else obs

    @staticmethod
    def _pose_from_info(info: dict | None) -> tuple[float, float, float]:
        if not isinstance(info, dict):
            return 0.0, 0.0, 0.0
        pos = info.get("pos") if isinstance(info.get("pos"), dict) else {}
        raw_x = info.get("POSITION_X", pos.get("agent_x", 0.0))
        raw_y = info.get("POSITION_Y", pos.get("agent_y", 0.0))
        raw_a = info.get("ANGLE", pos.get("agent_a", 0.0))
        try:
            x = float(raw_x)
            if not math.isfinite(x):
                x = 0.0
        except (TypeError, ValueError):
            x = 0.0
        try:
            y = float(raw_y)
            if not math.isfinite(y):
                y = 0.0
        except (TypeError, ValueError):
            y = 0.0
        try:
            a = float(raw_a)
            if not math.isfinite(a):
                a = 0.0
        except (TypeError, ValueError):
            a = 0.0
        return x, y, a

    def _relative_pose(self, info_i: dict | None, info_j: dict | None) -> np.ndarray:
        x_i, y_i, a_i = self._pose_from_info(info_i)
        x_j, y_j, a_j = self._pose_from_info(info_j)
        dx = float(np.clip((x_j - x_i) / self.map_scale, -1.0, 1.0))
        dy = float(np.clip((y_j - y_i) / self.map_scale, -1.0, 1.0))
        dist = float(np.clip(math.hypot(x_j - x_i, y_j - y_i) / self.map_scale, 0.0, 2.0))
        d_rad = math.radians(a_j - a_i)
        return np.array([dx, dy, dist, math.cos(d_rad), math.sin(d_rad)], dtype=np.float32)

    def _stack(self, obs_list, infos=None):
        if obs_list is None or any(o is None for o in obs_list):
            return obs_list
        imgs = [self._img(o) for o in obs_list]
        info_list = infos if isinstance(infos, (list, tuple)) and len(infos) == 2 else [{}, {}]
        out = []
        for i, obs in enumerate(obs_list):
            partner_idx = 1 - i
            stacked = np.concatenate([imgs[i], imgs[partner_idx]], axis=self.channel_axis)
            if self._output_is_dict:
                obs_dict = dict(obs) if isinstance(obs, dict) else {}
                obs_dict[self.obs_key] = stacked
                if self._emit_measurements:
                    parts = []
                    if self._has_input_measurements:
                        own_m = np.asarray(obs_list[i]["measurements"], dtype=np.float32)
                        parts.append(own_m)
                        if self.stack_measurements:
                            partner_m = np.asarray(obs_list[partner_idx]["measurements"], dtype=np.float32)
                            parts.append(partner_m)
                    if self.include_relative_pose:
                        parts.append(self._relative_pose(info_list[i], info_list[partner_idx]))
                    obs_dict["measurements"] = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
                out.append(obs_dict)
            else:
                out.append(stacked)
        return out

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return self._stack(obs, infos), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        pose_infos = infos
        # On an auto-reset step `obs` is already the next episode's first frame while `infos` still describe the
        # terminal frame; pair the new frame with the new episode's pose from `reset_info`.
        if (
            isinstance(infos, (list, tuple))
            and all(bool(t) or bool(tr) for t, tr in zip(terminated, truncated))
            and all(isinstance(i, dict) and isinstance(i.get("reset_info"), dict) for i in infos)
        ):
            pose_infos = [i["reset_info"] for i in infos]
        return self._stack(obs, pose_infos), rewards, terminated, truncated, infos
