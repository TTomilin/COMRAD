from __future__ import annotations

import math
from typing import Sequence

import gymnasium as gym
import numpy as np

from comrad.envs.doom_gym import GLOBAL_STATE_INFO_KEYS
from comrad.wrappers.state_observation import ego_pose_vitals, safe_float

# game variables already covered by the agents block (or irrelevant to the task)
POSE_VARIABLES = frozenset(
    {"POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "PITCH", "ROLL", "VIEW_HEIGHT", "HEALTH",
     "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z", "CAMERA_POSITION_X", "CAMERA_POSITION_Y", "CAMERA_POSITION_Z",
     "CAMERA_ANGLE", "CAMERA_PITCH", "CAMERA_ROLL", "CAMERA_FOV"}
)
DEFAULT_MAP_HALF_EXTENT = 2048.0
# symlog clip of the task block per encoding version: v1 clipped at |raw| ~ 147 (e.g. a 1000-HP core read as constant
# until it fell below 148, switch coordinates were constant); v2 keeps values up to ~1.6e5 distinct.
TASK_SYMLOG_CLIP = {1: 5.0, 2: 12.0}
AMMO_VARIABLES = tuple(f"AMMO{k}" for k in range(10))


def held_ammo(info: dict | None) -> float:
    """Ammo of the held weapon (SELECTED_WEAPON_AMMO), else the largest AMMOk exported by the scenario."""
    if not isinstance(info, dict):
        return 0.0
    if "SELECTED_WEAPON_AMMO" in info:
        return max(0.0, safe_float(info, "SELECTED_WEAPON_AMMO", 0.0))
    return max([0.0] + [safe_float(info, key, 0.0) for key in AMMO_VARIABLES if key in info])


def task_variable_keys(game_variables: Sequence[str]) -> list[str]:
    """Non-pose game variables in a stable (sorted) order."""
    return sorted({str(v).upper() for v in game_variables} - POSE_VARIABLES)


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def rasterize_blocking_lines(
    lines: np.ndarray, bounds: tuple[float, float, float, float], grid_size: int, anisotropic: bool = False
) -> np.ndarray:
    """Blocked-line coverage of each cell (row = y, col = x).

    Default (v1): square cells of the larger extent, length in cell widths clipped to 1. `anisotropic` (v2): cells
    span each axis of `bounds` separately (long corridors use all rows), length in units of sqrt(cell w * h), and the
    value is 1 - exp(-coverage / 2) so dense cells stay distinguishable instead of saturating at 1.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if lines is None or len(lines) == 0:
        return grid
    lines = np.asarray(lines, dtype=np.float32)
    lines = lines[lines[:, 4] > 0.5]
    if len(lines) == 0:
        return grid
    x0, y0, x1, y1 = bounds
    if anisotropic:
        cw, ch = max(x1 - x0, 1e-6) / grid_size, max(y1 - y0, 1e-6) / grid_size
        side = math.sqrt(cw * ch)
        step = min(cw, ch) / 8.0
    else:
        cw = ch = side = max(x1 - x0, y1 - y0, 1e-6) / grid_size
        step = side / 8.0
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    n = np.maximum(1, np.ceil(lengths / step).astype(np.int64))
    seg = np.repeat(np.arange(len(lines)), n)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = (offsets + 0.5) / n[seg]
    px = lines[seg, 0] + t * (lines[seg, 2] - lines[seg, 0])
    py = lines[seg, 1] + t * (lines[seg, 3] - lines[seg, 1])
    cx = np.clip(((px - x0) / cw).astype(np.int64), 0, grid_size - 1)
    cy = np.clip(((py - y0) / ch).astype(np.int64), 0, grid_size - 1)
    np.add.at(grid, (cy, cx), (lengths[seg] / n[seg]) / side)
    if anisotropic:
        return (1.0 - np.exp(-grid / 2.0)).astype(np.float32)
    return np.clip(grid, 0.0, 1.0)


class GlobalStateWrapper(gym.Wrapper):
    """
    True full observability for a MultiAgentEnv: every agent keeps its own frame in `obs` and receives the global
    environment state in `measurements` (appended after any existing measurements).

    Per-agent vector (own agent first, other agents in index order):
      agents   N x 9          x, y, z, cos yaw, sin yaw, health, ammo, alive, is_self
      task     N x task_dim   symlog of each agent's non-pose game variables (sorted keys, zero padded)
      objects  max_objects x (6 + type_bits)
                              present, x, y, z, vx, vy, type code (+-1 bits of the 16-bit actor name hash);
                              every non-player object, nearest to self first
      layout   grid x grid    blocked-line coverage of an allocentric grid over the map bounds
      time     1              step / horizon
    Positions are centred on the map and divided by its half extent, so they lie in [-1, 1].
    `version` 1 reproduces the original encoding (task clip 5, ammo from AMMO1 first, one half extent and square
    grid cells for both axes) for archived checkpoints. Version 2 clips the task block at TASK_SYMLOG_CLIP[2], reads
    the held weapon's ammo, and normalizes positions and the grid per axis (a long corridor such as Rhythm Sync
    otherwise collapses into one saturated grid row and near-constant y coordinates).
    Objects and layout come from agent 0's `info` (`_gs_objects`, `_gs_lines`, `_gs_layout_id`).
    """

    AGENT_DIM = 9
    OBJECT_BASE_DIM = 6
    VELOCITY_SCALE = 32.0
    Z_SCALE = 512.0

    def __init__(
        self,
        env: gym.Env,
        horizon: int,
        game_variables: Sequence[str] = (),
        max_objects: int = 48,
        grid_size: int = 16,
        task_dim: int = 24,
        type_bits: int = 16,
        obs_key: str = "obs",
        version: int = 2,
    ) -> None:
        super().__init__(env)
        if int(version) not in TASK_SYMLOG_CLIP:
            raise ValueError(f"Unsupported global_state version {version!r}")
        self.version = int(version)
        self.task_clip = TASK_SYMLOG_CLIP[self.version]
        self.num_agents = int(getattr(env, "num_agents", 1))
        self.is_multiagent = True
        self.horizon = max(1, int(horizon))
        self.task_keys = task_variable_keys(game_variables)[:task_dim]
        self.max_objects = int(max_objects)
        self.grid_size = int(grid_size)
        self.task_dim = int(task_dim)
        self.type_bits = int(type_bits)
        self.obs_key = obs_key

        self.object_dim = self.OBJECT_BASE_DIM + self.type_bits
        self.state_dim = (
            self.num_agents * (self.AGENT_DIM + self.task_dim)
            + self.max_objects * self.object_dim
            + self.grid_size * self.grid_size
            + 1
        )

        space = env.observation_space
        self._input_is_dict = isinstance(space, gym.spaces.Dict)
        spaces = dict(space.spaces) if self._input_is_dict else {obs_key: space}
        bound = max(5.0, self.task_clip)
        gs_low = np.full(self.state_dim, -bound, dtype=np.float32)
        gs_high = np.full(self.state_dim, bound, dtype=np.float32)
        self._has_input_measurements = "measurements" in spaces
        if self._has_input_measurements:
            m = spaces["measurements"]
            gs_low = np.concatenate([m.low.astype(np.float32), gs_low])
            gs_high = np.concatenate([m.high.astype(np.float32), gs_high])
        spaces["measurements"] = gym.spaces.Box(low=gs_low, high=gs_high, shape=gs_low.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

        self._step_count = 0
        self._layout_id = None
        self._bounds = (-DEFAULT_MAP_HALF_EXTENT, -DEFAULT_MAP_HALF_EXTENT, DEFAULT_MAP_HALF_EXTENT, DEFAULT_MAP_HALF_EXTENT)
        self._grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

    # ---------------------------------------------------------------- layout
    def _reset_layout(self) -> None:
        self._layout_id = None
        self._bounds = (-DEFAULT_MAP_HALF_EXTENT, -DEFAULT_MAP_HALF_EXTENT, DEFAULT_MAP_HALF_EXTENT, DEFAULT_MAP_HALF_EXTENT)
        self._grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

    def _update_layout(self, info: dict | None) -> None:
        if not isinstance(info, dict) or "_gs_layout_id" not in info:
            self._reset_layout()
            return
        lines = info.get("_gs_lines")
        if lines is not None:
            lines = np.asarray(lines, dtype=np.float32).reshape(-1, 5)
            if len(lines):
                xs = np.concatenate([lines[:, 0], lines[:, 2]])
                ys = np.concatenate([lines[:, 1], lines[:, 3]])
                cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
                if self.version >= 2:
                    hx = max(0.5 * (xs.max() - xs.min()), 1.0)
                    hy = max(0.5 * (ys.max() - ys.min()), 1.0)
                else:
                    hx = hy = max(0.5 * (xs.max() - xs.min()), 0.5 * (ys.max() - ys.min()), 1.0)
                self._bounds = (cx - hx, cy - hy, cx + hx, cy + hy)
            self._grid = rasterize_blocking_lines(lines, self._bounds, self.grid_size, anisotropic=self.version >= 2)
        elif self._layout_id is None:
            self._reset_layout()
            return
        self._layout_id = info["_gs_layout_id"]

    @property
    def _center_half(self):
        """(cx, cy, half x, half y); v1 bounds are square, so both halves are equal there."""
        x0, y0, x1, y1 = self._bounds
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1), max(0.5 * (x1 - x0), 1.0), max(0.5 * (y1 - y0), 1.0)

    # ---------------------------------------------------------------- blocks
    def _agent_rows(self, infos) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cx, cy, hx, hy = self._center_half
        agents = np.zeros((self.num_agents, self.AGENT_DIM), dtype=np.float32)
        task = np.zeros((self.num_agents, self.task_dim), dtype=np.float32)
        raw_xy = np.zeros((self.num_agents, 2), dtype=np.float32)
        for k in range(self.num_agents):
            info = infos[k] if k < len(infos) and isinstance(infos[k], dict) else {}
            x, y, z, yaw_deg, health, ammo = ego_pose_vitals(info)
            if self.version >= 2:
                ammo = held_ammo(info)
            yaw = math.radians(yaw_deg)
            alive = float(health > 0.0 and safe_float(info, "DEAD", 0.0) <= 0.0)
            agents[k] = (
                np.clip((x - cx) / hx, -2.0, 2.0),
                np.clip((y - cy) / hy, -2.0, 2.0),
                np.clip(z / self.Z_SCALE, -2.0, 2.0),
                math.cos(yaw),
                math.sin(yaw),
                np.clip(health / 200.0, 0.0, 1.0),
                np.clip(ammo / 200.0, 0.0, 1.0),
                alive,
                0.0,
            )
            raw_xy[k] = (x, y)
            if self.task_keys:
                vals = np.array([safe_float(info, key, 0.0) for key in self.task_keys], dtype=np.float32)
                task[k, : len(vals)] = np.clip(symlog(vals), -self.task_clip, self.task_clip)
        return agents, task, raw_xy

    def _object_features(self, objects) -> tuple[np.ndarray, np.ndarray]:
        """Returns (features (M, object_dim), raw xy (M, 2)) for all objects."""
        if objects is None:
            return np.zeros((0, self.object_dim), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        objects = np.asarray(objects, dtype=np.float32).reshape(-1, 6)
        cx, cy, hx, hy = self._center_half
        feats = np.zeros((len(objects), self.object_dim), dtype=np.float32)
        feats[:, 0] = 1.0
        feats[:, 1] = np.clip((objects[:, 0] - cx) / hx, -2.0, 2.0)
        feats[:, 2] = np.clip((objects[:, 1] - cy) / hy, -2.0, 2.0)
        feats[:, 3] = np.clip(objects[:, 2] / self.Z_SCALE, -2.0, 2.0)
        feats[:, 4] = np.clip(objects[:, 3] / self.VELOCITY_SCALE, -2.0, 2.0)
        feats[:, 5] = np.clip(objects[:, 4] / self.VELOCITY_SCALE, -2.0, 2.0)
        codes = objects[:, 5].astype(np.int64)
        bits = (codes[:, None] >> np.arange(self.type_bits)) & 1
        feats[:, self.OBJECT_BASE_DIM :] = 2.0 * bits - 1.0
        return feats, objects[:, :2]

    def _build(self, obs_list, infos):
        if obs_list is None or any(o is None for o in obs_list):
            return obs_list
        infos = list(infos) if isinstance(infos, (list, tuple)) else [{}] * self.num_agents
        agents, task, agent_xy = self._agent_rows(infos)
        info0 = infos[0] if infos and isinstance(infos[0], dict) else {}
        obj_feats, obj_xy = self._object_features(info0.get("_gs_objects"))
        layout = self._grid.reshape(-1)
        time = np.array([min(1.0, self._step_count / self.horizon)], dtype=np.float32)

        out = []
        for i, obs in enumerate(obs_list):
            order = [i] + [k for k in range(self.num_agents) if k != i]
            own_agents = agents[order].copy()
            own_agents[0, 8] = 1.0
            objects = np.zeros((self.max_objects, self.object_dim), dtype=np.float32)
            if len(obj_feats):
                dist = np.hypot(obj_xy[:, 0] - agent_xy[i, 0], obj_xy[:, 1] - agent_xy[i, 1])
                nearest = np.argsort(dist, kind="stable")[: self.max_objects]
                objects[: len(nearest)] = obj_feats[nearest]
            state = np.concatenate(
                [own_agents.reshape(-1), task[order].reshape(-1), objects.reshape(-1), layout, time]
            ).astype(np.float32, copy=False)

            obs_dict = dict(obs) if isinstance(obs, dict) else {self.obs_key: obs}
            if self._has_input_measurements:
                state = np.concatenate([np.asarray(obs_dict["measurements"], dtype=np.float32), state])
            obs_dict["measurements"] = state
            out.append(obs_dict)
        return out

    @staticmethod
    def _strip(infos) -> None:
        if not isinstance(infos, (list, tuple)):
            return
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in GLOBAL_STATE_INFO_KEYS:
                info.pop(key, None)
            reset_info = info.get("reset_info")
            if isinstance(reset_info, dict):
                for key in GLOBAL_STATE_INFO_KEYS:
                    reset_info.pop(key, None)

    # ---------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self._step_count = 0
        self._reset_layout()
        info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else None
        self._update_layout(info0)
        out = self._build(obs, infos)
        self._strip(infos)
        return out, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        state_infos = infos
        # On an auto-reset step `obs` is already the next episode's first frame while `infos` still describe the
        # terminal frame; build the state from the new episode's `reset_info`.
        if (
            isinstance(infos, (list, tuple))
            and all(bool(t) or bool(tr) for t, tr in zip(terminated, truncated))
            and all(isinstance(i, dict) and isinstance(i.get("reset_info"), dict) for i in infos)
        ):
            state_infos = [i["reset_info"] for i in infos]
            self._step_count = 0
            self._reset_layout()
        elif (
            self.version >= 2
            and isinstance(infos, (list, tuple))
            and infos
            and isinstance(infos[0], dict)
            and infos[0].get("_hidden_reset_count")
        ):
            # crash recovery restarted the game underneath us and `obs` is already one step into the new episode:
            # restart the clock and take the layout from the hidden reset (later infos may omit the lines)
            self._step_count = 1
            self._reset_layout()
            hidden_reset_info = infos[0].get("reset_info")
            if isinstance(hidden_reset_info, dict):
                self._update_layout(hidden_reset_info)
        else:
            self._step_count += 1
        info0 = state_infos[0] if isinstance(state_infos, (list, tuple)) and state_infos else None
        if isinstance(info0, dict) and "_gs_layout_id" in info0:
            self._update_layout(info0)
        out = self._build(obs, state_infos)
        self._strip(infos)
        return out, rewards, terminated, truncated, infos
