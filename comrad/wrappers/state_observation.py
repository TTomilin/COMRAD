from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


def safe_float(info: dict | None, key: str, default: float = 0.0) -> float:
    if not isinstance(info, dict):
        return default
    val = info.get(key, default)
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def ego_pose_vitals(info: dict | None) -> tuple[float, float, float, float, float, float]:
    """Raw `(x, y, z, yaw_deg, health, ammo)` of one agent from its env `info`."""
    pos = info.get("pos") if isinstance(info, dict) and isinstance(info.get("pos"), dict) else {}
    x = safe_float(info, "POSITION_X", safe_float(pos, "agent_x", 0.0))
    y = safe_float(info, "POSITION_Y", safe_float(pos, "agent_y", 0.0))
    z = safe_float(info, "POSITION_Z", safe_float(pos, "agent_z", 0.0))
    yaw_deg = safe_float(info, "ANGLE", safe_float(pos, "agent_a", 0.0))
    health = max(0.0, safe_float(info, "HEALTH", 100.0))
    ammo = max(
        0.0,
        safe_float(info, "AMMO1", safe_float(info, "AMMO2", safe_float(info, "SELECTED_WEAPON_AMMO", 0.0))),
    )
    return x, y, z, yaw_deg, health, ammo


TASK_EXTRA_DIM = 10
STATE_ENCODINGS = ("ego", "entity")
# 5 ego + 6 partner + 12 x (4 + 16) entities + 16 rays + 10 task values (see EntityStateWrapper)
ENTITY_STATE_DIM = 277


def rhythm_sync_target_keys(pid: int) -> tuple[str, str]:
    return ("USER33", "USER34") if pid == 0 else ("USER35", "USER36")


def task_extra_features(
    info: dict | None, scenario_name: str, pid: int, include_global_state: bool = False
) -> np.ndarray:
    """10D scenario task variables shared by the `ego` and `entity` state encodings (slot 0 is the role id)."""
    vec = np.zeros(TASK_EXTRA_DIM, dtype=np.float32)
    vec[0] = float(pid)
    if not isinstance(info, dict):
        return vec

    name = str(scenario_name or "")
    if "stealth_labyrinth" in name:
        total = max(1.0, safe_float(info, "USER47", 1.0))
        vec[1] = np.clip(safe_float(info, "USER41", 1.0), 0.0, 1.0)
        vec[2] = np.clip(safe_float(info, "USER42", 1.0), 0.0, 1.0)
        vec[3] = np.clip(safe_float(info, "USER44", 0.0) / total, 0.0, 1.0)
        vec[4] = np.clip(safe_float(info, "USER45", 0.0), 0.0, 1.0)
        vec[5] = np.clip(safe_float(info, "USER46", 0.0) / total, 0.0, 1.0)
        vec[6] = np.clip(safe_float(info, "USER48", 0.0) / 10.0, 0.0, 2.0)
        vec[7] = np.clip(safe_float(info, "USER50", 0.0) / 200.0, 0.0, 2.0)
        vec[8] = np.clip(safe_float(info, "USER51", 0.0) / 10.0, 0.0, 2.0)
        vec[9] = np.clip(safe_float(info, "KILLCOUNT", 0.0) / 20.0, 0.0, 2.0)
    elif "rhythm_sync" in name:
        num_sections = max(1.0, safe_float(info, "USER32", 1.0))
        vec[1] = np.clip(safe_float(info, "USER24", 0.0) / num_sections, 0.0, 1.0)
        vec[2] = np.clip(safe_float(info, "USER25", 0.0) / num_sections, 0.0, 1.0)
        vec[3] = np.clip(safe_float(info, "USER26", 0.0) / 2.0, 0.0, 1.0)
        cue_key = "USER37" if pid == 0 else "USER38"
        vec[4] = float(safe_float(info, cue_key, 0.0) > 0.0)
        pend_self = "USER27" if pid == 0 else "USER28"
        vec[5] = float(safe_float(info, pend_self, 0.0) > 0.0)
        if include_global_state:
            pend_other = "USER28" if pid == 0 else "USER27"
            vec[6] = float(safe_float(info, pend_other, 0.0) > 0.0)
        vec[7] = np.clip(safe_float(info, "USER31", -1.0), -1.0, 1.0)
        tx_key, ty_key = rhythm_sync_target_keys(pid)
        vec[8] = np.clip((safe_float(info, tx_key, 0.0) - safe_float(info, "POSITION_X", 0.0)) / 768.0, -2.0, 2.0)
        vec[9] = np.clip((safe_float(info, ty_key, 0.0) - safe_float(info, "POSITION_Y", 0.0)) / 768.0, -2.0, 2.0)
    elif "armory_siege" in name:
        vec[1] = np.clip(safe_float(info, "USER1", 100.0) / 500.0, 0.0, 2.0)
        vec[2] = np.clip(safe_float(info, "AMMO1", 0.0) / 200.0, 0.0, 1.0)
        vec[3] = np.clip(safe_float(info, "AMMO2", 0.0) / 100.0, 0.0, 1.0)
        vec[4] = float(safe_float(info, "WEAPON1", 0.0) > 0.0)
        vec[5] = float(safe_float(info, "WEAPON2", 0.0) > 0.0)
        vec[6] = np.clip(safe_float(info, "SELECTED_WEAPON", 0.0) / 3.0, 0.0, 1.0)
        vec[7] = np.clip(safe_float(info, "KILLCOUNT", 0.0) / 100.0, 0.0, 5.0)
        vec[8] = np.clip(safe_float(info, "HITCOUNT", 0.0) / 200.0, 0.0, 5.0)
        vec[9] = float(safe_float(info, "ATTACK_READY", 0.0) > 0.0)
    elif "ammo_carrier" in name:
        vec[1] = np.clip(safe_float(info, "USER41", 0.0) / 200.0, 0.0, 1.0)
        vec[2] = np.clip(safe_float(info, "USER42", 0.0) / 20.0, 0.0, 2.0)
        vec[3] = np.clip(safe_float(info, "USER43", 0.0) / 32.0, 0.0, 2.0)
        vec[4] = np.clip(safe_float(info, "USER44", 0.0) / 32.0, 0.0, 2.0)
        vec[5] = float(safe_float(info, "USER45", 0.0) > 0.0)
        vec[6] = np.clip(safe_float(info, "KILLCOUNT", 0.0) / 50.0, 0.0, 5.0)
        vec[7] = np.clip(safe_float(info, "DAMAGECOUNT", 0.0) / 500.0, 0.0, 5.0)
    else:
        vec[1] = np.clip(safe_float(info, "KILLCOUNT", 0.0) / 50.0, 0.0, 5.0)
        vec[2] = np.clip(safe_float(info, "HITCOUNT", 0.0) / 100.0, 0.0, 5.0)
        vec[3] = np.clip(safe_float(info, "DAMAGECOUNT", 0.0) / 500.0, 0.0, 5.0)
        user_keys = sorted(k for k in info.keys() if k.startswith("USER"))
        if "coop_puzzle" in name:
            # USER7+ (puzzle state for the global-state reference) are not part of this partial task vector
            user_keys = [k for k in user_keys if k in {f"USER{j}" for j in range(1, 7)}]
        user_keys = user_keys[:6]
        for idx, uk in enumerate(user_keys):
            vec[4 + idx] = np.clip(safe_float(info, uk, 0.0) / 100.0, -5.0, 5.0)

    return vec


class StateObservationWrapper(gym.Wrapper):
    """
    Replaces 3D RGB `obs_space['obs']` with a 1D `float32` compact state vector for MLP encoding.

    The state vector combines:
      1. 8D ego pose & vitals:
         `[x / 2048, y / 2048, z / 512, cos(yaw), sin(yaw), health / 200, ammo / 200, step / horizon]`
      2. Existing scenario `measurements` vector (if present on the wrapped env), scaled to roughly [-1, 1].
      3. 10D scenario-specific local/task game variables extracted from `info` (such as role id, switch/room/enemy
         progress counters, and local cue indicators), removing the separate `"measurements"` key so the entire
         compact state vector is encoded once by `MlpEncoder`.

    With `state_encoding="entity"` the wrapper only declares the `ENTITY_STATE_DIM` space and emits zeros; the joint
    `EntityStateWrapper` fills the vector, since it needs the partner's and the host's `info`.
    """

    EGO_DIM = 8
    TASK_EXTRA_DIM = TASK_EXTRA_DIM

    def __init__(
        self,
        env: gym.Env,
        scenario_name: str = "",
        include_global_state: bool = False,
        horizon: int = 4500,
        state_encoding: str = "ego",
    ) -> None:
        super().__init__(env)
        self.scenario_name = str(scenario_name or "")
        self.include_global_state = bool(include_global_state)
        self.horizon = max(1, int(horizon))
        self._step_count = 0
        self.state_encoding = str(state_encoding or "ego").lower()
        if self.state_encoding not in STATE_ENCODINGS:
            raise ValueError(f"Unknown state_encoding {state_encoding!r}, expected one of {STATE_ENCODINGS}")

        orig_space = env.observation_space
        self._input_is_dict = isinstance(orig_space, gym.spaces.Dict)
        self._has_measurements = self._input_is_dict and "measurements" in orig_space.spaces
        if self._has_measurements:
            m_space = orig_space["measurements"]
            self._meas_dim = int(m_space.shape[0])
            self._meas_scale = np.maximum(np.abs(m_space.high), 1.0).astype(np.float32)
        else:
            self._meas_dim = 0
            self._meas_scale = np.ones((0,), dtype=np.float32)

        if self.state_encoding == "entity":
            self.state_dim = ENTITY_STATE_DIM
        else:
            self.state_dim = self.EGO_DIM + self._meas_dim + self.TASK_EXTRA_DIM
        state_box = gym.spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        spaces = dict(orig_space.spaces) if self._input_is_dict else {}
        spaces.pop("measurements", None)
        spaces["obs"] = state_box
        self.observation_space = gym.spaces.Dict(spaces)

    def _player_id(self) -> int:
        pid = getattr(self.env.unwrapped, "player_id", 0)
        try:
            return max(0, int(pid))
        except (TypeError, ValueError):
            return 0

    def _ego_features(self, info: dict | None) -> np.ndarray:
        x, y, z, yaw_deg, health, ammo = ego_pose_vitals(info)
        yaw_rad = math.radians(yaw_deg)
        progress = min(1.0, float(self._step_count) / float(self.horizon))

        return np.array(
            [
                np.clip(x / 2048.0, -2.0, 2.0),
                np.clip(y / 2048.0, -2.0, 2.0),
                np.clip(z / 512.0, -2.0, 2.0),
                math.cos(yaw_rad),
                math.sin(yaw_rad),
                np.clip(health / 200.0, 0.0, 1.0),
                np.clip(ammo / 200.0, 0.0, 1.0),
                progress,
            ],
            dtype=np.float32,
        )

    def _task_extra_features(self, info: dict | None) -> np.ndarray:
        return task_extra_features(info, self.scenario_name, self._player_id(), self.include_global_state)

    def _build_obs(self, obs, info: dict | None):
        if obs is None:
            return None
        if self.state_encoding == "entity":
            out = dict(obs) if isinstance(obs, dict) else {}
            out.pop("measurements", None)
            out["obs"] = np.zeros(self.state_dim, dtype=np.float32)
            return out
        ego = self._ego_features(info)
        parts = [ego]
        if self._has_measurements and isinstance(obs, dict) and "measurements" in obs:
            raw_m = np.asarray(obs["measurements"], dtype=np.float32)
            scaled_m = np.clip(raw_m / self._meas_scale, -5.0, 5.0)
            parts.append(scaled_m)
        elif self._meas_dim > 0:
            parts.append(np.zeros(self._meas_dim, dtype=np.float32))
        parts.append(self._task_extra_features(info))

        state_vec = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        out = dict(obs) if isinstance(obs, dict) else {}
        out.pop("measurements", None)
        out["obs"] = state_vec
        return out

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        if not info and hasattr(self.env.unwrapped, "get_info"):
            info = self.env.unwrapped.get_info()
        return self._build_obs(obs, info), info

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._build_obs(obs, info), reward, terminated, truncated, info


class BlindObservationWrapper(gym.Wrapper):
    """
    Zeroes out 3D RGB frames (`obs = 0`) and injects the normalized timestep `t / H` into the first pixel
    to test SMACv2-style blind / open-loop temporal memorization policies while keeping the observation space
    shape and dtype unchanged for both zero-shot checkpoint evaluation and blind training from scratch.
    """

    def __init__(self, env: gym.Env, horizon: int = 4500) -> None:
        super().__init__(env)
        self.horizon = max(1, int(horizon))
        self._step_count = 0

    def _blind_frame(self, img: np.ndarray) -> np.ndarray:
        zeroed = np.zeros_like(img)
        progress = min(1.0, float(self._step_count) / float(self.horizon))
        if zeroed.size > 0:
            if np.issubdtype(zeroed.dtype, np.integer):
                zeroed.flat[0] = int(round(progress * 255.0))
            else:
                zeroed.flat[0] = float(progress)
        return zeroed

    def _transform(self, obs):
        if obs is None:
            return None
        if isinstance(obs, dict):
            out = dict(obs)
            if "obs" in out and out["obs"] is not None:
                out["obs"] = self._blind_frame(np.asarray(out["obs"]))
            return out
        return self._blind_frame(np.asarray(obs))

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        return self._transform(obs), info

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info
