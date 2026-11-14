from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from comrad.envs.doom_gym import DEPTH_ROW_INFO_KEY, GLOBAL_STATE_INFO_KEYS
from comrad.wrappers.episode_boundary import joint_new_episode
from comrad.wrappers.state_observation import (
    ENTITY_STATE_DIM,
    TASK_EXTRA_DIM,
    clock_horizon,
    ego_pose_vitals,
    rhythm_sync_target_keys,
    safe_float,
    task_extra_features,
)

ENTITY_INFO_KEYS = GLOBAL_STATE_INFO_KEYS + (DEPTH_ROW_INFO_KEY,)

# Scenarios whose task block is the shared task extractor plus the radius gating of EntityStateWrapper._task.
ENTITY_SHARED_TASK_SCENARIOS = frozenset(
    {"armory_siege", "ammo_carrier", "rhythm_sync", "rhythm_sync_dense", "stealth_labyrinth", "stealth_labyrinth_lit",
     "stealth_labyrinth_shared_vision"}
)
_LAVA_MAZE_PUBLIC = ("USER11", "USER16", "USER17", "USER22")  # grid size, broadcast colour cue and flash, levels
_COOP_PUZZLE_PARTIAL = ("USER1", "USER2", "USER3", "USER4", "USER5", "USER6")
# For every other scenario, per role: the USER variables (task slots 4-9, / 100) after the role id and the agent's own
# KILLCOUNT / HITCOUNT / DAMAGECOUNT. Only own statistics, team progress and cues shown to that role are listed: goals,
# positions, global enemy counters and partner-private variables never enter the partial observation.
ENTITY_USER_FEATURES = {
    "lavapit": ((), ()),
    "platform_chain": ((), ()),
    "platform_chain_easy": ((), ()),
    "coop_puzzle": (_COOP_PUZZLE_PARTIAL, _COOP_PUZZLE_PARTIAL),
    "lava_maze": (_LAVA_MAZE_PUBLIC, _LAVA_MAZE_PUBLIC),
    "lava_maze_simple": (_LAVA_MAZE_PUBLIC, _LAVA_MAZE_PUBLIC),
    "lava_maze_oracle_msg": (_LAVA_MAZE_PUBLIC, _LAVA_MAZE_PUBLIC),
    "lava_maze_oracle_hint": (_LAVA_MAZE_PUBLIC, _LAVA_MAZE_PUBLIC),
    "smart_enemies": ((), ()),
    "dumb_enemies": ((), ()),
    "stag_hunt_arena": (("USER56",), ("USER57",)),
    "stag_hunt_arena_continuous": (("USER56",), ("USER57",)),
    "stag_hunt_arena_continuous_full": (("USER56",), ("USER57",)),
    "foraging_commons": (("USER4", "USER8"), ("USER5", "USER9")),
    "coop_health_gathering": ((), ()),
}


def has_entity_task_schema(scenario_name: str) -> bool:
    return scenario_name in ENTITY_SHARED_TASK_SCENARIOS or scenario_name in ENTITY_USER_FEATURES


class EntityStateWrapper(gym.Wrapper):
    """
    SMAC-style partially observable state vector for a MultiAgentEnv (`--state_encoding=entity`).

    Fills `obs['obs']` of every agent (declared by the per-agent `StateObservationWrapper` placeholder) with:
      ego       5                        cos yaw, sin yaw, health / 200, ammo / 200, step / horizon
      partner   6                        in range, forward, left, distance, cos and sin of relative heading
      entities  max_entities x (4 + type_bits)
                                         present, forward, left, distance, +-1 bits of the 16-bit actor name hash;
                                         the nearest non-player objects within the sight radius
      rays      num_rays                 min depth per equal slice of the centre depth-buffer row, / 255
      task      10                       scenario task variables; partner and target variables are radius-gated
    Offsets are relative to the agent, rotated into its heading frame (forward, left) and divided by `sight_range`.
    The sight radius applies in every direction and through walls; everything outside it is zero.
    Objects come from agent 0's `info` (`_gs_objects`), poses and depth rows from each agent's own `info`.
    """

    EGO_DIM = 5
    PARTNER_DIM = 6
    ENTITY_BASE_DIM = 4

    def __init__(
        self,
        env: gym.Env,
        scenario_name: str,
        horizon: float,
        sight_range: float = 512.0,
        max_entities: int = 12,
        num_rays: int = 16,
        type_bits: int = 16,
        obs_key: str = "obs",
    ) -> None:
        super().__init__(env)
        self.num_agents = int(getattr(env, "num_agents", 1))
        if self.num_agents < 2:
            raise ValueError("EntityStateWrapper (--state_encoding=entity) needs a multi-agent env with >= 2 agents")
        self.is_multiagent = True
        self.scenario_name = str(scenario_name or "")
        if not has_entity_task_schema(self.scenario_name):
            raise ValueError(f"No entity-state task schema for scenario {self.scenario_name!r} (see ENTITY_USER_FEATURES)")
        self._user_features = ENTITY_USER_FEATURES.get(self.scenario_name)
        if self._user_features is not None and len(self._user_features) < self.num_agents:
            raise ValueError(f"Entity-state task schema of {self.scenario_name!r} covers {len(self._user_features)} roles, "
                             f"the env has {self.num_agents} agents")
        self.horizon = clock_horizon(horizon)
        self.sight_range = float(sight_range)
        if self.sight_range <= 0:
            raise ValueError(f"sight_range must be positive, got {sight_range}")
        self.max_entities = int(max_entities)
        self.num_rays = int(num_rays)
        self.type_bits = int(type_bits)
        self.obs_key = obs_key
        self.entity_dim = self.ENTITY_BASE_DIM + self.type_bits
        self.state_dim = (
            self.EGO_DIM + self.PARTNER_DIM + self.max_entities * self.entity_dim + self.num_rays + TASK_EXTRA_DIM
        )

        space = env.observation_space
        declared = space[obs_key].shape if isinstance(space, gym.spaces.Dict) and obs_key in space.spaces else None
        if declared != (self.state_dim,):
            raise ValueError(
                f"EntityStateWrapper builds {self.state_dim} values but the per-agent space declares {declared}; "
                f"wrap agents with StateObservationWrapper(state_encoding='entity') (ENTITY_STATE_DIM={ENTITY_STATE_DIM})"
            )
        self._step_count = 0

    # ---------------------------------------------------------------- helpers
    def _to_frame(self, dx, dy, yaw_rad: float):
        """World offsets -> (forward, left) in the agent's heading frame."""
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        return dx * c + dy * s, -dx * s + dy * c

    def _ego(self, health: float, ammo: float, yaw_rad: float) -> np.ndarray:
        return np.array(
            [
                math.cos(yaw_rad),
                math.sin(yaw_rad),
                np.clip(health / 200.0, 0.0, 1.0),
                np.clip(ammo / 200.0, 0.0, 1.0),
                min(1.0, self._step_count / self.horizon),
            ],
            dtype=np.float32,
        )

    def _partner(self, i: int, poses) -> tuple[np.ndarray, bool]:
        x, y, yaw = poses[i]
        others = [k for k in range(self.num_agents) if k != i]
        dists = [math.hypot(poses[k][0] - x, poses[k][1] - y) for k in others]
        k = others[int(np.argmin(dists))]
        dist = min(dists)
        block = np.zeros(self.PARTNER_DIM, dtype=np.float32)
        if dist > self.sight_range:
            return block, False
        fwd, left = self._to_frame(poses[k][0] - x, poses[k][1] - y, yaw)
        rel_yaw = poses[k][2] - yaw
        block[:] = (1.0, fwd / self.sight_range, left / self.sight_range, dist / self.sight_range,
                    math.cos(rel_yaw), math.sin(rel_yaw))
        return block, True

    def _entities(self, objects, x: float, y: float, yaw: float) -> np.ndarray:
        block = np.zeros((self.max_entities, self.entity_dim), dtype=np.float32)
        if objects is None:
            return block.reshape(-1)
        objects = np.asarray(objects, dtype=np.float32).reshape(-1, 6)
        if len(objects) == 0:
            return block.reshape(-1)
        dx, dy = objects[:, 0] - x, objects[:, 1] - y
        dist = np.hypot(dx, dy)
        idx = np.flatnonzero(dist <= self.sight_range)
        idx = idx[np.argsort(dist[idx], kind="stable")][: self.max_entities]
        if len(idx) == 0:
            return block.reshape(-1)
        fwd, left = self._to_frame(dx[idx], dy[idx], yaw)
        n = len(idx)
        block[:n, 0] = 1.0
        block[:n, 1] = fwd / self.sight_range
        block[:n, 2] = left / self.sight_range
        block[:n, 3] = dist[idx] / self.sight_range
        codes = objects[idx, 5].astype(np.int64)
        bits = (codes[:, None] >> np.arange(self.type_bits)) & 1
        block[:n, self.ENTITY_BASE_DIM :] = 2.0 * bits - 1.0
        return block.reshape(-1)

    def _rays(self, depth_row) -> np.ndarray:
        if depth_row is None:
            return np.zeros(self.num_rays, dtype=np.float32)
        row = np.asarray(depth_row, dtype=np.float32).reshape(-1)
        if row.size < self.num_rays:
            return np.zeros(self.num_rays, dtype=np.float32)
        return np.array([s.min() for s in np.array_split(row, self.num_rays)], dtype=np.float32) / 255.0

    def _own_task(self, i: int, info: dict) -> np.ndarray:
        vec = np.zeros(TASK_EXTRA_DIM, dtype=np.float32)
        vec[0] = float(i)
        vec[1] = np.clip(safe_float(info, "KILLCOUNT", 0.0) / 50.0, 0.0, 5.0)
        vec[2] = np.clip(safe_float(info, "HITCOUNT", 0.0) / 100.0, 0.0, 5.0)
        vec[3] = np.clip(safe_float(info, "DAMAGECOUNT", 0.0) / 500.0, 0.0, 5.0)
        for slot, key in enumerate(self._user_features[i], start=4):
            vec[slot] = np.clip(safe_float(info, key, 0.0) / 100.0, -5.0, 5.0)
        return vec

    def _task(self, i: int, info: dict, x: float, y: float, yaw: float, partner_in_range: bool) -> np.ndarray:
        if self._user_features is not None:
            return self._own_task(i, info)
        vec = task_extra_features(info, self.scenario_name, i, include_global_state=False)
        name = self.scenario_name
        if "rhythm_sync" in name:
            # own target: rotated offset inside the sight radius, otherwise hidden
            vec[8:10] = 0.0
            tx_key, ty_key = rhythm_sync_target_keys(i)
            if tx_key in info and ty_key in info:
                dx, dy = safe_float(info, tx_key) - x, safe_float(info, ty_key) - y
                if math.hypot(dx, dy) <= self.sight_range:
                    fwd, left = self._to_frame(dx, dy, yaw)
                    vec[8], vec[9] = fwd / self.sight_range, left / self.sight_range
        elif "ammo_carrier" in name:
            vec[4] = 0.0  # global enemy count
            if i != 0 and not partner_in_range:
                vec[1] = 0.0  # shooter's (player 0) ammo, visible to the runner only within the radius
        return vec

    # ---------------------------------------------------------------- build
    def _build(self, obs_list, infos):
        if obs_list is None or any(o is None for o in obs_list):
            return obs_list
        infos = list(infos) if isinstance(infos, (list, tuple)) else [{}] * self.num_agents
        infos = [infos[k] if k < len(infos) and isinstance(infos[k], dict) else {} for k in range(self.num_agents)]
        vitals = [ego_pose_vitals(info) for info in infos]
        poses = [(v[0], v[1], math.radians(v[3])) for v in vitals]
        objects = infos[0].get("_gs_objects")

        out = []
        for i, obs in enumerate(obs_list):
            x, y, yaw = poses[i]
            partner, in_range = self._partner(i, poses)
            state = np.concatenate(
                [
                    self._ego(vitals[i][4], vitals[i][5], yaw),
                    partner,
                    self._entities(objects, x, y, yaw),
                    self._rays(infos[i].get(DEPTH_ROW_INFO_KEY)),
                    self._task(i, infos[i], x, y, yaw, in_range),
                ]
            ).astype(np.float32, copy=False)
            state = np.clip(state, -5.0, 5.0)
            obs_dict = dict(obs) if isinstance(obs, dict) else {self.obs_key: obs}
            obs_dict[self.obs_key] = state
            out.append(obs_dict)
        return out

    @staticmethod
    def _strip(infos) -> None:
        if not isinstance(infos, (list, tuple)):
            return
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in ENTITY_INFO_KEYS:
                info.pop(key, None)
            reset_info = info.get("reset_info")
            if isinstance(reset_info, dict):
                for key in ENTITY_INFO_KEYS:
                    reset_info.pop(key, None)

    # ---------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self._step_count = 0
        out = self._build(obs, infos)
        self._strip(infos)
        return out, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        state_infos = infos
        new_episode = joint_new_episode(terminated, truncated, infos)
        if new_episode is not None:
            state_infos = new_episode.infos
            self._step_count = new_episode.steps
        else:
            self._step_count += 1
        out = self._build(obs, state_infos)
        self._strip(infos)
        return out, rewards, terminated, truncated, infos
