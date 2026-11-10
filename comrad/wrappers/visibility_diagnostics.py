from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from comrad.envs.wad_geometry import PLAYER_HEIGHT


# Doom player camera height above the feet in map units
PLAYER_VIEW_HEIGHT = 41.0

_NON_TARGET_OBJECTS = {
    "doomplayer",
    "bulletpuff",
    "blood",
    "teleportfog",
    "playerspawn",
}


@dataclass
class EpisodeVisibilityStats:
    steps: int = 0
    teammate_in_frustum_steps: int = 0
    teammate_visible_steps: int = 0
    teammate_obscured_steps: int = 0
    mutual_visibility_steps: int = 0
    target_visible_steps: int = 0
    target_obscured_steps: int = 0
    mean_teammate_pixel_fraction: float = 0.0
    # Per-viewpoint counters (index = observing agent); the scalar counters above sum over both agents
    agent_in_frustum_steps: list[int] = field(default_factory=lambda: [0, 0])
    agent_visible_steps: list[int] = field(default_factory=lambda: [0, 0])
    agent_obscured_steps: list[int] = field(default_factory=lambda: [0, 0])

    @property
    def agent_steps(self) -> int:
        return max(1, self.steps * 2)

    def per_agent_rates(self, agent_idx: int) -> dict[str, float]:
        steps = float(max(1, self.steps))
        in_frustum = self.agent_in_frustum_steps[agent_idx]
        visible = self.agent_visible_steps[agent_idx]
        obscured = self.agent_obscured_steps[agent_idx]
        return {
            f"agent{agent_idx}_teammate_in_frustum_rate": float(in_frustum) / steps,
            f"agent{agent_idx}_teammate_visible_rate": float(visible) / steps,
            f"agent{agent_idx}_teammate_obscured_rate": float(obscured) / float(in_frustum) if in_frustum > 0 else 0.0,
        }

    @property
    def teammate_in_frustum_rate(self) -> float:
        return float(self.teammate_in_frustum_steps) / float(self.agent_steps)

    @property
    def teammate_visible_rate(self) -> float:
        return float(self.teammate_visible_steps) / float(self.agent_steps)

    @property
    def teammate_obscured_rate(self) -> float:
        if self.teammate_in_frustum_steps <= 0:
            return 0.0
        return float(self.teammate_obscured_steps) / float(self.teammate_in_frustum_steps)

    @property
    def teammate_out_of_view_rate(self) -> float:
        return 1.0 - self.teammate_visible_rate

    @property
    def mutual_visibility_rate(self) -> float:
        return float(self.mutual_visibility_steps) / float(max(1, self.steps))

    @property
    def target_visible_rate(self) -> float:
        return float(self.target_visible_steps) / float(self.agent_steps)

    @property
    def target_obscured_rate(self) -> float:
        denom = self.target_visible_steps + self.target_obscured_steps
        if denom <= 0:
            return 0.0
        return float(self.target_obscured_steps) / float(denom)

    def to_dict(self) -> dict[str, float | int]:
        per_agent = {**self.per_agent_rates(0), **self.per_agent_rates(1)}
        return {
            **per_agent,
            "steps": int(self.steps),
            "teammate_in_frustum_steps": int(self.teammate_in_frustum_steps),
            "teammate_visible_steps": int(self.teammate_visible_steps),
            "teammate_obscured_steps": int(self.teammate_obscured_steps),
            "mutual_visibility_steps": int(self.mutual_visibility_steps),
            "target_visible_steps": int(self.target_visible_steps),
            "target_obscured_steps": int(self.target_obscured_steps),
            "teammate_in_frustum_rate": float(self.teammate_in_frustum_rate),
            "teammate_visible_rate": float(self.teammate_visible_rate),
            "teammate_obscured_rate": float(self.teammate_obscured_rate),
            "teammate_out_of_view_rate": float(self.teammate_out_of_view_rate),
            "mutual_visibility_rate": float(self.mutual_visibility_rate),
            "target_visible_rate": float(self.target_visible_rate),
            "target_obscured_rate": float(self.target_obscured_rate),
            "mean_teammate_pixel_fraction": float(self.mean_teammate_pixel_fraction),
        }


def is_in_frustum(
    observer_pose: tuple[float, float, float, float, float],
    target_xyz: tuple[float, float, float],
    hfov_deg: float = 90.0,
    vfov_deg: float = 72.0,
    min_dist: float = 1.0,
    view_height: float = PLAYER_VIEW_HEIGHT,
    target_height: float = PLAYER_HEIGHT,
) -> bool:
    """
    Check whether a target standing at `target_xyz = (x_j, y_j, z_j)` (feet position) lies inside the 3D perspective
    camera cone of `observer_pose = (x_i, y_i, z_i, yaw_deg, pitch_deg)`.

    `z_i`/`z_j` are feet heights as reported by ViZDoom's POSITION_Z; the camera sits `view_height` above the observer's
    feet and the target occupies `[z_j, z_j + target_height]`. `pitch_deg` follows the ZDoom convention used by the
    PITCH game variable (positive = looking down), so the camera's elevation is `-pitch_deg`. The target counts as
    in-cone when any part of its vertical extent falls inside the vertical field of view.
    """
    x_i, y_i, z_i, yaw_deg, pitch_deg = observer_pose
    x_j, y_j, z_j = target_xyz
    if not (
        math.isfinite(x_i)
        and math.isfinite(y_i)
        and math.isfinite(x_j)
        and math.isfinite(y_j)
        and math.isfinite(yaw_deg)
    ):
        return False

    dx = x_j - x_i
    dy = y_j - y_i
    dz = (z_j - z_i) if (math.isfinite(z_i) and math.isfinite(z_j)) else 0.0
    dist_2d = math.hypot(dx, dy)
    if dist_2d < min_dist and abs(dz) < min_dist:
        return False

    bearing_deg = math.degrees(math.atan2(dy, dx))
    yaw_diff = abs((bearing_deg - yaw_deg + 180.0) % 360.0 - 180.0)
    if yaw_diff > hfov_deg * 0.5:
        return False

    view_elev_deg = -pitch_deg if math.isfinite(pitch_deg) else 0.0
    horiz = max(dist_2d, 1e-6)
    elev_bottom = math.degrees(math.atan2(dz - view_height, horiz))
    elev_top = math.degrees(math.atan2(dz - view_height + target_height, horiz))
    half_v = vfov_deg * 0.5
    return bool(elev_top >= view_elev_deg - half_v and elev_bottom <= view_elev_deg + half_v)


class VisibilityDiagnosticsWrapper(gym.Wrapper):
    """
    Joint MultiAgentEnv wrapper that logs 3D frustum vs. ViZDoom `labels_buffer` visibility/occlusion metrics
    and optionally masks out teammate sprites (`DoomPlayer`) and partner-transmitted cues at evaluation time.
    """

    def __init__(
        self,
        env: gym.Env,
        scenario_name: str = "",
        partner_occlusion_mode: str = "none",
        hfov_deg: float = 90.0,
        vfov_deg: float = 72.0,
        strip_raw_buffers: bool = True,
    ) -> None:
        super().__init__(env)
        if partner_occlusion_mode not in ("none", "sprite", "cues", "both"):
            raise ValueError(f"Unsupported partner_occlusion_mode={partner_occlusion_mode!r}")
        self.scenario_name = str(scenario_name or "")
        self.partner_occlusion_mode = partner_occlusion_mode
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.strip_raw_buffers = bool(strip_raw_buffers)

        self.num_agents = getattr(env, "num_agents", 2)
        self.is_multiagent = getattr(env, "is_multiagent", True)
        self.current_stats = EpisodeVisibilityStats()
        self.last_episode_stats: EpisodeVisibilityStats | None = None
        self._pixel_fraction_sum = 0.0

    def _reset_stats(self) -> None:
        self.current_stats = EpisodeVisibilityStats()
        self._pixel_fraction_sum = 0.0

    @staticmethod
    def _extract_pose(info: dict | None) -> tuple[float, float, float, float, float]:
        if not isinstance(info, dict):
            return (math.nan, math.nan, 0.0, math.nan, 0.0)
        pos = info.get("pos") if isinstance(info.get("pos"), dict) else {}

        def _get(k1: str, k2: str | None, default: float) -> float:
            raw = info.get(k1, pos.get(k2, default) if k2 else default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        x = _get("POSITION_X", "agent_x", math.nan)
        y = _get("POSITION_Y", "agent_y", math.nan)
        z = _get("POSITION_Z", "agent_z", 0.0)
        yaw = _get("ANGLE", "agent_a", math.nan)
        pitch = _get("PITCH", "agent_pitch", 0.0)
        return (x, y, z, yaw, pitch)

    @staticmethod
    def _teammate_mask_and_visibility(
        info_i: dict | None,
        ego_pose: tuple[float, float, float, float, float],
        partner_pose: tuple[float, float, float, float, float],
    ) -> tuple[bool, float, np.ndarray | None]:
        if not isinstance(info_i, dict):
            return False, 0.0, None
        labels = info_i.get("_labels")
        labels_buf = info_i.get("_labels_buffer")
        if not labels:
            return False, 0.0, None

        x_i, y_i, *_ = ego_pose
        x_j, y_j, *_ = partner_pose
        teammate_values = []
        visible_from_bbox = False
        for lb in labels:
            obj_name = str(lb.get("object_name", "")).lower()
            if "player" not in obj_name and "marine" not in obj_name:
                continue
            ox = lb.get("object_position_x", math.nan)
            oy = lb.get("object_position_y", math.nan)
            if math.isfinite(ox) and math.isfinite(oy) and math.isfinite(x_i) and math.isfinite(y_i):
                dist_ego = math.hypot(ox - x_i, oy - y_i)
                if dist_ego < 16.0:
                    continue
                if math.isfinite(x_j) and math.isfinite(y_j):
                    dist_partner = math.hypot(ox - x_j, oy - y_j)
                    if dist_partner > 192.0 and dist_ego < dist_partner:
                        continue
            val = int(lb.get("value", 0))
            if val > 0:
                teammate_values.append(val)
            if int(lb.get("width", 0)) * int(lb.get("height", 0)) > 0:
                visible_from_bbox = True

        if labels_buf is not None and teammate_values:
            mask = np.isin(labels_buf, teammate_values)
            count = int(np.count_nonzero(mask))
            frac = float(count) / float(max(1, labels_buf.size))
            return count > 0, frac, mask

        return visible_from_bbox, 0.0, None

    def _target_visibility(self, agent_idx: int, info_i: dict | None) -> tuple[bool, bool]:
        if not isinstance(info_i, dict):
            return False, False
        name = self.scenario_name
        if "rhythm_sync" in name:
            cue_key = "USER37" if agent_idx == 0 else "USER38"
            vis = float(info_i.get(cue_key, 0.0)) > 0.0
            return vis, not vis
        if "stealth_labyrinth" in name:
            lit = float(info_i.get("USER45", 0.0)) > 0.0
            remaining = float(info_i.get("USER44", 1.0)) > 0.0
            if lit:
                return True, False
            return False, remaining
        if "lava_maze" in name and "USER17" in info_i:
            los = float(info_i.get("USER17", 0.0)) > 0.0
            return los, not los

        labels = info_i.get("_labels")
        labels_buf = info_i.get("_labels_buffer")
        if labels is not None:
            target_vals = [
                int(lb.get("value", 0))
                for lb in labels
                if str(lb.get("object_name", "")).lower() not in _NON_TARGET_OBJECTS and int(lb.get("value", 0)) > 0
            ]
            if labels_buf is not None and target_vals:
                vis = bool(np.any(np.isin(labels_buf, target_vals)))
                return vis, not vis
            if target_vals:
                return True, False
            return False, True
        return False, False

    @staticmethod
    def _mask_sprite_in_image(img: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
        out = img.copy()
        if out.ndim != 3:
            return out
        # Determine whether CHW or HWC
        chw = out.shape[0] in (1, 3, 4, 6) and out.shape[0] < out.shape[1] and out.shape[0] < out.shape[2]
        h, w = (out.shape[1], out.shape[2]) if chw else (out.shape[0], out.shape[1])
        if mask_2d.shape != (h, w):
            resized = cv2.resize(mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            resized = mask_2d.astype(bool)
        if not np.any(resized):
            return out

        bg_mask = ~resized
        if chw:
            # Mask only the ego agent's own 3 RGB channels (0:3)
            c_end = min(3, out.shape[0])
            for c in range(c_end):
                plane = out[c]
                fill_val = np.median(plane[bg_mask]) if np.any(bg_mask) else 0
                plane[resized] = np.asarray(fill_val, dtype=out.dtype)
        else:
            c_end = min(3, out.shape[2])
            for c in range(c_end):
                plane = out[:, :, c]
                fill_val = np.median(plane[bg_mask]) if np.any(bg_mask) else 0
                plane[resized] = np.asarray(fill_val, dtype=out.dtype)
        return out

    def _mask_partner_cues(self, agent_idx: int, obs_i: Any, info_i: dict | None):
        if obs_i is None:
            return None
        name = self.scenario_name
        if isinstance(obs_i, dict):
            out = dict(obs_i)
            if "measurements" in out and out["measurements"] is not None:
                m = np.asarray(out["measurements"], dtype=np.float32).copy()
                if "rhythm_sync" in name and m.shape[0] >= 7:
                    # Zero cue visibility and cue owner / partner pending flags
                    m[3] = 0.0
                    m[4] = 0.0
                    if m.shape[0] >= 9:
                        m[5] = 0.0
                        m[6] = 0.0
                elif "ammo_carrier" in name and agent_idx == 1 and m.shape[0] >= 6:
                    # Mask defender ammo / shortage / low-alert signals from runner
                    m[3:6] = 0.0
                elif "lava_maze" in name and agent_idx == 0 and m.shape[0] >= 2:
                    # Mask oracle/partner hint channels if present
                    m[:] = 0.0
                out["measurements"] = m
            if "lava_maze" in name and agent_idx == 0 and "obs" in out and isinstance(info_i, dict):
                # When observer fires colour flash (USER16 > 0), neutralize chroma difference across RGB channels
                if float(info_i.get("USER16", 0.0)) > 0.0 and out["obs"] is not None and out["obs"].ndim == 3:
                    img = out["obs"].copy()
                    if img.shape[0] in (3, 6):
                        gray = np.mean(img[:3], axis=0, keepdims=True).astype(img.dtype)
                        img[:3] = np.repeat(gray, 3, axis=0)
                    out["obs"] = img
            return out
        return obs_i

    def _process(self, obs_list, infos, is_step: bool = True, dones: list[bool] | None = None):
        if obs_list is None or not isinstance(infos, (list, tuple)) or len(infos) < 2:
            return obs_list, infos

        poses = [self._extract_pose(infos[i]) for i in range(2)]
        # On an auto-reset step the returned obs already belong to the next episode while `infos` describe the terminal
        # frame: telemetry is finalized from `infos`, but occlusion masks for the new frame come from `reset_info`.
        obs_infos = None
        if (
            is_step
            and dones is not None
            and all(dones)
            and all(isinstance(infos[k], dict) and isinstance(infos[k].get("reset_info"), dict) for k in range(2))
        ):
            obs_infos = [infos[k]["reset_info"] for k in range(2)]
            obs_poses = [self._extract_pose(obs_infos[k]) for k in range(2)]
        vis_flags = [False, False]
        in_frustum_flags = [False, False]
        obscured_flags = [False, False]
        out_obs = list(obs_list)

        for i in range(2):
            j = 1 - i
            p_i = poses[i]
            p_j = poses[j]
            in_fov = is_in_frustum(p_i, (p_j[0], p_j[1], p_j[2]), hfov_deg=self.hfov_deg, vfov_deg=self.vfov_deg)
            vis, pix_frac, mask = self._teammate_mask_and_visibility(infos[i], p_i, p_j)
            # In scenarios with disjoint rooms/lanes (e.g., rhythm_sync, coop_puzzle), if labels_buffer is not enabled,
            # line-of-sight is blocked by solid walls even when facing the partner's coordinates.
            if "_labels" not in (infos[i] or {}):
                if any(k in self.scenario_name for k in ("rhythm_sync", "coop_puzzle")):
                    vis = False
                elif "stealth_labyrinth" in self.scenario_name and float((infos[i] or {}).get("USER45", 0.0)) <= 0.0:
                    vis = False

            # Any teammate sprite with rendered pixels on the viewport is inside the effective camera frustum
            in_fov = bool(in_fov or vis)
            obscured = bool(in_fov and not vis)
            in_frustum_flags[i] = in_fov
            vis_flags[i] = vis
            obscured_flags[i] = obscured
            target_vis, target_obs = self._target_visibility(i, infos[i])

            if is_step:
                if in_fov:
                    self.current_stats.teammate_in_frustum_steps += 1
                    self.current_stats.agent_in_frustum_steps[i] += 1
                if vis:
                    self.current_stats.teammate_visible_steps += 1
                    self.current_stats.agent_visible_steps[i] += 1
                if obscured:
                    self.current_stats.teammate_obscured_steps += 1
                    self.current_stats.agent_obscured_steps[i] += 1
                if target_vis:
                    self.current_stats.target_visible_steps += 1
                if target_obs:
                    self.current_stats.target_obscured_steps += 1
                self._pixel_fraction_sum += pix_frac

            info_for_obs = infos[i]
            if obs_infos is not None:
                info_for_obs = obs_infos[i]
                if self.partner_occlusion_mode in ("sprite", "both"):
                    _, _, mask = self._teammate_mask_and_visibility(info_for_obs, obs_poses[i], obs_poses[j])

            if self.partner_occlusion_mode in ("sprite", "both") and mask is not None and out_obs[i] is not None:
                if isinstance(out_obs[i], dict) and "obs" in out_obs[i]:
                    d = dict(out_obs[i])
                    d["obs"] = self._mask_sprite_in_image(np.asarray(d["obs"]), mask)
                    out_obs[i] = d
                elif isinstance(out_obs[i], np.ndarray):
                    out_obs[i] = self._mask_sprite_in_image(out_obs[i], mask)

            if self.partner_occlusion_mode in ("cues", "both"):
                out_obs[i] = self._mask_partner_cues(i, out_obs[i], info_for_obs)

            if isinstance(infos[i], dict):
                infos[i]["visibility"] = {
                    "teammate_in_frustum": bool(in_fov),
                    "teammate_visible": bool(vis),
                    "teammate_obscured": bool(obscured),
                    "target_visible": bool(target_vis),
                    "target_obscured": bool(target_obs),
                    "teammate_pixel_fraction": float(pix_frac),
                }
                if self.strip_raw_buffers:
                    infos[i].pop("_labels_buffer", None)
                    infos[i].pop("_labels", None)

        if is_step:
            self.current_stats.steps += 1
            if vis_flags[0] and vis_flags[1]:
                self.current_stats.mutual_visibility_steps += 1
            self.current_stats.mean_teammate_pixel_fraction = self._pixel_fraction_sum / float(
                self.current_stats.agent_steps
            )

        if is_step and dones is not None and any(dones):
            self.last_episode_stats = self.current_stats
            summary = self.last_episode_stats.to_dict()
            for i in range(2):
                if isinstance(infos[i], dict):
                    infos[i]["episode_visibility_stats"] = summary
            if all(dones):
                self._reset_stats()

        return out_obs, infos

    def reset(self, **kwargs):
        self._reset_stats()
        obs, infos = self.env.reset(**kwargs)
        return self._process(obs, infos, is_step=False)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = [bool(t) or bool(tr) for t, tr in zip(terminated, truncated)]
        obs, infos = self._process(obs, infos, is_step=True, dones=dones)
        return obs, rewards, terminated, truncated, infos
