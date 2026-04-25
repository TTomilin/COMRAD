from __future__ import annotations

import json
import math
import os
import uuid
from typing import Sequence

import gymnasium as gym
import numpy as np

def _as_bool_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return None
        return value.tolist()
    return None

def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.dtype == np.uint8: return arr

    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        max = np.nanmax(arr) if arr.size > 0 else 1.0
        if max <= 1.0:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _tile_ahwc(frames: Sequence[np.ndarray]) -> np.ndarray:
    if len(frames) == 1: return frames[0]

    a = len(frames)
    h, w, c = frames[0].shape
    cols = int(math.ceil(math.sqrt(a)))
    rows = int(math.ceil(a / cols))
    canvas = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for i, f in enumerate(frames):
        r, cid = divmod(i, cols)
        canvas[r * h : (r + 1) * h, cid * w : (cid + 1) * w] = f
    return canvas

def _select_image(obs) -> np.ndarray | None:
    if obs is None: return None
    if isinstance(obs, np.ndarray): return obs
    if isinstance(obs, dict): # Check additional_input.py
        if "obs" in obs and isinstance(obs["obs"], np.ndarray):
            return obs["obs"]
        for i in obs.values():
            o = _select_image(i)
            if o is not None: return o
    return None

#==================================================
# Those are for logging metadata only

def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return None

def _info_scalars(info):
    if not isinstance(info, dict):
        return {}

    scalars = {}
    for key, value in info.items():
        if key in ("episode_extra_stats", "reset_info"):
            continue
        scalar = _json_scalar(value)
        if scalar is not None:
            scalars[key] = scalar
    return scalars

def _multi_info_scalars(infos):
    if not isinstance(infos, (list, tuple)):
        return {}

    keys = set()
    for item in infos:
        if isinstance(item, dict):
            keys.update(item.keys())

    scalars = {}
    for key in sorted(keys):
        if key in ("episode_extra_stats", "reset_info"):
            continue

        values = []
        valid = True
        for item in infos:
            value = item.get(key) if isinstance(item, dict) else None
            scalar = _json_scalar(value)
            if value is not None and scalar is None:
                valid = False
                break
            values.append(scalar)

        if valid:
            scalars[f"agent_{key}"] = values

    return scalars


def _hidden_reset_count(info) -> int:
    if isinstance(info, list):
        counts = []
        for item in info:
            if isinstance(item, dict):
                try:
                    counts.append(int(item.get("_hidden_reset_count", 0)))
                except (TypeError, ValueError):
                    continue
        return max(counts, default=0)

    if isinstance(info, dict):
        try:
            return int(info.get("_hidden_reset_count", 0))
        except (TypeError, ValueError):
            return 0

    return 0

#==================================================


class VideoLoggerWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        record_every: int = 0,
        fps: int = 35,
        max_frames: int = 1000,
        is_multi: bool = False,
        done_mode: str = "all",
        output_dir: str | None = None,
    ):
        super().__init__(env)
        self.every_n = max(0, int(record_every))
        self.fps = int(fps)
        self.max_frames = int(max_frames)
        self.is_multi = is_multi
        if done_mode not in ("all", "any"):
            raise ValueError(f"done_mode must be 'all' or 'any', not {done_mode!r}")
        self.done_mode = done_mode

        self.out_dir = os.fspath(output_dir) if output_dir is not None else None
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        self._ep_idx = 0
        self._recording = False
        self._frames: list[np.ndarray] = [] # list of tiled HWC uint8 frames

    # def _env_indices(self):
    #     unwrapped = self.env.unwrapped
    #     worker_index = getattr(unwrapped, "worker_index", None)
    #     vector_index = getattr(unwrapped, "vector_index", None)
    #
    #     if (worker_index is None or vector_index is None) and hasattr(unwrapped, "env_config"):
    #         env_config = getattr(unwrapped, "env_config")
    #         if worker_index is None and env_config is not None and "worker_index" in env_config:
    #             worker_index = env_config.worker_index
    #         if vector_index is None and env_config is not None and "vector_index" in env_config:
    #             vector_index = env_config.vector_index
    #
    #     return worker_index, vector_index

    def _capture(self, obs):
        if not self._recording: return
        # Maybe we dont need this as technically not necessary
        # if len(self._frames) >= self.max_frames: return

        lst: list[np.ndarray] = [] # frames list
        if self.is_multi and isinstance(obs, (list, tuple)):
            wd = []
            for i in obs:
                o = _select_image(i)
                if o is None:
                    continue
                wd.append(_to_hwc_uint8(o))
            if wd:
                lst.append(_tile_ahwc(wd))
        else:
            o = _select_image(obs)
            if o is not None:
                lst.append(_to_hwc_uint8(o))

        for i in lst:
            self._frames.append(i)

    def _save(self, metadata=None):
        if not (self._recording and self._frames and self.out_dir is not None):
            self._frames.clear()
            return None

        vf = np.stack(self._frames, axis=0)
        vf = np.transpose(vf, (0, 3, 1, 2))
        f = os.path.join(self.out_dir, f"{uuid.uuid4().hex}.npz")
        np.savez_compressed(f, frames=vf)

        # worker_index, vector_index = self._env_indices()
        dct = dict(path=f, fps=self.fps, episode=self._ep_idx)
        # if worker_index is not None:
        #     dct["worker_index"] = int(worker_index)
        # if vector_index is not None:
        #     dct["vector_index"] = int(vector_index)

        #Logging metadata
        if metadata:
            meta_path = os.path.join(self.out_dir, f"{uuid.uuid4().hex}.json")
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2, sort_keys=True)
            dct.update(metadata)
            dct["meta_path"] = meta_path

        self._frames.clear()
        return dct

    def _maybe_start_ep(self, obs):
        self._ep_idx += 1
        self._recording = self.every_n > 0 and (self._ep_idx % self.every_n == 0)
        self._frames.clear()
        if self._recording:
            self._capture(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._maybe_start_ep(obs)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        term_list = _as_bool_sequence(term)
        trunc_list = _as_bool_sequence(trunc)
        if term_list is not None or trunc_list is not None:
            if term_list is None:
                term_list = [bool(term)] * len(trunc_list)
            if trunc_list is None:
                trunc_list = [bool(trunc)] * len(term_list)
            done_flags = [bool(t) or bool(tr) for t, tr in zip(term_list, trunc_list)]
            ep_done = all(done_flags) if self.done_mode == "all" else any(done_flags)
        else:
            ep_done = bool(term) or bool(trunc)

        # Check if env reseted itself and returned the next episode's observation
        # This happens on ordinary auto-reset boundaries and on crash-recovery retries in multiplayer.
        # When the env reset itself, obs is from new ep, so we need to call _maybe_start_ep again
        # But with env didn't auto reset, _maybe_start_ep will be called from next reset()
        # So this check is necessary to not miss starting recording for ep auto reseted
        reset_boundary = False
        if isinstance(info, list):
            reset_boundary = any("reset_info" in d for d in info if isinstance(d, dict))
        elif isinstance(info, dict):
            reset_boundary = "reset_info" in info

        hidden_reset_count = _hidden_reset_count(info)
        if hidden_reset_count > 0:
            self._recording = False
            self._frames.clear()

            skipped_episodes = max(0, hidden_reset_count - 1)
            if skipped_episodes > 0:
                self._ep_idx += skipped_episodes

            if not (ep_done and reset_boundary):
                self._maybe_start_ep(obs)
                if not ep_done:
                    return obs, r, term, trunc, info
            else:
                # Current observation already belongs to the episode after this terminal transition
                # So count the hidden current episode but dont start recording the next one until the normal reset boundary handling at the end of the method
                self._ep_idx += 1

        # Multi-agent envs auto-reset before returning on terminal transitions.
        # Do not contaminate the finished episode with the first frame of the next one.
        if not (ep_done and reset_boundary):
            self._capture(obs)

        # This is the payload to send to video_uploader
        data = None
        if ep_done:
            # Logging purpose only.
            metadata = {
                "recorded_frames": len(self._frames),
                "reset_boundary": bool(reset_boundary),
            }

            if term_list is not None or trunc_list is not None:
                if self.done_mode == "all":
                    metadata["terminated"] = bool(term_list is not None and all(bool(t) for t in term_list))
                    metadata["truncated"] = bool(trunc_list is not None and all(bool(t) for t in trunc_list))
                else:
                    metadata["terminated"] = bool(term_list is not None and any(bool(t) for t in term_list))
                    metadata["truncated"] = bool(trunc_list is not None and any(bool(t) for t in trunc_list))
                if term_list is not None:
                    metadata["agent_terminated"] = [bool(t) for t in term_list]
                if trunc_list is not None:
                    metadata["agent_truncated"] = [bool(t) for t in trunc_list]
            else:
                metadata["terminated"] = bool(term)
                metadata["truncated"] = bool(trunc)
            metadata["done_mode"] = self.done_mode

            if isinstance(info, dict):
                metadata.update(_info_scalars(info))
            elif isinstance(info, list):
                metadata.update(_multi_info_scalars(info))
                for item in info:
                    if isinstance(item, dict) and "true_objective" in item:
                        scalar = _json_scalar(item.get("true_objective"))
                        if scalar is not None:
                            metadata["true_objective"] = scalar
                            break
            # Set to none if no logging.
            data = self._save(metadata=metadata)
            self._recording = False
            self._frames.clear()
            if reset_boundary or self.done_mode == "any":
                self._maybe_start_ep(obs)

        if data is not None:
            if isinstance(info, dict):
                info.setdefault("episode_extra_stats", {})["wandb_video"] = data
            elif isinstance(info, list) and info:
                info[0].setdefault("episode_extra_stats", {})["wandb_video"] = data

        return obs, r, term, trunc, info

    def close(self):
        self._frames.clear()
        super().close()
