from __future__ import annotations

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

class VideoLoggerWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        record_every: int = 0,
        fps: int = 35,
        max_frames: int = 1000,
        is_multi: bool = False,
        output_dir: str | None = None,
    ):
        super().__init__(env)
        self.every_n = max(0, int(record_every))
        self.fps = int(fps)
        self.max_frames = int(max_frames)
        self.is_multi = is_multi

        self.out_dir = os.fspath(output_dir) if output_dir is not None else None
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        self._ep_idx = 0
        self._recording = False
        self._frames: list[np.ndarray] = [] # list of tiled HWC uint8 frames

    def _env_indices(self):
        unwrapped = self.env.unwrapped
        worker_index = getattr(unwrapped, "worker_index", None)
        vector_index = getattr(unwrapped, "vector_index", None)

        if (worker_index is None or vector_index is None) and hasattr(unwrapped, "env_config"):
            env_config = getattr(unwrapped, "env_config")
            if worker_index is None and env_config is not None and "worker_index" in env_config:
                worker_index = env_config.worker_index
            if vector_index is None and env_config is not None and "vector_index" in env_config:
                vector_index = env_config.vector_index

        return worker_index, vector_index

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

    def _save(self):
        if not (self._recording and self._frames and self.out_dir is not None):
            self._frames.clear()
            return None

        vf = np.stack(self._frames, axis=0)
        vf = np.transpose(vf, (0, 3, 1, 2))
        f = os.path.join(self.out_dir, f"{uuid.uuid4().hex}.npz")
        np.savez_compressed(f, frames=vf)

        worker_index, vector_index = self._env_indices()
        dct = dict(path=f, fps=self.fps, episode=self._ep_idx)
        if worker_index is not None:
            dct["worker_index"] = int(worker_index)
        if vector_index is not None:
            dct["vector_index"] = int(vector_index)
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
            ep_done = all(t or tr for t, tr in zip(term_list, trunc_list))
        else:
            ep_done = bool(term) or bool(trunc)

        # Check if env reseted itself
        # When the env reset itself, obs is from new ep, so we need to call _maybe_start_ep again
        # But with env didn't auto reset, _maybe_start_ep will be called from next reset()
        # So this check is necessary to not miss starting recording for ep auto reseted
        end = False
        if ep_done:
            if isinstance(info, list):
                end = any("reset_info" in d for d in info if isinstance(d, dict))
            elif isinstance(info, dict):
                end = "reset_info" in info

        # Multi-agent envs auto-reset before returning on terminal transitions.
        # Do not contaminate the finished episode with the first frame of the next one.
        if not (ep_done and end):
            self._capture(obs)

        # This is the payload to send to video_uploader
        data = None
        if ep_done:
            data = self._save()
            self._recording = False
            self._frames.clear()
            if end:
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
