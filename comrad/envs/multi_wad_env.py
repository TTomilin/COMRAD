import os
import random
import tempfile
import shutil
from typing import Optional

import numpy as np
import gymnasium as gym

from comrad.envs.wad_catalog import WadBatch, WadInfo
from comrad.utils.wad_utils import patch_wad_path


class MultiWADEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        batch: WadBatch,
        base_cfg: str,
        swap_every: int = 5,
        seed: int = 0,
        curriculum=None,
    ):
        super().__init__(env)
        self.num_agents = getattr(env, 'num_agents', None) or getattr(env.unwrapped, 'num_agents', 1)
        self.is_multiagent = getattr(env, 'is_multiagent', self.num_agents > 1)
        self.batch = batch
        self.base_cfg = base_cfg
        self.swap_every = max(1, swap_every)
        self._rng = random.Random(seed)
        self._eps = 0
        self.curriculum = curriculum
        self._episode_return = 0.0
        self._cfg_dir = tempfile.mkdtemp(prefix="comrad_wad_")
        self._current: Optional[WadInfo] = None
        self._current_task_idx = self._sample_wad_idx()
        self._apply_swap(self.batch.entries[self._current_task_idx])

    def close(self):
        super().close()
        if self._cfg_dir and os.path.isdir(self._cfg_dir):
            shutil.rmtree(self._cfg_dir, ignore_errors=True)
            self._cfg_dir = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        r = float(np.mean(reward)) if isinstance(reward, (list, tuple)) else float(reward)
        self._episode_return += r

        if self._did_inner_env_auto_reset(terminated, truncated, info):
            self._eps += 1
            if self.curriculum is not None:
                self.curriculum.update(self._current_task_idx, self._episode_return)
            self._episode_return = 0.0
            if self._should_swap_before_next_episode():
                next_idx = self._sample_wad_idx()
                if next_idx != self._current_task_idx:
                    self._current_task_idx = next_idx
                    self._apply_swap(self.batch.entries[next_idx])

        return obs, reward, terminated, truncated, info
    
    def _sample_wad_idx(self) -> int:
        if self.curriculum is not None:
            return self.curriculum.sample()
        return self._rng.randrange(len(self.batch.entries))

    def _apply_swap(self, wad: WadInfo) -> None:
        cfg_out = os.path.join(self._cfg_dir, f"{wad.name}.cfg")
        patch_wad_path(self.base_cfg, wad.wad_path, cfg_out)
        self._current = wad
        self.env.unwrapped.swap_scenario(cfg_out)

    def _should_swap_before_next_episode(self) -> bool:
        return self._eps > 0 and self._eps % self.swap_every == 0

    def _did_inner_env_auto_reset(self, terminated, truncated, info) -> bool:
        if isinstance(terminated, (list, tuple)):
            return all(t or tr for t, tr in zip(terminated, truncated))
        return bool(terminated) or bool(truncated)
