import os
import random
import tempfile
from typing import TYPE_CHECKING, Dict, Optional

import gymnasium as gym

from comrad.envs.wad_catalog import WadBatch, WadInfo
from comrad.utils.wad_utils import patch_wad_path

# if TYPE_CHECKING:
#     from comrad.curriculum.scheduler import CurriculumScheduler


class MultiWADEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        batch: WadBatch,
        base_cfg: str,
        swap_every: int = 1,
        strategy: str = "round_robin",
        seed: int = 0,
        # scheduler: Optional["CurriculumScheduler"] = None,
    ):
        super().__init__(env)
        self.batch = batch
        self.base_cfg = base_cfg
        self.swap_every = max(1, swap_every)
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._eps = 0
        self.scheduler = None  # set later via set_scheduler() when curriculum is enabled
        self._cfg_dir = tempfile.mkdtemp(prefix="comrad_wad_")
        self._current: Optional[WadInfo] = None
        initial_wad = batch.sample(strategy, self._rng)
        self._apply_swap(initial_wad)

    def reset(self, **kwargs):
        if self._should_swap_before_next_episode():
            self._swap_to_next_wad()
        self._eps += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._did_inner_env_auto_reset(terminated, truncated, info):
            self._eps += 1
            if self._should_swap_before_next_episode():
                self._swap_to_next_wad()
                obs, _ = self.env.reset()

        return obs, reward, terminated, truncated, info

    def _apply_swap(self, info: WadInfo) -> None:
        cfg_out = os.path.join(self._cfg_dir, f"{info.name}.cfg")
        patch_wad_path(self.base_cfg, info.wad_path, cfg_out)
        self._current = info
        target = self.env if hasattr(self.env, "swap_scenario") else self.env.unwrapped
        target.swap_scenario(cfg_out)

    def _swap_to_next_wad(self) -> None:
        if self.scheduler:
            self.batch.set_weights(self.scheduler.get_weights())
        self._apply_swap(self.batch.sample(self.strategy, self._rng))

    def _should_swap_before_next_episode(self) -> bool:
        return self._eps > 0 and self._eps % self.swap_every == 0

    def _did_inner_env_auto_reset(self, terminated, truncated, info) -> bool:
        if isinstance(terminated, (list, tuple)):
            all_done = all(t or tr for t, tr in zip(terminated, truncated))
            has_reset = isinstance(info, list) and any(isinstance(i, dict) and "reset_info" in i for i in info)
            return all_done and has_reset
            
        all_done = bool(terminated) or bool(truncated)
        has_reset = isinstance(info, dict) and "reset_info" in info
        return all_done and has_reset
