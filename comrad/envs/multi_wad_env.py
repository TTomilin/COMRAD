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
        swap_every: int = 1,
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
        self._episode_reward = 0.0
        self._cfg_dir = tempfile.mkdtemp(prefix="comrad_wad_")
        self._current: Optional[WadInfo] = None
        self._current_task_idx: Optional[int] = None

    def close(self):
        super().close()
        if self._cfg_dir and os.path.isdir(self._cfg_dir):
            shutil.rmtree(self._cfg_dir, ignore_errors=True)
            self._cfg_dir = None

    def step(self, action):
        self._ensure_current_wad()
        obs, reward, terminated, truncated, info = self.env.step(action)
        # expose current task index so SF can store it in the trajectory buffer
        if isinstance(info, dict):
            info["task_idx"] = self._current_task_idx
        elif isinstance(info, (list, tuple)):
            for d in info:
                if isinstance(d, dict):
                    d["task_idx"] = self._current_task_idx
        r = float(np.mean(reward)) if isinstance(reward, (list, tuple)) else float(reward)
        self._episode_reward += r

        if self._did_inner_env_auto_reset(terminated, truncated, info):
            self._eps += 1
            if self.curriculum is not None:
                episode_success = self._episode_success_metric(info, fallback=self._episode_reward)
                self.curriculum.update(self._current_task_idx, episode_success)
            self._episode_reward = 0.0
            if self._should_swap_before_next_episode():
                next_idx = self._sample_wad_idx()
                if next_idx != self._current_task_idx:
                    self._current_task_idx = next_idx
                    obs, info = self._swap_for_next_episode(obs, info, self.batch.entries[next_idx])

        return obs, reward, terminated, truncated, info

    def _sample_wad_idx(self) -> int:
        if self.curriculum is not None:
            return self.curriculum.sample()
        return self._rng.randrange(len(self.batch.entries))

    def _ensure_current_wad(self) -> None:
        if self._current_task_idx is not None:
            return

        self._current_task_idx = self._sample_wad_idx()
        self._apply_swap(self.batch.entries[self._current_task_idx])

    def _apply_swap(self, wad: WadInfo) -> None:
        cfg_out = os.path.join(self._cfg_dir, f"{wad.name}.cfg")
        patch_wad_path(self.base_cfg, wad.wad_path, cfg_out)
        self._current = wad
        self.env.unwrapped.swap_scenario(cfg_out)

    def _swap_for_next_episode(self, obs, info, wad: WadInfo):
        self._apply_swap(wad)

        if not self._step_already_reset(info):
            return obs, info

        obs, reset_info = self.env.reset()
        if isinstance(info, dict):
            info["reset_info"] = reset_info
        elif isinstance(info, (list, tuple)):
            for agent_info, agent_reset_info in zip(info, reset_info):
                if isinstance(agent_info, dict):
                    agent_info["reset_info"] = agent_reset_info
        return obs, info

    @staticmethod
    def _step_already_reset(info) -> bool:
        if isinstance(info, dict):
            return "reset_info" in info
        if isinstance(info, (list, tuple)):
            return any(isinstance(d, dict) and "reset_info" in d for d in info)
        return False

    @staticmethod
    def _episode_success_metric(info, fallback: float) -> float:
        """
        Prefer per-episode `true_objective` as the curriculum success signal.
        Fall back to episodic reward only for scenarios that do not emit it.
        """
        if isinstance(info, dict):
            true_objective = info.get("true_objective", None)
            if true_objective is not None:
                return float(true_objective)
            return float(fallback)

        if isinstance(info, (list, tuple)):
            true_objectives = [
                float(agent_info["true_objective"])
                for agent_info in info
                if isinstance(agent_info, dict) and "true_objective" in agent_info
            ]
            if true_objectives:
                return float(np.mean(true_objectives))

        return float(fallback)

    def _should_swap_before_next_episode(self) -> bool:
        return self._eps > 0 and self._eps % self.swap_every == 0

    def _did_inner_env_auto_reset(self, terminated, truncated, info) -> bool:
        if isinstance(terminated, (list, tuple)):
            return all(t or tr for t, tr in zip(terminated, truncated))
        return bool(terminated) or bool(truncated)

    def reset(self, **kwargs):
        self._ensure_current_wad()
        obs, info = self.env.reset(**kwargs)
        if isinstance(info, dict):
            info["task_idx"] = self._current_task_idx
        elif isinstance(info, (list, tuple)):
            for d in info:
                if isinstance(d, dict):
                    d["task_idx"] = self._current_task_idx
        return obs, info
