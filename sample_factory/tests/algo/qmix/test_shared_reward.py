from __future__ import annotations

import gymnasium as gym
from types import SimpleNamespace
import torch

from sample_factory.utils.attr_dict import AttrDict

from comrad.utils.doom_utils import (
    get_alpha,
    get_scalarisation,
)
from comrad.wrappers.shared_reward import SharedRewardWrapper


class _DummyJointEnv(gym.Env):
    metadata = {}

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

    def step(self, action):
        obs = [0 for _ in self._rewards]
        terminated = [False for _ in self._rewards]
        truncated = [False for _ in self._rewards]
        infos = [{} for _ in self._rewards]
        return obs, list(self._rewards), terminated, truncated, infos

    def reset(self, **kwargs):
        return [0 for _ in self._rewards], [{} for _ in self._rewards]


def test_shared_reward_wrapper_broadcasts_sum_reward():
    env = SharedRewardWrapper(_DummyJointEnv([1.0, 3.0]), alpha=1.0, scalarisation="sum")

    _, rewards, _, _, _ = env.step(0)

    assert rewards == [4.0, 4.0]


def test_shared_reward_wrapper_blends_mean_reward():
    env = SharedRewardWrapper(_DummyJointEnv([2.0, 6.0]), alpha=0.25, scalarisation="mean")

    _, rewards, _, _, _ = env.step(0)

    assert rewards == [2.5, 5.5]


def test_broadcast_shared_reward_would_distort_qmix_under_reward_clipping():
    env = SharedRewardWrapper(_DummyJointEnv([100.0, 0.0]), alpha=1.0, scalarisation="sum")

    _, broadcast_rewards, _, _, _ = env.step(0)

    clipped_broadcast = torch.tensor(broadcast_rewards).clamp(-50.0, 50.0)
    clipped_individual = torch.tensor([100.0, 0.0]).clamp(-50.0, 50.0)

    assert clipped_broadcast.sum().item() == 100.0
    assert clipped_individual.sum().item() == 50.0


def test_shared_reward_cli_overrides_spec_defaults():
    spec = SimpleNamespace(shared_reward_alpha=1.0, shared_reward_scalarisation="sum")
    cfg = AttrDict({
        "algo": "MAPPO",
        "shared_reward_alpha": 0.25,
        "shared_reward_scalarisation": "mean",
    })

    assert get_alpha(cfg, spec) == 0.25
    assert get_scalarisation(cfg, spec) == "mean"
