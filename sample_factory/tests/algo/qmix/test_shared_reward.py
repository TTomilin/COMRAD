from __future__ import annotations

import gymnasium as gym
from types import SimpleNamespace
import torch

from sample_factory.utils.attr_dict import AttrDict

from comrad.utils.doom_utils import (
    get_alpha,
    get_scalarisation,
    make_doom_multiplayer_env,
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


def test_partial_shared_reward_blend_does_not_force_equal_rewards():
    env = SharedRewardWrapper(_DummyJointEnv([1.0, 3.0]), alpha=0.5, scalarisation="sum")

    _, rewards, _, _, _ = env.step(0)

    assert rewards == [2.5, 3.5]
    assert rewards[0] != rewards[1]


def test_make_doom_multiplayer_env_uses_wrapper_api_keywords(monkeypatch):
    import comrad.envs.multiagent.doom_multiagent_wrapper as doom_multiagent_wrapper

    captured = {}

    class _DummyMultiAgentEnv:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _CapturingSharedRewardWrapper:
        def __init__(self, env, *, alpha, scalarisation):
            captured["env"] = env
            captured["alpha"] = alpha
            captured["scalarisation"] = scalarisation

    monkeypatch.setattr(doom_multiagent_wrapper, "MultiAgentEnv", _DummyMultiAgentEnv)
    monkeypatch.setattr("comrad.utils.doom_utils.SharedRewardWrapper", _CapturingSharedRewardWrapper)

    doom_spec = SimpleNamespace(
        num_agents=2,
        num_bots=0,
        shared_reward_alpha=1.0,
        shared_reward_scalarisation="sum",
    )
    cfg = AttrDict(
        {
            "env_frameskip": 4,
            "num_bots": -1,
            "num_agents": 2,
            "num_humans": 0,
            "algo": "MAPPO",
            "wandb_record_every": 0,
            "with_wandb": False,
            "shared_reward_alpha": 0.25,
            "shared_reward_scalarisation": "mean",
        }
    )

    env = make_doom_multiplayer_env(doom_spec, cfg=cfg, env_config=None)

    assert env is not None
    assert captured["alpha"] == 0.25
    assert captured["scalarisation"] == "mean"
