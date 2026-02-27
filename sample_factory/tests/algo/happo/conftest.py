"""HAPPO test fixtures and shared helpers."""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.context import global_model_factory, sf_global_context
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from sf.doom.happo_model import (
    HAPPOActorCritic,
    _group_by_env,
    make_happo_actor_critic,
    remove_agentid,
)
from sf.doom.wrappers.agent_id_wrapper import AgentIDWrapper
from sample_factory.algo.learning.learner_happo import HAPPOLearner


def _make_happo_cfg(
    *,
    num_agents: int = 2,
    use_rnn: bool = False,
    rnn_type: str = "gru",
    rnn_size: int = 64,
    rnn_num_layers: int = 1,
    hidden_size: int = 32,
    happo_critic_hidden_sizes: list = None,
    happo_critic_rnn: bool = False,
) -> AttrDict:
    if happo_critic_hidden_sizes is None:
        happo_critic_hidden_sizes = [64, 32]
    return AttrDict(
        {
            "encoder_conv_architecture": "convnet_simple",
            "encoder_conv_mlp_layers": [],
            "encoder_extra_fc_layers": 0,
            "hidden_size": hidden_size,
            "nonlinearity": "relu",
            "use_rnn": use_rnn,
            "rnn_type": rnn_type,
            "rnn_size": rnn_size,
            "rnn_num_layers": rnn_num_layers,
            "decoder_mlp_layers": [],
            "normalize_input": False,
            "normalize_returns": False,
            "obs_subtract_mean": 0.0,
            "obs_scale": 1.0,
            "num_agents": num_agents,
            "algo": "HAPPO",
            "adaptive_stddev": True,
            "initial_stddev": 1.0,
            "policy_initialization": "orthogonal",
            "policy_init_gain": 1.0,
            "actor_critic_share_weights": True,
            "happo_critic_hidden_sizes": happo_critic_hidden_sizes,
            "happo_critic_rnn": happo_critic_rnn,
        }
    )


def _make_obs_space(obs_shape=(3, 64, 64), num_agents=2):
    """Create observation space WITH agent_id (as HAPPO expects after AgentIDWrapper)."""
    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(0, 1, shape=obs_shape, dtype=np.float32),
            "agent_id": gym.spaces.Box(0, 1, shape=(num_agents,), dtype=np.float32),
        }
    )


def _make_obs_space_no_id(obs_shape=(3, 64, 64)):
    """Create observation space WITHOUT agent_id (raw env obs)."""
    return gym.spaces.Dict(
        {"obs": gym.spaces.Box(0, 1, shape=obs_shape, dtype=np.float32)}
    )


def _make_action_space(n_actions=4):
    return gym.spaces.Discrete(n_actions)


def _make_happo_model(num_agents=2, use_rnn=False, obs_shape=(3, 64, 64), n_actions=4):
    """Create a HAPPOActorCritic model with default config."""
    cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=use_rnn)
    obs_space = _make_obs_space(obs_shape=obs_shape, num_agents=num_agents)
    action_space = _make_action_space(n_actions)
    return make_happo_actor_critic(cfg, obs_space, action_space)


def _make_obs_batch(batch_size, num_agents, obs_shape=(3, 64, 64)):
    """Create a batch of observations with agent_id one-hot vectors."""
    obs = torch.randn(batch_size, *obs_shape)
    agent_id = torch.zeros(batch_size, num_agents)
    for i in range(batch_size):
        agent_id[i, i % num_agents] = 1.0
    return {"obs": obs, "agent_id": agent_id}


@pytest.fixture
def happo_context():
    """Initialize sf_global_context for tests that need it."""
    sf_global_context()
    yield
