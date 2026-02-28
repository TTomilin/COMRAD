"""MAPPO test fixtures and shared helpers."""

import gymnasium as gym
import torch

from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from comrad.models.mappo_model import MAPPOActorCritic, make_mappo_actor_critic


def _make_cfg(
    *,
    algo: str = "MAPPO",
    num_agents: int = 2,
    use_rnn: bool = False,
    rnn_type: str = "gru",
    rnn_size: int = 64,
    rnn_num_layers: int = 1,
) -> AttrDict:
    return AttrDict(
        {
            "encoder_conv_architecture": "convnet_simple",
            "encoder_conv_mlp_layers": [],
            "encoder_extra_fc_layers": 0,
            "hidden_size": 32,
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
            "algo": algo,
            "adaptive_stddev": True,
            "initial_stddev": 1.0,
            "policy_initialization": "orthogonal",
            "policy_init_gain": 1.0,
            "actor_critic_share_weights": True,
        }
    )


def _make_spaces():
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))})
    action_space = gym.spaces.Discrete(4)
    return obs_space, action_space


def _make_mappo(*, algo: str = "MAPPO", num_agents: int = 2, use_rnn: bool = False, **kw):
    cfg = _make_cfg(algo=algo, num_agents=num_agents, use_rnn=use_rnn, **kw)
    obs_space, action_space = _make_spaces()
    return make_mappo_actor_critic(cfg, obs_space, action_space)
