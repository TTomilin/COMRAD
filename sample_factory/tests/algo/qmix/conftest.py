"""QMIX test fixtures and shared stubs."""

import gymnasium as gym
import pytest
import torch
from torch import nn

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.joint_sequence_replay_buffer import JointSequenceReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import preprocess_cfg
from sample_factory.utils.attr_dict import AttrDict
from comrad.models.qmix_model import QMixAgentNet, QMixMixer, make_mixer


class AgentRnnStub:
    def __init__(self, rnn_size: int = 2, num_actions: int = 3, enc_dim: int = 2):
        self._rnn_size = rnn_size
        self.num_actions = num_actions
        self.enc_dim = enc_dim
        self.action_sizes = [num_actions]

    def get_rnn_size(self) -> int:
        return self._rnn_size

    def encode(self, obs: TensorDict) -> torch.Tensor:
        if isinstance(obs, dict) and "obs" in obs:
            val = obs["obs"]
        else:
            val = obs if torch.is_tensor(obs) else list(obs.values())[0]
        batch_size = val.shape[0]
        return torch.zeros(batch_size, self.enc_dim, device=val.device)

    def forward_head(self, encoded: torch.Tensor, rnn_states: torch.Tensor):
        """Core -> decoder -> Q-head on pre-encoded features"""
        new_rnn = rnn_states + 1.0
        batch = rnn_states.shape[0]
        q_values = torch.zeros(batch, self.num_actions, device=new_rnn.device)
        q_values[:, 0] = new_rnn[:, 0]
        return q_values, new_rnn

    def forward_decomposed(self, obs: TensorDict, rnn_states: torch.Tensor):
        encoded = self.encode(obs)
        q_values, new_rnn = self.forward_head(encoded, rnn_states)
        return q_values, new_rnn, encoded


class SumMixer(nn.Module):
    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return agent_qs.sum(dim=-1)


class DummyEnvInfo:
    def __init__(self, num_agents: int = 2):
        self.obs_space = None
        self.num_agents = num_agents


def make_qmix_cfg(use_rnn: bool, rnn_size: int = 16, rnn_num_layers: int = 1) -> AttrDict:
    return AttrDict(
        {
            'encoder_conv_architecture': 'convnet_simple',
            'encoder_conv_mlp_layers': [],
            'encoder_extra_fc_layers': 0,
            'hidden_size': 32,
            'nonlinearity': 'relu',
            'use_rnn': use_rnn,
            'rnn_type': 'gru',
            'rnn_size': rnn_size,
            'rnn_num_layers': rnn_num_layers,
            'decoder_mlp_layers': [],
        }
    )


def make_full_qmix_cfg(**overrides) -> AttrDict:
    """Create a complete QMIX config that passes through preprocess_cfg + verify_cfg."""
    cfg = AttrDict({
        'algo': 'QMIX',
        'recurrence': -1,
        'rollout': 8,
        'use_rnn': False,
        'rnn_type': 'gru',
        'per': False,
        'actor_critic_share_weights': True,
        'qmix_buffer_batch_size': 32,
        'qmix_sequence_batch_size': 8,
        'qmix_log_interval': 100,
        'mixer': 'qmix',
        'num_agents': 2,
        'cli_args': {},
        'replay_buffer_size': 10000,
        'learning_starts': 5000,
        # verify_cfg requirements
        'num_envs_per_worker': 2,
        'worker_num_splits': 1,
        'normalize_returns': False,
        'with_vtrace': False,
        'async_rl': True,
        'serial_mode': False,
        'num_policies': 1,
        'batched_sampling': True,
        'batch_size': 256,
        'num_batches_per_epoch': 1,
        'num_workers': 2,
    })
    cfg.update(overrides)
    return cfg


def make_spaces():
    obs_space = gym.spaces.Dict({'obs': gym.spaces.Box(0, 1, shape=(3, 64, 64))})
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
    return obs_space, action_space


def small_obs_space():
    return gym.spaces.Dict({'obs': gym.spaces.Box(0, 1, shape=(3, 64, 64), dtype='float32')})


def compound_action_space():
    return gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))


def single_action_space():
    return gym.spaces.Discrete(4)


def make_sequence_batch(num_envs: int = 2, num_agents: int = 2, rollout: int = 3, rnn_size: int = 4):
    num_traj = num_envs * num_agents
    obs = torch.randn(num_traj, rollout + 1, 1)
    actions = torch.randint(0, 3, (num_traj, rollout))
    rewards = torch.randn(num_traj, rollout)
    dones = torch.zeros(num_traj, rollout)
    time_outs = torch.zeros(num_traj, rollout)
    rnn_states = torch.randn(num_traj, rollout + 1, rnn_size)
    env_idx = torch.tensor([[0] * rollout, [0] * rollout, [1] * rollout, [1] * rollout], dtype=torch.long)
    agent_idx = torch.tensor([[0] * rollout, [1] * rollout, [0] * rollout, [1] * rollout], dtype=torch.long)
    return TensorDict(
        {
            'obs': TensorDict({'obs': obs}),
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'time_outs': time_outs,
            'rnn_states': rnn_states,
            'env_idx': env_idx,
            'agent_idx': agent_idx,
        }
    )


def make_real_obs_batch(batch_size: int, t_plus_one: int, num_agents: int, obs_shape=(3, 64, 64)):
    """Create a TensorDict of obs with shape [B, T+1, N, *obs_shape]."""
    return TensorDict({'obs': torch.rand(batch_size, t_plus_one, num_agents, *obs_shape)})


def real_model_cfg(rnn_size: int = 32, rnn_num_layers: int = 1, mixer: str = 'qmix') -> AttrDict:
    """Config suitable for constructing a real QMixAgentNet + mixer."""
    return AttrDict({
        'encoder_conv_architecture': 'convnet_simple',
        'encoder_conv_mlp_layers': [],
        'encoder_extra_fc_layers': 0,
        'hidden_size': 32,
        'nonlinearity': 'relu',
        'use_rnn': True,
        'rnn_type': 'gru',
        'rnn_size': rnn_size,
        'rnn_num_layers': rnn_num_layers,
        'decoder_mlp_layers': [],
        'mixer': mixer,
        'qmix_embed_dim': 16,
        'qmix_hypernet_hidden': 32,
        'gamma': 0.99,
        'double_dqn': True,
        'q_value_clamp': 100.0,
        'use_huber_loss': True,
    })
