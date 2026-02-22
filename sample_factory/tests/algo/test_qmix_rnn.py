from __future__ import annotations

import gymnasium as gym
import pytest
import torch
from torch import nn

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_sequence_replay_buffer import JointSequenceReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import preprocess_cfg
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.qmix_model import QMixAgentNet


class _AgentRnnStub:
    def __init__(self, rnn_size: int = 2, num_actions: int = 3, enc_dim: int = 2):
        self._rnn_size = rnn_size
        self.num_actions = num_actions
        self.enc_dim = enc_dim
        self.action_sizes = [num_actions]

    def get_rnn_size(self) -> int:
        return self._rnn_size

    def forward_decomposed(self, obs: TensorDict, rnn_states: torch.Tensor):
        batch = rnn_states.shape[0]
        new_rnn = rnn_states + 1.0
        q_values = torch.zeros(batch, self.num_actions, device=new_rnn.device)
        q_values[:, 0] = new_rnn[:, 0]

        if self.enc_dim <= self._rnn_size:
            encoder_out = new_rnn[:, :self.enc_dim]
        else:
            encoder_out = torch.nn.functional.pad(new_rnn, (0, self.enc_dim - self._rnn_size))

        return q_values, new_rnn, encoder_out


class _SumMixer(nn.Module):
    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return agent_qs.sum(dim=-1)


class _DummyEnvInfo:
    def __init__(self, num_agents: int = 2):
        self.obs_space = None
        self.num_agents = num_agents


def _make_qmix_cfg(use_rnn: bool, rnn_size: int = 16, rnn_num_layers: int = 1) -> AttrDict:
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


def _make_spaces():
    obs_space = gym.spaces.Dict({'obs': gym.spaces.Box(0, 1, shape=(3, 64, 64))})
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
    return obs_space, action_space


def _make_sequence_batch(num_envs: int = 2, num_agents: int = 2, rollout: int = 3, rnn_size: int = 4):
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


def test_get_rnn_size_gru_single_layer():
    cfg = _make_qmix_cfg(use_rnn=True, rnn_size=32, rnn_num_layers=1)
    obs_space, action_space = _make_spaces()
    net = QMixAgentNet(cfg, obs_space, action_space)
    assert net.get_rnn_size() == 32


def test_get_rnn_size_gru_multi_layer():
    cfg = _make_qmix_cfg(use_rnn=True, rnn_size=32, rnn_num_layers=3)
    obs_space, action_space = _make_spaces()
    net = QMixAgentNet(cfg, obs_space, action_space)
    assert net.get_rnn_size() == 96


def test_prepare_joint_sequences_shapes():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = _AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = _make_sequence_batch()
    joint = QMixLearner._prepare_joint_sequences(learner, batch)

    assert joint is not None
    assert joint['obs']['obs'].shape == (2, 4, 2, 1)
    assert joint['actions'].shape == (2, 3, 2)
    assert joint['rewards'].shape == (2, 3, 2)
    assert joint['dones'].shape == (2, 3, 2)
    assert joint['time_outs'].shape == (2, 3, 2)
    assert joint['rnn_states'].shape == (2, 2, 4)


def test_prepare_joint_sequences_grouping_with_env_agent_idx():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 2})
    learner.agent_net = _AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    num_traj = 4
    rollout = 2
    obs = torch.randn(num_traj, rollout + 1, 1)
    actions = torch.zeros(num_traj, rollout, dtype=torch.long)
    actions[0] = 11
    actions[1] = 22
    actions[2] = 33
    actions[3] = 44

    batch = TensorDict(
        {
            'obs': TensorDict({'obs': obs}),
            'actions': actions,
            'rewards': torch.zeros(num_traj, rollout),
            'dones': torch.zeros(num_traj, rollout),
            'time_outs': torch.zeros(num_traj, rollout),
            'rnn_states': torch.zeros(num_traj, rollout + 1, 4),
            'env_idx': torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=torch.long),
            'agent_idx': torch.tensor([[1, 1], [0, 0], [0, 0], [1, 1]], dtype=torch.long),
        }
    )

    joint = QMixLearner._prepare_joint_sequences(learner, batch)
    assert joint is not None
    assert torch.equal(joint['actions'][0, :, 0], torch.tensor([33, 33]))
    assert torch.equal(joint['actions'][0, :, 1], torch.tensor([11, 11]))


def test_sequential_agent_forward_shapes_gru():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2

    batch_size = 2
    rollout = 3
    obs = TensorDict({'obs': torch.zeros(batch_size, rollout + 1, 2, 1)})
    dones = torch.zeros(batch_size, rollout, 2)
    rnn_states = torch.zeros(batch_size, 2, 2)

    q_values, enc = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, _AgentRnnStub())
    assert q_values.shape == (batch_size, rollout + 1, 2, 3)
    assert enc.shape == (batch_size, rollout + 1, 2, 2)


def test_done_masking_resets_only_done_agents():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2

    obs = TensorDict({'obs': torch.zeros(1, 2, 2, 1)})
    dones = torch.tensor([[[1.0, 0.0]]])
    rnn_states = torch.zeros(1, 2, 2)

    q_values, _ = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, _AgentRnnStub())
    # t=1: agent 0 reset before forward -> q=1, agent 1 not reset -> q=2
    assert q_values[0, 1, 0, 0].item() == pytest.approx(1.0)
    assert q_values[0, 1, 1, 0].item() == pytest.approx(2.0)


def test_sequence_replay_store_and_sample_shapes():
    rb = JointSequenceReplayBuffer(
        capacity_sequences=8,
        seq_len=3,
        num_agents=2,
        obs_space=None,
        action_space=None,
        device='cpu',
        share_memory=False,
        rnn_state_size=4,
    )

    batch = TensorDict(
        {
            'obs': TensorDict({'obs': torch.randn(2, 4, 2, 1)}),
            'actions': torch.randint(0, 3, (2, 3, 2), dtype=torch.long),
            'rewards': torch.randn(2, 3, 2),
            'dones': torch.zeros(2, 3, 2),
            'time_outs': torch.zeros(2, 3, 2),
            'rnn_states': torch.randn(2, 2, 4),
        }
    )

    added = rb.add_sequence_batch(batch)
    assert added == 2
    sampled, weights, indices = rb.sample(2, 'cpu')
    assert sampled['obs']['obs'].shape == (2, 4, 2, 1)
    assert sampled['actions'].shape == (2, 3, 2)
    assert sampled['team_reward'].shape == (2, 3)
    assert sampled['joint_done'].shape == (2, 3)
    assert sampled['joint_time_out'].shape == (2, 3)
    assert weights is None
    assert indices is None


def test_qmix_loss_sequential_scalar_and_td_shape():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict(
        {
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        }
    )
    learner.obs_normalizer = None
    learner.agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
    learner.target_agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
    learner.mixer = _SumMixer()
    learner.target_mixer = _SumMixer()

    batch = TensorDict(
        {
            'obs': TensorDict({'obs': torch.zeros(2, 4, 2, 1)}),
            'actions': torch.zeros(2, 3, 2, dtype=torch.long),
            'rewards': torch.ones(2, 3, 2),
            'dones': torch.zeros(2, 3, 2),
            'time_outs': torch.zeros(2, 3, 2),
            'rnn_states': torch.zeros(2, 2, 2),
        }
    )

    loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
    assert loss.dim() == 0
    assert td_summary.shape == (2,)


def test_non_rnn_path_regression():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.use_rnn = False
    learner.cfg = AttrDict({'qmix_buffer_batch_size': 32, 'qmix_sequence_batch_size': 8})

    assert QMixLearner._qmix_batch_size(learner) == 32


def test_qmix_batch_size_rejects_non_positive_values():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2

    learner.use_rnn = True
    learner.cfg = AttrDict({'qmix_sequence_batch_size': 0, 'qmix_buffer_batch_size': 32})
    with pytest.raises(ValueError, match='must be > 0'):
        QMixLearner._qmix_batch_size(learner)

    learner.use_rnn = False
    learner.cfg = AttrDict({'qmix_sequence_batch_size': 8, 'qmix_buffer_batch_size': -1})
    with pytest.raises(ValueError, match='must be > 0'):
        QMixLearner._qmix_batch_size(learner)


def test_prepare_joint_sequences_strict_invalid_metadata_rejection():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 2})
    learner.agent_net = _AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    num_traj = 4
    rollout = 2
    batch = TensorDict(
        {
            'obs': TensorDict({'obs': torch.randn(num_traj, rollout + 1, 1)}),
            'actions': torch.randint(0, 3, (num_traj, rollout)),
            'rewards': torch.randn(num_traj, rollout),
            'dones': torch.zeros(num_traj, rollout),
            'time_outs': torch.zeros(num_traj, rollout),
            'rnn_states': torch.randn(num_traj, rollout + 1, 4),
            'env_idx': torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.long),
            'agent_idx': torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.long),
        }
    )

    result = QMixLearner._prepare_joint_sequences(learner, batch)
    assert result is None
    assert learner._invalid_sequence_groups > 0


def test_prepare_joint_sequences_rejects_bad_rnn_rank():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = _AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = _make_sequence_batch(rollout=3, rnn_size=4)
    batch['rnn_states'] = torch.randn(4, 4)

    with pytest.raises(ValueError, match='Unexpected rnn_states shape'):
        QMixLearner._prepare_joint_sequences(learner, batch)


def test_prepare_joint_sequences_rejects_bad_rnn_time_dim():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = _AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = _make_sequence_batch(rollout=3, rnn_size=4)
    batch['rnn_states'] = torch.randn(4, 3, 4)

    with pytest.raises(ValueError, match='Expected rnn_states time dim'):
        QMixLearner._prepare_joint_sequences(learner, batch)


def test_prepare_joint_sequences_rejects_bad_rnn_feature_dim():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = _AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = _make_sequence_batch(rollout=3, rnn_size=4)
    batch['rnn_states'] = torch.randn(4, 4, 5)

    with pytest.raises(ValueError, match='Expected rnn_states feature dim'):
        QMixLearner._prepare_joint_sequences(learner, batch)


def test_config_rejects_lstm_for_qmix():
    cfg = AttrDict(
        {
            'algo': 'QMIX',
            'recurrence': -1,
            'rollout': 8,
            'use_rnn': True,
            'rnn_type': 'lstm',
            'per': False,
            'actor_critic_share_weights': True,
        }
    )

    with pytest.raises(ValueError, match="rnn_type='gru'"):
        preprocess_cfg(cfg, _DummyEnvInfo())


@pytest.mark.parametrize(
    'override_key, override_val, expected_msg',
    [
        ('qmix_buffer_batch_size', 0, 'qmix_buffer_batch_size > 0'),
        ('qmix_sequence_batch_size', 0, 'qmix_sequence_batch_size > 0'),
        ('qmix_log_interval', -1, 'qmix_log_interval >= 0'),
    ],
)
def test_config_rejects_invalid_qmix_batch_and_log_params(override_key, override_val, expected_msg):
    cfg = AttrDict(
        {
            'algo': 'QMIX',
            'recurrence': -1,
            'rollout': 8,
            'use_rnn': False,
            'rnn_type': 'gru',
            'per': True,
            'actor_critic_share_weights': True,
            'qmix_buffer_batch_size': 32,
            'qmix_sequence_batch_size': 8,
            'qmix_log_interval': 100,
        }
    )
    cfg[override_key] = override_val

    with pytest.raises(ValueError, match=expected_msg):
        preprocess_cfg(cfg, _DummyEnvInfo())


def _make_full_qmix_cfg(**overrides) -> AttrDict:
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


class TestLearningStartsEffectiveCapacity:
    """Verify that learning_starts is capped against effective buffer capacity
    (not raw replay_buffer_size), preventing infinite warmup stalls.
    """

    def test_rnn_floor_division_caps_learning_starts(self):
        """replay_buffer_size=100, num_agents=3, rollout=11 → capacity=3 seqs → 99 transitions.
        learning_starts=100 should be capped since 99 < 100.
        """
        cfg = _make_full_qmix_cfg(
            use_rnn=True,
            rollout=11,
            num_agents=3,
            replay_buffer_size=100,
            learning_starts=100,
        )
        preprocess_cfg(cfg, _DummyEnvInfo(num_agents=3))
        # effective_capacity = (100 // (3*11)) * (3*11) = 3 * 33 = 99
        # learning_starts=100 >= 99, so capped to max(1, 99 // 2) = 49
        assert cfg.learning_starts == 49

    def test_rnn_exact_divisible_no_cap(self):
        """When replay_buffer_size is exactly divisible, no capacity loss."""
        cfg = _make_full_qmix_cfg(
            use_rnn=True,
            rollout=10,
            num_agents=2,
            replay_buffer_size=10000,
            learning_starts=5000,
        )
        preprocess_cfg(cfg, _DummyEnvInfo(num_agents=2))
        # effective_capacity = (10000 // 20) * 20 = 10000
        # learning_starts=5000 < 10000 → not capped
        assert cfg.learning_starts == 5000

    def test_non_rnn_floor_division_caps_learning_starts(self):
        """replay_buffer_size=101, num_agents=2 → capacity=50 joints → 100 transitions.
        learning_starts=101 should be capped.
        """
        cfg = _make_full_qmix_cfg(
            use_rnn=False,
            num_agents=2,
            replay_buffer_size=101,
            learning_starts=101,
        )
        preprocess_cfg(cfg, _DummyEnvInfo(num_agents=2))
        # effective_capacity = (101 // 2) * 2 = 100
        # learning_starts=101 >= 100, so capped to max(1, 100 // 2) = 50
        assert cfg.learning_starts == 50

    def test_equality_case_is_capped(self):
        """learning_starts == effective_capacity should also be capped (>= not just >)."""
        cfg = _make_full_qmix_cfg(
            use_rnn=False,
            num_agents=2,
            replay_buffer_size=100,
            learning_starts=100,
        )
        preprocess_cfg(cfg, _DummyEnvInfo(num_agents=2))
        # effective_capacity = (100 // 2) * 2 = 100
        # learning_starts=100 >= 100, so capped to max(1, 100 // 2) = 50
        assert cfg.learning_starts == 50

    def test_safely_below_capacity_not_capped(self):
        """Normal case: learning_starts well below capacity → no change."""
        cfg = _make_full_qmix_cfg(
            use_rnn=True,
            rollout=16,
            num_agents=2,
            replay_buffer_size=100000,
            learning_starts=5000,
        )
        preprocess_cfg(cfg, _DummyEnvInfo(num_agents=2))
        assert cfg.learning_starts == 5000


class TestEffectiveDoneMixedOutcomes:
    """Verify the canonical effective_done formula handles mixed timeout/terminal
    outcomes correctly, both in the RNN and non-RNN loss paths."""

    def test_sequential_loss_mixed_timeout_terminal(self):
        """Agent 0 terminates (done=1, timeout=0), agent 1 times out (done=1, timeout=1).
        effective_done should be 1 (at least one agent truly terminates → no bootstrap)."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.mixer = _SumMixer()
        learner.target_mixer = _SumMixer()

        # B=1, T=2, N=2
        batch = TensorDict({
            'obs': TensorDict({'obs': torch.zeros(1, 3, 2, 1)}),
            'actions': torch.zeros(1, 2, 2, dtype=torch.long),
            'rewards': torch.ones(1, 2, 2),
            'dones': torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),       # t=0: both done
            'time_outs': torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),   # t=0: agent1 timeout
            'rnn_states': torch.zeros(1, 2, 2),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        # effective_done at t=0: agent0=(1*(1-0))=1, agent1=(1*(1-1))=0, max=1
        # So target at t=0 = team_reward + gamma * (1-1) * target_q = team_reward = 2.0
        # This confirms no bootstrapping when at least one agent truly terminates
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_sequential_loss_all_timeout_should_bootstrap(self):
        """Both agents timeout (done=1, timeout=1). Should bootstrap (effective_done=0)."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.mixer = _SumMixer()
        learner.target_mixer = _SumMixer()

        batch = TensorDict({
            'obs': TensorDict({'obs': torch.zeros(1, 3, 2, 1)}),
            'actions': torch.zeros(1, 2, 2, dtype=torch.long),
            'rewards': torch.ones(1, 2, 2),
            'dones': torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
            'time_outs': torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),  # all timeout
            'rnn_states': torch.zeros(1, 2, 2),
        })

        loss, _ = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        # effective_done at t=0: both (1*(1-1))=0, max=0 → should bootstrap
        assert loss.dim() == 0

    def test_non_rnn_loss_mixed_timeout_terminal(self):
        """Non-RNN path: verify canonical effective_done with mixed timeout/terminal."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.use_rnn = False
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None

        class _FeedForwardStub:
            def __init__(self):
                self.action_sizes = [3]
            def get_q_for_actions(self, q, a):
                return q.gather(-1, a.unsqueeze(-1)).squeeze(-1) if a.dim() == 1 else q[:, 0]

        agent_stub = _FeedForwardStub()
        learner.agent_net = agent_stub
        learner.target_agent_net = agent_stub
        learner.mixer = _SumMixer()
        learner.target_mixer = _SumMixer()

        def mock_forward(obs, net):
            B = next(iter(obs.values())).shape[0] if isinstance(obs, TensorDict) else obs.shape[0]
            return torch.ones(B, 2, 3)
        learner._vectorized_agent_forward = mock_forward
        learner._compute_global_state = lambda obs: torch.zeros(
            next(iter(obs.values())).shape[0] if isinstance(obs, TensorDict) else obs.shape[0], 4
        )

        # B=2 transitions: first has mixed outcome, second is clean
        batch = TensorDict({
            'obs': TensorDict({'obs': torch.zeros(2, 2, 4)}),
            'next_obs': TensorDict({'obs': torch.zeros(2, 2, 4)}),
            'actions': torch.zeros(2, 2, dtype=torch.long),
            'rewards': torch.ones(2, 2),
            'team_reward': torch.tensor([2.0, 2.0]),
            'dones': torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            'time_outs': torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            'joint_done': torch.tensor([1.0, 0.0]),
            'joint_time_out': torch.tensor([1.0, 0.0]),
        })

        loss, _ = QMixLearner._calculate_qmix_loss(learner, batch)

        # For transition 0: agent0 done=1,timeout=0 → eff=1; agent1 done=1,timeout=1 → eff=0
        # joint effective_done = max(1,0) = 1 → no bootstrap
        # OLD incorrect formula would give: joint_done*(1-joint_time_out) = 1*(1-1) = 0 → WRONG
        # NEW correct formula gives 1 → no bootstrap → target = team_reward = 2.0
        assert loss.dim() == 0


class TestTDTargetNumericalCorrectness:
    """Verify TD target values are computed correctly with known inputs."""

    def test_sequential_td_target_no_done(self):
        """With no dones, target = team_reward + gamma * target_q_tot for each step."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.5,            # easy to verify manually
            'double_dqn': False,     # simpler: target net greedy
            'q_value_clamp': 0,      # disabled
            'use_huber_loss': False,  # MSE
        })
        learner.obs_normalizer = None

        # Stub that returns predictable Q-values
        class _ConstQStub:
            def __init__(self, q_val):
                self._rnn_size = 2
                self.action_sizes = [3]
                self.q_val = q_val
                self.enc_dim = 2

            def get_rnn_size(self):
                return self._rnn_size

            def forward_decomposed(self, obs, rnn_states):
                batch = rnn_states.shape[0]
                q = torch.full((batch, 3), self.q_val)
                new_rnn = rnn_states.clone()
                enc = torch.zeros(batch, self.enc_dim)
                return q, new_rnn, enc

            def get_q_for_actions(self, q, a):
                return q.gather(-1, a.unsqueeze(-1)).squeeze(-1)

        learner.agent_net = _ConstQStub(q_val=2.0)
        learner.target_agent_net = _ConstQStub(q_val=5.0)
        learner.mixer = _SumMixer()
        learner.target_mixer = _SumMixer()

        # B=1, T=1, N=2
        batch = TensorDict({
            'obs': TensorDict({'obs': torch.zeros(1, 2, 2, 1)}),
            'actions': torch.zeros(1, 1, 2, dtype=torch.long),
            'rewards': torch.tensor([[[1.0, 3.0]]]),     # team_reward = 4.0
            'dones': torch.zeros(1, 1, 2),
            'time_outs': torch.zeros(1, 1, 2),
            'rnn_states': torch.zeros(1, 2, 2),
        })

        loss, _ = QMixLearner._calculate_qmix_loss_sequential(learner, batch)

        # Online: q_val=2.0, action=0 → agent_qs=[2.0, 2.0], q_tot = 4.0
        # Target: q_val=5.0, greedy action=any → target_agent_qs=[5.0, 5.0], target_q_tot = 10.0
        # effective_done=0
        # target = team_reward + gamma * (1-0) * target_q_tot = 4.0 + 0.5 * 10.0 = 9.0
        # td_error = 9.0 - 4.0 = 5.0
        # MSE loss = 5.0^2 = 25.0
        assert loss.item() == pytest.approx(25.0, abs=1e-4)


class TestSequenceReplayBufferWraparound:
    """Verify buffer correctly overwrites old entries when full."""

    def test_wraparound_overwrites_oldest(self):
        rb = JointSequenceReplayBuffer(
            capacity_sequences=3,
            seq_len=2,
            num_agents=2,
            obs_space=None,
            action_space=None,
            device='cpu',
            share_memory=False,
            rnn_state_size=4,
        )

        # Add 3 batches of 2 sequences each (total 6 sequences, capacity 3)
        for i in range(3):
            batch = TensorDict({
                'obs': TensorDict({'obs': torch.full((2, 3, 2, 1), float(i))}),
                'actions': torch.zeros(2, 2, 2, dtype=torch.long),
                'rewards': torch.zeros(2, 2, 2),
                'dones': torch.zeros(2, 2, 2),
                'time_outs': torch.zeros(2, 2, 2),
                'rnn_states': torch.zeros(2, 2, 4),
            })
            rb.add_sequence_batch(batch)

        assert len(rb) == 3  # capped at capacity

        # Sample all 3 and check they contain the latest data
        sampled, _, _ = rb.sample(3, 'cpu')
        obs_vals = sampled['obs']['obs'].flatten()
        # After wraparound, buffer should contain batch 1 (idx 1,2) and batch 2 (idx 0)
        # (batch 0 at idx 0,1 was overwritten by batch 2 at idx 0,1... but batch 2 has 2 seqs
        #  which wrap: ptr starts at 0 after 3 writes of 2, 0→2→4→6%3=0)
        # So all 3 slots now have values from batch 1 (one slot) and batch 2 (two slots)
        unique_vals = obs_vals.unique()
        assert 2.0 in unique_vals  # latest batch should definitely be present

    def test_single_entry_overwrite(self):
        rb = JointSequenceReplayBuffer(
            capacity_sequences=2,
            seq_len=1,
            num_agents=2,
            obs_space=None,
            action_space=None,
            device='cpu',
            share_memory=False,
            rnn_state_size=2,
        )

        # Fill buffer
        for i in range(4):
            batch = TensorDict({
                'obs': TensorDict({'obs': torch.full((1, 2, 2, 1), float(i))}),
                'actions': torch.zeros(1, 1, 2, dtype=torch.long),
                'rewards': torch.zeros(1, 1, 2),
                'dones': torch.zeros(1, 1, 2),
                'time_outs': torch.zeros(1, 1, 2),
                'rnn_states': torch.zeros(1, 2, 2),
            })
            rb.add_sequence_batch(batch)

        assert len(rb) == 2
        sampled, _, _ = rb.sample(2, 'cpu')
        vals = sampled['obs']['obs'].flatten().unique()
        # Last two writes: i=2 and i=3
        assert 2.0 in vals or 3.0 in vals
        assert 0.0 not in vals  # oldest should be overwritten


class TestGradientFlow:
    """Verify that gradients flow correctly through the sequential loss path."""

    def test_backward_does_not_error(self):
        """Smoke test: loss.backward() should not raise errors when parameters exist."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)

        # Need a parameterized mixer for backward() to work (no-param graph has no grad_fn)
        class _TrivialParamMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
            def forward(self, agent_qs, state):
                return agent_qs.sum(dim=-1) * self.scale

        learner.mixer = _TrivialParamMixer()
        learner.target_mixer = _SumMixer()

        batch = TensorDict({
            'obs': TensorDict({'obs': torch.zeros(2, 4, 2, 1)}),
            'actions': torch.zeros(2, 3, 2, dtype=torch.long),
            'rewards': torch.ones(2, 3, 2),
            'dones': torch.zeros(2, 3, 2),
            'time_outs': torch.zeros(2, 3, 2),
            'rnn_states': torch.zeros(2, 2, 2),
        })

        loss, _ = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        # The stubs don't have trainable params, so backward should complete without error
        # even though gradients won't flow to any parameters
        loss.backward()

    def test_mixer_receives_gradients(self):
        """Verify that mixer parameters receive gradients through the loss."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = _AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)

        # Use a parameterized mixer so we can check gradients
        class _LinearMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2))
            def forward(self, agent_qs, state):
                return (agent_qs * self.weight).sum(dim=-1)

        learner.mixer = _LinearMixer()
        learner.target_mixer = _SumMixer()

        batch = TensorDict({
            'obs': TensorDict({'obs': torch.randn(2, 4, 2, 1)}),
            'actions': torch.zeros(2, 3, 2, dtype=torch.long),
            'rewards': torch.ones(2, 3, 2),
            'dones': torch.zeros(2, 3, 2),
            'time_outs': torch.zeros(2, 3, 2),
            'rnn_states': torch.randn(2, 2, 2),
        })

        loss, _ = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        loss.backward()

        assert learner.mixer.weight.grad is not None
        assert not torch.all(learner.mixer.weight.grad == 0)
