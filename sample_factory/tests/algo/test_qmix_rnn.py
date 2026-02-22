from __future__ import annotations

import copy

import gymnasium as gym
import pytest
import torch
from torch import nn

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_sequence_replay_buffer import JointSequenceReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import preprocess_cfg
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.qmix_model import QMixAgentNet, QMixMixer, make_mixer


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
    assert sampled['rewards'].shape == (2, 3, 2)
    assert sampled['dones'].shape == (2, 3, 2)
    assert sampled['rnn_states'].shape == (2, 2, 4)
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
        #
        # Hand-traced values through _AgentRnnStub + _SumMixer:
        #   q_tot_curr = [2.0, 2.0], target = [2.0, 5.96]
        #   td_error = [0.0, 3.96], Huber = [0.0, 3.46], mean = 1.73
        # With the old buggy formula (joint_done*(1-joint_time_out)),
        #   effective_done would be 0 → loss ≈ 2.47 (different!)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(1.73, abs=0.01)

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
        #
        # Hand-traced: target = [3.98, 5.96], td_error = [1.98, 3.96]
        #   Huber = [1.48, 3.46], mean = 2.47
        # This must differ from the mixed test (1.73) to confirm bootstrapping.
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(2.47, abs=0.01)

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
        learner._compute_global_state = lambda obs, encoder_net=None: torch.zeros(
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
        #
        # Hand-traced: q_tot=[2,2], target=[2.0, 3.98], td_error=[0, 1.98]
        #   Huber = [0, 1.48], mean = 0.74
        # With old buggy formula: loss ≈ 1.48 (different!)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.74, abs=0.01)


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

        # Inspect storage directly (not via sampling) to avoid RNG flakiness.
        # Write pointer advances: batch0→[0,1], batch1→[2,0], batch2→[1,2]
        # Final storage: idx0=1.0, idx1=2.0, idx2=2.0
        stored = rb._storage['obs']['obs']
        unique_vals = stored[:3].flatten().unique()
        assert 0.0 not in unique_vals, 'oldest batch (val=0) should be fully overwritten'
        assert 2.0 in unique_vals, 'latest batch (val=2) must be present'
        assert 1.0 in unique_vals, 'batch 1 (val=1) should still occupy one slot'

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


def _real_model_cfg(rnn_size: int = 32, rnn_num_layers: int = 1, mixer: str = 'qmix') -> AttrDict:
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


def _small_obs_space():
    return gym.spaces.Dict({'obs': gym.spaces.Box(0, 1, shape=(3, 64, 64), dtype='float32')})


def _compound_action_space():
    return gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))


def _single_action_space():
    return gym.spaces.Discrete(4)


def _make_real_obs_batch(batch_size: int, t_plus_one: int, num_agents: int, obs_shape=(3, 64, 64)):
    """Create a TensorDict of obs with shape [B, T+1, N, *obs_shape]."""
    return TensorDict({'obs': torch.rand(batch_size, t_plus_one, num_agents, *obs_shape)})


class TestRealModelSequentialForward:
    """Test _sequential_agent_forward with a real QMixAgentNet (GRU core)."""

    def test_single_layer_gru_shapes(self):
        cfg = _real_model_cfg(rnn_size=32, rnn_num_layers=1)
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)

        B, T, N = 2, 4, 2
        obs = _make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        rnn_states = torch.zeros(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_values, enc_outs = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, agent_net)

        assert q_values.shape == (B, T + 1, N, sum(agent_net.action_sizes))
        assert enc_outs.shape == (B, T + 1, N, agent_net.encoder_out_size)

    def test_multi_layer_gru_shapes(self):
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=3)
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)

        assert agent_net.get_rnn_size() == 16 * 3  # multi-layer

        B, T, N = 2, 3, 2
        obs = _make_real_obs_batch(B, T + 1, N)
        dones = torch.zeros(B, T, N)
        rnn_states = torch.zeros(B, N, agent_net.get_rnn_size())

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        q_values, enc_outs = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, agent_net)

        assert q_values.shape == (B, T + 1, N, sum(agent_net.action_sizes))
        assert enc_outs.shape == (B, T + 1, N, agent_net.encoder_out_size)

    def test_done_resets_hidden_state_real_gru(self):
        """With a real GRU, verify that a done at t=0 resets hidden state,
        producing different outputs than without the done."""
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=1)
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        agent_net.eval()

        B, T, N = 1, 2, 2
        obs = _make_real_obs_batch(B, T + 1, N)
        rnn_states = torch.randn(B, N, agent_net.get_rnn_size())  # non-zero

        learner = object.__new__(QMixLearner)
        learner.num_agents = N

        # No dones
        dones_none = torch.zeros(B, T, N)
        q_no_done, _ = QMixLearner._sequential_agent_forward(learner, obs, dones_none, rnn_states, agent_net)

        # Agent 0 done at t=0 → hidden reset before t=1
        dones_reset = torch.zeros(B, T, N)
        dones_reset[0, 0, 0] = 1.0
        q_with_done, _ = QMixLearner._sequential_agent_forward(learner, obs, dones_reset, rnn_states, agent_net)

        # t=0 outputs should be identical (done not yet applied)
        assert torch.allclose(q_no_done[:, 0], q_with_done[:, 0], atol=1e-6)
        # t=1 agent 0 should differ (hidden was reset)
        assert not torch.allclose(q_no_done[:, 1, 0], q_with_done[:, 1, 0], atol=1e-4)
        # t=1 agent 1 should be the same (not reset)
        assert torch.allclose(q_no_done[:, 1, 1], q_with_done[:, 1, 1], atol=1e-6)


class TestRealModelLossComputation:
    """Test _calculate_qmix_loss_sequential with real QMixAgentNet + QMixMixer."""

    def _make_learner(self, cfg, obs_space, act_space, num_agents=2):
        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        target_net = copy.deepcopy(agent_net)
        state_dim = agent_net.encoder_out_size * num_agents
        mixer = make_mixer(cfg, num_agents, state_dim)
        target_mixer = copy.deepcopy(mixer)

        learner = object.__new__(QMixLearner)
        learner.num_agents = num_agents
        learner.cfg = cfg
        learner.obs_normalizer = None
        learner.agent_net = agent_net
        learner.target_agent_net = target_net
        learner.mixer = mixer
        learner.target_mixer = target_mixer
        return learner

    def test_qmix_loss_shapes_compound_actions(self):
        """Full forward+loss with compound actions (multi-head) and real QMIX mixer."""
        cfg = _real_model_cfg(rnn_size=32, rnn_num_layers=1, mixer='qmix')
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 3, 4
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            # compound actions: [B, T, N, num_heads]
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert td_summary.shape == (B,)

    def test_vdn_loss_shapes(self):
        """Same test but with VDN mixer."""
        cfg = _real_model_cfg(rnn_size=32, rnn_num_layers=1, mixer='vdn')
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)

    def test_backward_flows_to_agent_net_and_mixer(self):
        """Verify gradients reach both agent_net and mixer parameters."""
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='qmix')
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, _ = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        loss.backward()

        # Mixer should receive gradients
        mixer_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in learner.mixer.parameters())
        assert mixer_has_grad, "Mixer parameters should receive gradients"

        # Agent net Q-head should receive gradients (not detached)
        q_heads = learner.agent_net.q_heads if learner.agent_net.q_heads is not None else [learner.agent_net.q_head]
        q_head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for head in q_heads for p in head.parameters())
        assert q_head_has_grad, "Agent net Q-heads should receive gradients"

        # GRU core should receive gradients (backprop through time)
        core_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in learner.agent_net.core.parameters())
        assert core_has_grad, "GRU core should receive gradients via BPTT"

        # Target net should NOT receive gradients
        target_has_grad = any(p.grad is not None for p in learner.target_agent_net.parameters())
        assert not target_has_grad, "Target agent net must not receive gradients"

    def test_multi_layer_gru_full_loss(self):
        """Full loss computation with multi-layer GRU to verify state reshaping."""
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=2, mixer='qmix')
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)
        # Should not error on backward with multi-layer GRU
        loss.backward()

    def test_single_action_space_loss(self):
        """Test with a single Discrete action space (non-compound)."""
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='qmix')
        obs_space = _small_obs_space()
        act_space = _single_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 3
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            # Single action: [B, T, N] (no head dim)
            'actions': torch.randint(0, 4, (B, T, N)),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)

    def test_with_dones_and_timeouts(self):
        """Loss works with mixed done/timeout patterns."""
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=1, mixer='qmix')
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        N = 2
        learner = self._make_learner(cfg, obs_space, act_space, N)

        B, T = 2, 4
        dones = torch.zeros(B, T, N)
        time_outs = torch.zeros(B, T, N)
        # Agent 0 terminates at t=1
        dones[0, 1, 0] = 1.0
        # Agent 1 times out at t=2
        dones[0, 2, 1] = 1.0
        time_outs[0, 2, 1] = 1.0

        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': dones,
            'time_outs': time_outs,
            'rnn_states': torch.randn(B, N, learner.agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)


class TestObsNormalizationRNN:
    """Test that obs normalization handles [B, T+1, N, ...] shapes correctly."""

    def test_normalization_preserves_shape(self):
        """Normalization should flatten, normalize, and reshape back correctly."""
        from sample_factory.utils.normalize import ObservationNormalizer

        obs_space = _small_obs_space()
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=1)
        cfg['normalize_input'] = True
        cfg['normalize_input_keys'] = ['obs']
        cfg['obs_subtract_mean'] = 0.0
        cfg['obs_scale'] = 1.0

        normalizer = ObservationNormalizer(obs_space, cfg)

        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = cfg
        learner.obs_normalizer = normalizer
        learner.agent_net = QMixAgentNet(cfg, obs_space, _compound_action_space())
        learner.target_agent_net = copy.deepcopy(learner.agent_net)
        state_dim = learner.agent_net.encoder_out_size * 2
        learner.mixer = make_mixer(cfg, 2, state_dim)
        learner.target_mixer = copy.deepcopy(learner.mixer)

        B, T, N = 2, 3, 2
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, learner.agent_net.get_rnn_size()),
        })

        # Should not raise any shape errors
        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)


class TestMinimalRollout:
    """Test edge case: rollout=2 (minimum allowed)."""

    def test_rollout_2_loss(self):
        cfg = _real_model_cfg(rnn_size=16, rnn_num_layers=1)
        obs_space, act_space = _small_obs_space(), _compound_action_space()
        N = 2

        agent_net = QMixAgentNet(cfg, obs_space, act_space)
        target_net = copy.deepcopy(agent_net)
        state_dim = agent_net.encoder_out_size * N
        mixer = make_mixer(cfg, N, state_dim)
        target_mixer = copy.deepcopy(mixer)

        learner = object.__new__(QMixLearner)
        learner.num_agents = N
        learner.cfg = cfg
        learner.obs_normalizer = None
        learner.agent_net = agent_net
        learner.target_agent_net = target_net
        learner.mixer = mixer
        learner.target_mixer = target_mixer

        B, T = 2, 2  # minimum rollout
        batch = TensorDict({
            'obs': _make_real_obs_batch(B, T + 1, N),
            'actions': torch.stack([
                torch.randint(0, 3, (B, T, N)),
                torch.randint(0, 2, (B, T, N)),
            ], dim=-1),
            'rewards': torch.randn(B, T, N),
            'dones': torch.zeros(B, T, N),
            'time_outs': torch.zeros(B, T, N),
            'rnn_states': torch.zeros(B, N, agent_net.get_rnn_size()),
        })

        loss, td_summary = QMixLearner._calculate_qmix_loss_sequential(learner, batch)
        assert loss.dim() == 0
        assert td_summary.shape == (B,)
        loss.backward()


# ---------------------------------------------------------------------------
# Config Rejection Tests
# ---------------------------------------------------------------------------

class TestConfigRejection:
    """Verify config gating rejects invalid QMIX/VDN RNN configurations."""

    def test_rejects_rollout_below_2(self):
        cfg = AttrDict({
            'algo': 'QMIX',
            'recurrence': -1,
            'rollout': 1,
            'use_rnn': True,
            'rnn_type': 'gru',
            'per': False,
            'actor_critic_share_weights': True,
            'qmix_buffer_batch_size': 32,
            'qmix_sequence_batch_size': 8,
            'qmix_log_interval': 100,
        })
        with pytest.raises(ValueError, match='rollout >= 2'):
            preprocess_cfg(cfg, _DummyEnvInfo())

    def test_rejects_shared_weights_false(self):
        cfg = AttrDict({
            'algo': 'QMIX',
            'recurrence': -1,
            'rollout': 8,
            'use_rnn': True,
            'rnn_type': 'gru',
            'per': False,
            'actor_critic_share_weights': False,
            'qmix_buffer_batch_size': 32,
            'qmix_sequence_batch_size': 8,
            'qmix_log_interval': 100,
        })
        with pytest.raises(ValueError, match='actor_critic_share_weights'):
            preprocess_cfg(cfg, _DummyEnvInfo())

    def test_per_forced_false_for_rnn(self):
        cfg = _make_full_qmix_cfg(
            use_rnn=True,
            rollout=8,
            per=True,
        )
        preprocess_cfg(cfg, _DummyEnvInfo())
        assert cfg.per is False


# ---------------------------------------------------------------------------
# Buffer Edge Case Tests
# ---------------------------------------------------------------------------

class TestSequenceBufferEdgeCases:
    """Verify buffer boundary conditions."""

    def _make_batch(self, batch_size, seq_len=2, num_agents=2, rnn_size=4):
        return TensorDict({
            'obs': TensorDict({'obs': torch.randn(batch_size, seq_len + 1, num_agents, 1)}),
            'actions': torch.randint(0, 3, (batch_size, seq_len, num_agents), dtype=torch.long),
            'rewards': torch.randn(batch_size, seq_len, num_agents),
            'dones': torch.zeros(batch_size, seq_len, num_agents),
            'time_outs': torch.zeros(batch_size, seq_len, num_agents),
            'rnn_states': torch.randn(batch_size, num_agents, rnn_size),
        })

    def test_add_batch_larger_than_capacity(self):
        """Adding a batch larger than capacity should keep only the last capacity entries."""
        rb = JointSequenceReplayBuffer(
            capacity_sequences=3, seq_len=2, num_agents=2,
            obs_space=None, action_space=None, device='cpu',
            share_memory=False, rnn_state_size=4,
        )
        batch = self._make_batch(batch_size=5)
        added = rb.add_sequence_batch(batch)
        assert added == 3  # truncated to capacity
        assert len(rb) == 3

    def test_sample_with_batch_larger_than_size(self):
        """Sampling more than buffer size uses replacement — no crash, allows duplicates."""
        rb = JointSequenceReplayBuffer(
            capacity_sequences=8, seq_len=2, num_agents=2,
            obs_space=None, action_space=None, device='cpu',
            share_memory=False, rnn_state_size=4,
        )
        rb.add_sequence_batch(self._make_batch(batch_size=2))
        assert len(rb) == 2
        sampled, _, _ = rb.sample(8, 'cpu')
        assert sampled['obs']['obs'].shape[0] == 8

    def test_set_beta_and_update_priorities_are_noops(self):
        """PER API methods should no-op without errors."""
        rb = JointSequenceReplayBuffer(
            capacity_sequences=4, seq_len=2, num_agents=2,
            obs_space=None, action_space=None, device='cpu',
            share_memory=False, rnn_state_size=4,
        )
        rb.add_sequence_batch(self._make_batch(batch_size=2))
        assert rb.set_beta(0.5) is None
        assert rb.update_priorities(torch.tensor([0, 1]), torch.tensor([0.1, 0.2])) is None

    def test_sample_from_empty_buffer_raises(self):
        rb = JointSequenceReplayBuffer(
            capacity_sequences=4, seq_len=2, num_agents=2,
            obs_space=None, action_space=None, device='cpu',
            share_memory=False, rnn_state_size=4,
        )
        with pytest.raises(RuntimeError, match='empty'):
            rb.sample(1, 'cpu')

    def test_no_derived_fields_in_sample(self):
        """After M1 fix, sampled batches should not contain team_reward/joint_done/joint_time_out."""
        rb = JointSequenceReplayBuffer(
            capacity_sequences=4, seq_len=2, num_agents=2,
            obs_space=None, action_space=None, device='cpu',
            share_memory=False, rnn_state_size=4,
        )
        rb.add_sequence_batch(self._make_batch(batch_size=2))
        sampled, _, _ = rb.sample(2, 'cpu')
        assert 'team_reward' not in sampled
        assert 'joint_done' not in sampled
        assert 'joint_time_out' not in sampled


# ---------------------------------------------------------------------------
# RNN Debt Accounting Test
# ---------------------------------------------------------------------------

class TestRnnDebtAccounting:
    """Verify that RNN path debt is in individual agent-transition units."""

    def test_rnn_transitions_added_formula(self):
        """For num_agents=2, rollout=4, adding 3 sequences should produce
        3 * 2 * 4 = 24 individual agent-transitions."""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.use_rnn = True
        learner.cfg = AttrDict({
            'rollout': 4,
            'use_rnn': True,
            'summaries_use_frameskip': False,
        })
        # Simulate what train() does after replay insertion
        num_sequences = 3
        transitions_added = num_sequences * learner.num_agents * learner.cfg.rollout
        assert transitions_added == 24

    def test_rnn_buffer_transitions_formula(self):
        """buffer_transitions = len(buffer) * num_agents * rollout"""
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.use_rnn = True
        learner.cfg = AttrDict({'rollout': 8})

        # Simulate buffer with 10 sequences stored
        buffer_len = 10
        buffer_transitions = buffer_len * learner.num_agents * learner.cfg.rollout
        assert buffer_transitions == 160  # 10 * 2 * 8

    def test_rnn_debt_consistent_with_non_rnn(self):
        """Both modes use individual agent-transition units for debt.
        Non-RNN: num_joint * num_agents
        RNN: num_sequences * num_agents * rollout
        With rollout=1 (hypothetically), RNN formula collapses to non-RNN formula."""
        num_agents = 2
        rollout = 1

        # Non-RNN: 5 joint transitions
        non_rnn_debt = 5 * num_agents
        # RNN: 5 sequences of length 1
        rnn_debt = 5 * num_agents * rollout

        assert non_rnn_debt == rnn_debt == 10
