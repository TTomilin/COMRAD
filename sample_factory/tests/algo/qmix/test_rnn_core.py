from __future__ import annotations

import pytest
import torch

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import preprocess_cfg
from sample_factory.utils.attr_dict import AttrDict
from comrad.models.qmix_model import QMixAgentNet

from .conftest import (
    AgentRnnStub,
    DummyEnvInfo,
    SumMixer,
    make_full_qmix_cfg,
    make_qmix_cfg,
    make_sequence_batch,
    make_spaces,
)


def test_get_rnn_size_gru_single_layer():
    cfg = make_qmix_cfg(use_rnn=True, rnn_size=32, rnn_num_layers=1)
    obs_space, action_space = make_spaces()
    net = QMixAgentNet(cfg, obs_space, action_space)
    assert net.get_rnn_size() == 32


def test_get_rnn_size_gru_multi_layer():
    cfg = make_qmix_cfg(use_rnn=True, rnn_size=32, rnn_num_layers=3)
    obs_space, action_space = make_spaces()
    net = QMixAgentNet(cfg, obs_space, action_space)
    assert net.get_rnn_size() == 96


def test_prepare_joint_sequences_shapes():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = make_sequence_batch()
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
    learner.agent_net = AgentRnnStub(rnn_size=4)
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

    q_values, enc = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, AgentRnnStub())
    assert q_values.shape == (batch_size, rollout + 1, 2, 3)
    assert enc.shape == (batch_size, rollout + 1, 2, 2)


def test_done_masking_resets_only_done_agents():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2

    obs = TensorDict({'obs': torch.zeros(1, 2, 2, 1)})
    dones = torch.tensor([[[1.0, 0.0]]])
    rnn_states = torch.zeros(1, 2, 2)

    q_values, _ = QMixLearner._sequential_agent_forward(learner, obs, dones, rnn_states, AgentRnnStub())
    # t=1: agent 0 reset before forward -> q=1, agent 1 not reset -> q=2
    assert q_values[0, 1, 0, 0].item() == pytest.approx(1.0)
    assert q_values[0, 1, 1, 0].item() == pytest.approx(2.0)


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
    learner.agent_net = AgentRnnStub(rnn_size=4)
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
    learner.agent_net = AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = make_sequence_batch(rollout=3, rnn_size=4)
    batch['rnn_states'] = torch.randn(4, 4)

    with pytest.raises(ValueError, match='Unexpected rnn_states shape'):
        QMixLearner._prepare_joint_sequences(learner, batch)


def test_prepare_joint_sequences_rejects_bad_rnn_time_dim():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = make_sequence_batch(rollout=3, rnn_size=4)
    batch['rnn_states'] = torch.randn(4, 3, 4)

    with pytest.raises(ValueError, match='Expected rnn_states time dim'):
        QMixLearner._prepare_joint_sequences(learner, batch)


def test_prepare_joint_sequences_rejects_bad_rnn_feature_dim():
    learner = object.__new__(QMixLearner)
    learner.num_agents = 2
    learner.cfg = AttrDict({'rollout': 3})
    learner.agent_net = AgentRnnStub(rnn_size=4)
    learner._invalid_sequence_groups = 0

    batch = make_sequence_batch(rollout=3, rnn_size=4)
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
        preprocess_cfg(cfg, DummyEnvInfo())


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
        preprocess_cfg(cfg, DummyEnvInfo())


class TestLearningStartsEffectiveCapacity:

    def test_rnn_floor_division_caps_learning_starts(self):
        """replay_buffer_size=100, num_agents=3, rollout=11 → capacity=3 seqs → 99 transitions.
        learning_starts=100 should be capped since 99 < 100.
        """
        cfg = make_full_qmix_cfg(
            use_rnn=True,
            rollout=11,
            num_agents=3,
            replay_buffer_size=100,
            learning_starts=100,
        )
        preprocess_cfg(cfg, DummyEnvInfo(num_agents=3))
        # effective_capacity = (100 // (3*11)) * (3*11) = 3 * 33 = 99
        # learning_starts=100 >= 99, so capped to max(1, 99 // 2) = 49
        assert cfg.learning_starts == 49

    def test_rnn_exact_divisible_no_cap(self):
        cfg = make_full_qmix_cfg(
            use_rnn=True,
            rollout=10,
            num_agents=2,
            replay_buffer_size=10000,
            learning_starts=5000,
        )
        preprocess_cfg(cfg, DummyEnvInfo(num_agents=2))
        # effective_capacity = (10000 // 20) * 20 = 10000
        # learning_starts=5000 < 10000 → not capped
        assert cfg.learning_starts == 5000

    def test_non_rnn_floor_division_caps_learning_starts(self):
        """replay_buffer_size=101, num_agents=2 → capacity=50 joints → 100 transitions.
        learning_starts=101 should be capped.
        """
        cfg = make_full_qmix_cfg(
            use_rnn=False,
            num_agents=2,
            replay_buffer_size=101,
            learning_starts=101,
        )
        preprocess_cfg(cfg, DummyEnvInfo(num_agents=2))
        # effective_capacity = (101 // 2) * 2 = 100
        # learning_starts=101 >= 100, so capped to max(1, 100 // 2) = 50
        assert cfg.learning_starts == 50

    def test_equality_case_is_capped(self):
        cfg = make_full_qmix_cfg(
            use_rnn=False,
            num_agents=2,
            replay_buffer_size=100,
            learning_starts=100,
        )
        preprocess_cfg(cfg, DummyEnvInfo(num_agents=2))
        # effective_capacity = (100 // 2) * 2 = 100
        # learning_starts=100 >= 100, so capped to max(1, 100 // 2) = 50
        assert cfg.learning_starts == 50

    def test_safely_below_capacity_not_capped(self):
        cfg = make_full_qmix_cfg(
            use_rnn=True,
            rollout=16,
            num_agents=2,
            replay_buffer_size=100000,
            learning_starts=5000,
        )
        preprocess_cfg(cfg, DummyEnvInfo(num_agents=2))
        assert cfg.learning_starts == 5000


class TestConfigRejection:

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
            preprocess_cfg(cfg, DummyEnvInfo())

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
            preprocess_cfg(cfg, DummyEnvInfo())

    def test_per_forced_false_for_rnn(self):
        cfg = make_full_qmix_cfg(
            use_rnn=True,
            rollout=8,
            per=True,
        )
        preprocess_cfg(cfg, DummyEnvInfo())
        assert cfg.per is False
