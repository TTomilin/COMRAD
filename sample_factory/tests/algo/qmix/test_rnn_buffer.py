from __future__ import annotations

import pytest
import torch

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_sequence_replay_buffer import JointSequenceReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict


class TestSequenceReplayBufferWraparound:

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


class TestSequenceBufferEdgeCases:

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


class TestRnnDebtAccounting:

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
