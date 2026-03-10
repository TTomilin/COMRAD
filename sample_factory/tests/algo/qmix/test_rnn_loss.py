from __future__ import annotations

import pytest
import torch
from torch import nn

from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_sequence_replay_buffer import JointSequenceReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict

from .conftest import AgentRnnStub, SumMixer, make_qmix_cfg, make_spaces, make_sequence_batch


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
    learner.agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
    learner.target_agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
    learner.mixer = SumMixer()
    learner.target_mixer = SumMixer()

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


class TestEffectiveDoneMixedOutcomes:

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
        learner.agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.mixer = SumMixer()
        learner.target_mixer = SumMixer()

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
        # Hand-traced values through AgentRnnStub + SumMixer:
        #   q_tot_curr = [2.0, 2.0], target = [2.0, 5.96]
        #   td_error = [0.0, 3.96], Huber = [0.0, 3.46], mean = 1.73
        # With the old buggy formula (joint_done*(1-joint_time_out)),
        #   effective_done would be 0 → loss ≈ 2.47 (different!)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(1.73, abs=0.01)

    def test_sequential_loss_all_timeout_should_bootstrap(self):
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.mixer = SumMixer()
        learner.target_mixer = SumMixer()

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
        learner.mixer = SumMixer()
        learner.target_mixer = SumMixer()

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

    def test_sequential_td_target_no_done(self):
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.5, # easy to verify manually
            'double_dqn': False, # simpler: target net greedy
            'q_value_clamp': 0, # disabled
            'use_huber_loss': False, # MSE
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

            def encode(self, obs):
                if isinstance(obs, dict) and "obs" in obs:
                    val = obs["obs"]
                else:
                    val = obs if torch.is_tensor(obs) else list(obs.values())[0]
                return torch.zeros(val.shape[0], self.enc_dim)

            def forward_head(self, encoded, rnn_states):
                batch = rnn_states.shape[0]
                q = torch.full((batch, 3), self.q_val)
                new_rnn = rnn_states.clone()
                return q, new_rnn

            def forward_decomposed(self, obs, rnn_states):
                enc = self.encode(obs)
                q, new_rnn = self.forward_head(enc, rnn_states)
                return q, new_rnn, enc

            def get_q_for_actions(self, q, a):
                return q.gather(-1, a.unsqueeze(-1)).squeeze(-1)

        learner.agent_net = _ConstQStub(q_val=2.0)
        learner.target_agent_net = _ConstQStub(q_val=5.0)
        learner.mixer = SumMixer()
        learner.target_mixer = SumMixer()

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


class TestGradientFlow:

    def test_backward_does_not_error(self):
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)

        # Need a parameterized mixer for backward() to work (no-param graph has no grad_fn)
        class _TrivialParamMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
            def forward(self, agent_qs, state):
                return agent_qs.sum(dim=-1) * self.scale

        learner.mixer = _TrivialParamMixer()
        learner.target_mixer = SumMixer()

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
        learner = object.__new__(QMixLearner)
        learner.num_agents = 2
        learner.cfg = AttrDict({
            'gamma': 0.99,
            'double_dqn': True,
            'q_value_clamp': 100.0,
            'use_huber_loss': True,
        })
        learner.obs_normalizer = None
        learner.agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)
        learner.target_agent_net = AgentRnnStub(rnn_size=2, num_actions=3, enc_dim=2)

        # Use a parameterized mixer so we can check gradients
        class _LinearMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2))
            def forward(self, agent_qs, state):
                return (agent_qs * self.weight).sum(dim=-1)

        learner.mixer = _LinearMixer()
        learner.target_mixer = SumMixer()

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
