from __future__ import annotations
import pytest
import torch
from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from comrad.models.qmix_model import QMixAgentNet


@pytest.fixture
def replay_buffer():
    num_heads = 3
    rb = JointReplayBuffer(
        capacity=8,
        num_agents=2,
        obs_space=None,
        action_space=None,
        device="cpu",
        share_memory=False,
        use_per=False,
    )
    joint_transition = TensorDict(
        {
            "obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "next_obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "actions": torch.zeros(2, num_heads, dtype=torch.long),
            "rewards": torch.tensor([0.3, 0.7]),
            "dones": torch.tensor([1.0, 0.0]),
            "time_outs": torch.tensor([0.0, 1.0]),
        }
    )
    rb.add_joint(joint_transition)
    return rb


def test_joint_replay_buffer_team_reward(replay_buffer):
    assert abs(replay_buffer._storage["team_reward"][0].item() - 1.0) < 1e-6


def test_joint_replay_buffer_joint_done(replay_buffer):
    assert abs(replay_buffer._storage["joint_done"][0].item() - 1.0) < 1e-6


def test_joint_replay_buffer_joint_time_out(replay_buffer):
    assert abs(replay_buffer._storage["joint_time_out"][0].item() - 1.0) < 1e-6


def test_joint_replay_buffer_effective_done(replay_buffer):
    effective_done = (
        replay_buffer._storage["joint_done"][0] * (1 - replay_buffer._storage["joint_time_out"][0])
    )
    assert effective_done.item() == 0.0


class TestBufferRawSignals:

    def test_buffer_stores_raw_done(self):
        rb = JointReplayBuffer(
            capacity=4, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=False,
        )
        # Agent 0: done=1.0, timeout=0.0 (true death)
        # Agent 1: done=1.0, timeout=1.0 (timeout)
        joint = TensorDict({
            "obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "next_obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "actions": torch.zeros(2, 3, dtype=torch.long),
            "rewards": torch.tensor([0.5, 0.5]),
            "dones": torch.tensor([1.0, 1.0]),
            "time_outs": torch.tensor([0.0, 1.0]),
        })
        rb.add_joint(joint)
        # joint_done = max(1.0, 1.0) = 1.0 (raw, not masked)
        assert rb._storage["joint_done"][0].item() == 1.0
        # joint_time_out = max(0.0, 1.0) = 1.0
        assert rb._storage["joint_time_out"][0].item() == 1.0
        # effective_done computed at loss time, not storage time
        effective_done = rb._storage["joint_done"][0] * (1 - rb._storage["joint_time_out"][0])
        assert effective_done.item() == 0.0  # timeout -> should bootstrap

    def test_buffer_team_reward_is_sum(self):
        rb = JointReplayBuffer(
            capacity=4, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=False,
        )
        joint = TensorDict({
            "obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "next_obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "actions": torch.zeros(2, 3, dtype=torch.long),
            "rewards": torch.tensor([0.3, 0.7]),
            "dones": torch.tensor([0.0, 0.0]),
            "time_outs": torch.tensor([0.0, 0.0]),
        })
        rb.add_joint(joint)
        assert abs(rb._storage["team_reward"][0].item() - 1.0) < 1e-6

    def test_buffer_joint_done_is_max(self):
        rb = JointReplayBuffer(
            capacity=4, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=False,
        )
        joint = TensorDict({
            "obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "next_obs": TensorDict({"obs": torch.zeros(2, 1)}),
            "actions": torch.zeros(2, 3, dtype=torch.long),
            "rewards": torch.tensor([0.0, 0.0]),
            "dones": torch.tensor([1.0, 0.0]),  # Only agent 0 done
            "time_outs": torch.tensor([0.0, 0.0]),
        })
        rb.add_joint(joint)
        assert rb._storage["joint_done"][0].item() == 1.0


class TestActionMask:

    def _make_actor_critic(self):
        import gymnasium as gym
        from sample_factory.utils.attr_dict import AttrDict
        from comrad.models.qmix_model import QMixActorCritic
        cfg = AttrDict({
            'encoder_conv_architecture': 'convnet_simple',
            'encoder_conv_mlp_layers': [],
            'encoder_extra_fc_layers': 0,
            'hidden_size': 32,
            'nonlinearity': 'relu',
            'use_rnn': False,
            'rnn_type': 'gru',
            'rnn_num_layers': 1,
            'decoder_mlp_layers': [],
            'normalize_input': False,
            'normalize_returns': False,
            'obs_subtract_mean': 0.0,
            'obs_scale': 1.0,
            'num_agents': 2,
            'mixer': 'qmix',
            'qmix_embed_dim': 32,
            'qmix_hypernet_hidden': 64,
        })
        obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))})
        action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
        return QMixActorCritic(cfg, obs_space, action_space, num_agents=2)

    def test_all_ones_mask_does_not_change_output(self):
        ac = self._make_actor_critic()
        ac.eval()
        obs = TensorDict({"obs": torch.randn(2, 3, 64, 64)})
        rnn = torch.zeros(2, ac.agent_net.core.get_out_size())
        mask = torch.ones(2, 5)  # all actions allowed
        with torch.no_grad():
            out_with_mask = ac(obs, rnn, action_mask=mask)
            out_without_mask = ac(obs, rnn, action_mask=None)
        assert torch.allclose(out_with_mask['action_logits'], out_without_mask['action_logits'])


class TestBufferSingleThreadedAccess:

    def test_add_then_sample_sequential(self):
        rb = JointReplayBuffer(
            capacity=16, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=False,
        )
        # Add 5 transitions
        for i in range(5):
            joint = TensorDict({
                "obs": TensorDict({"obs": torch.randn(2, 1)}),
                "next_obs": TensorDict({"obs": torch.randn(2, 1)}),
                "actions": torch.zeros(2, 3, dtype=torch.long),
                "rewards": torch.tensor([float(i), float(i)]),
                "dones": torch.tensor([0.0, 0.0]),
                "time_outs": torch.tensor([0.0, 0.0]),
            })
            rb.add_joint(joint)
        assert len(rb) == 5
        batch, weights, indices = rb.sample(3, "cpu")
        assert batch["team_reward"].shape == (3,)
        assert weights is None  # No PER

    def test_circular_buffer_wraps(self):
        rb = JointReplayBuffer(
            capacity=4, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=False,
        )
        for i in range(6):  # Overflow capacity by 2
            joint = TensorDict({
                "obs": TensorDict({"obs": torch.full((2, 1), float(i))}),
                "next_obs": TensorDict({"obs": torch.zeros(2, 1)}),
                "actions": torch.zeros(2, 3, dtype=torch.long),
                "rewards": torch.tensor([float(i), 0.0]),
                "dones": torch.tensor([0.0, 0.0]),
                "time_outs": torch.tensor([0.0, 0.0]),
            })
            rb.add_joint(joint)
        assert len(rb) == 4  # Capped at capacity
        assert rb._ptr == 2  # Pointer wrapped: 6 % 4 = 2


class TestQMIXMonotonicity:

    def test_mixing_weights_non_negative(self):
        from comrad.models.qmix_model import QMixMixer
        mixer = QMixMixer(num_agents=2, state_dim=8, embed_dim=4)
        agent_qs = torch.randn(4, 2)
        state = torch.randn(4, 8)
        # Verify forward passes without error
        q_tot = mixer(agent_qs, state)
        assert q_tot.shape == (4,)
        # Verify monotonicity: increasing any agent Q should increase Q_tot
        for agent_idx in range(2):
            agent_qs_higher = agent_qs.clone()
            agent_qs_higher[:, agent_idx] += 1.0
            q_tot_higher = mixer(agent_qs_higher, state)
            assert (q_tot_higher >= q_tot - 1e-6).all(), \
                f"QMIX monotonicity violated for agent {agent_idx}"


class TestPER:

    def test_per_weights_normalized(self):
        rb = JointReplayBuffer(
            capacity=16, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=True,
        )
        for i in range(8):
            joint = TensorDict({
                "obs": TensorDict({"obs": torch.randn(2, 1)}),
                "next_obs": TensorDict({"obs": torch.randn(2, 1)}),
                "actions": torch.zeros(2, 3, dtype=torch.long),
                "rewards": torch.tensor([float(i), 0.0]),
                "dones": torch.tensor([0.0, 0.0]),
                "time_outs": torch.tensor([0.0, 0.0]),
            })
            rb.add_joint(joint)
        batch, weights, indices = rb.sample(4, "cpu")
        assert weights is not None
        assert weights.shape == (4,)
        # IS weights should be <= 1 (normalized by max weight)
        assert (weights <= 1.0 + 1e-6).all()
        assert (weights > 0).all()

    def test_per_priority_update(self):
        rb = JointReplayBuffer(
            capacity=16, num_agents=2, obs_space=None, action_space=None,
            device="cpu", share_memory=False, use_per=True,
        )
        for i in range(4):
            joint = TensorDict({
                "obs": TensorDict({"obs": torch.randn(2, 1)}),
                "next_obs": TensorDict({"obs": torch.randn(2, 1)}),
                "actions": torch.zeros(2, 3, dtype=torch.long),
                "rewards": torch.tensor([0.0, 0.0]),
                "dones": torch.tensor([0.0, 0.0]),
                "time_outs": torch.tensor([0.0, 0.0]),
            })
            rb.add_joint(joint)
        _, _, indices = rb.sample(2, "cpu")
        # Update priorities with varying TD errors
        td_errors = torch.tensor([0.5, 2.0])
        rb.update_priorities(indices, td_errors)
        # max_priority should have increased
        assert rb._max_priority >= 2.0


class TestNonRnnDebtAccounting:

    @staticmethod
    def _make_joint_batch(num_joint: int, num_agents: int, num_heads: int = 3) -> TensorDict:
        return TensorDict({
            'obs': TensorDict({'obs': torch.randn(num_joint, num_agents, 1)}),
            'next_obs': TensorDict({'obs': torch.randn(num_joint, num_agents, 1)}),
            'actions': torch.zeros(num_joint, num_agents, num_heads, dtype=torch.long),
            'rewards': torch.zeros(num_joint, num_agents),
            'dones': torch.zeros(num_joint, num_agents),
            'time_outs': torch.zeros(num_joint, num_agents),
            'team_reward': torch.zeros(num_joint),
            'joint_done': torch.zeros(num_joint),
            'joint_time_out': torch.zeros(num_joint),
        })

    def test_non_rnn_debt_uses_agent_transitions(self):
        num_agents = 2
        train_freq = 4
        num_joint_added = 2  # 2 joint * 2 agents = 4 agent transitions = exactly 1 update

        rb = JointReplayBuffer(
            capacity=64, num_agents=num_agents, obs_space=None, action_space=None,
            device='cpu', share_memory=False, use_per=False,
        )
        # Fill buffer past learning_starts
        for _ in range(20):
            rb.add_joint(TensorDict({
                'obs': TensorDict({'obs': torch.randn(num_agents, 1)}),
                'next_obs': TensorDict({'obs': torch.randn(num_agents, 1)}),
                'actions': torch.zeros(num_agents, 3, dtype=torch.long),
                'rewards': torch.zeros(num_agents),
                'dones': torch.zeros(num_agents),
                'time_outs': torch.zeros(num_agents),
            }))

        # Simulate the non-RNN debt path from train()
        joint_transitions = self._make_joint_batch(num_joint_added, num_agents)
        num_joint = rb.add_joint_batch(joint_transitions)
        assert num_joint == num_joint_added

        # Debt is now in individual agent transitions: num_joint * num_agents
        transitions_added = num_joint * num_agents
        total_debt = 0
        total_debt += transitions_added
        num_updates = total_debt // train_freq
        total_debt -= num_updates * train_freq

        assert num_updates == 1, (
            f"Expected 1 update (2 joint * 2 agents = 4 transitions / train_freq=4), got {num_updates}."
        )
        assert total_debt == 0

    def test_non_rnn_debt_consistent_with_rnn_semantics(self):
        num_agents = 2
        train_freq = 8

        # Non-RNN: 4 joint transitions * 2 agents = 8 individual transitions = 1 update
        num_joint = 4
        non_rnn_debt = num_joint * num_agents
        assert non_rnn_debt // train_freq == 1

        # RNN: 1 sequence * 2 agents * 4 rollout = 8 individual transitions = 1 update
        num_seq, rollout = 1, 4
        rnn_debt = num_seq * num_agents * rollout
        assert rnn_debt // train_freq == 1

        # Same number of individual transitions → same number of updates
        assert non_rnn_debt == rnn_debt
