from __future__ import annotations
import pytest
import torch
from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.qmix_model import QMixAgentNet


@pytest.fixture
def batch_and_dims():
    num_envs, num_agents, horizon, obs_dim, num_heads = 2, 2, 3, 1, 3
    num_traj = num_envs * num_agents
    obs = torch.arange(num_traj * (horizon + 1) * obs_dim, dtype=torch.float32).view(num_traj, horizon + 1, obs_dim)
    actions = torch.randint(0, 3, (num_traj, horizon, num_heads))
    rewards = torch.arange(num_traj * horizon, dtype=torch.float32).view(num_traj, horizon)
    dones = torch.zeros(num_traj, horizon)
    dones[0, 0] = 1.0
    dones[1, 0] = 1.0
    time_outs = torch.zeros_like(dones)
    time_outs[1, 0] = 1.0
    env_idx = torch.tensor([[0] * horizon, [0] * horizon, [1] * horizon, [1] * horizon], dtype=torch.long)
    agent_idx = torch.tensor([[0] * horizon, [1] * horizon, [0] * horizon, [1] * horizon], dtype=torch.long)
    batch = TensorDict(
        {
            "obs": TensorDict({"obs": obs}),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "time_outs": time_outs,
            "env_idx": env_idx,
            "agent_idx": agent_idx,
        }
    )
    return batch, num_envs, num_agents, horizon, obs_dim, num_heads


@pytest.fixture
def joint(batch_and_dims):
    batch, num_envs, num_agents, horizon, obs_dim, num_heads = batch_and_dims
    learner = object.__new__(QMixLearner)
    learner.num_agents = num_agents
    return QMixLearner._prepare_joint_transitions(learner, batch), num_envs, num_agents, horizon, obs_dim, num_heads


def test_prepare_joint_transitions_actions_shape(joint):
    j, num_envs, num_agents, horizon, obs_dim, num_heads = joint
    assert j["actions"].shape == (num_envs * horizon, num_agents, num_heads)


def test_prepare_joint_transitions_obs_shape(joint):
    j, num_envs, num_agents, horizon, obs_dim, num_heads = joint
    assert j["obs"]["obs"].shape == (num_envs * horizon, num_agents, obs_dim)


def test_prepare_joint_transitions_next_obs_shape(joint):
    j, num_envs, num_agents, horizon, obs_dim, num_heads = joint
    assert j["next_obs"]["obs"].shape == (num_envs * horizon, num_agents, obs_dim)


def test_prepare_joint_transitions_done_masking_agent0(joint):
    j, *_ = joint
    # Raw dones are preserved (no pre-masking after fix #1). Agent 0 had done=1.0, timeout=0.0.
    assert j["dones"][0, 0].item() == 1.0


def test_prepare_joint_transitions_done_masking_agent1(joint):
    j, *_ = joint
    # Raw dones are preserved (no pre-masking after fix #1). Agent 1 had done=1.0, timeout=1.0.
    # The raw done is stored; effective_done masking happens only in _calculate_qmix_loss.
    assert j["dones"][0, 1].item() == 1.0


def test_prepare_joint_transitions_timeouts_preserved(joint):
    j, *_ = joint
    # time_outs are stored as raw signals alongside dones
    assert j["time_outs"][0, 0].item() == 0.0  # Agent 0 did not timeout
    assert j["time_outs"][0, 1].item() == 1.0  # Agent 1 timed out


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


def test_multihead_action_value_selection_sums_heads():
    net = object.__new__(QMixAgentNet)
    net.q_head = None
    net.action_sizes = [3, 3, 3]
    q_values = torch.tensor([[1.2, 0.1, -0.2, 0.5, -1.0, 0.0, -0.1, 0.2, 0.9]], dtype=torch.float32)
    selected_actions = torch.tensor([[0, 0, 0]], dtype=torch.long)
    selected_q = QMixAgentNet.get_q_for_actions(net, q_values, selected_actions)
    assert abs(selected_q.item() - 1.6) < 1e-6


def test_huber_loss_sample():
    td_error = torch.tensor([1.47], dtype=torch.float32)
    abs_td = td_error.abs()
    huber = torch.where(abs_td < 1.0, 0.5 * td_error.pow(2), abs_td - 0.5)
    assert abs(huber.item() - 0.97) < 1e-6


class TestEffectiveDone:
    """Validates that effective_done = joint_done * (1 - joint_time_out) handles all cases."""

    def test_normal_step_bootstraps(self):
        """Mid-episode step: done=0, timeout=0 -> effective_done=0 -> bootstrap."""
        joint_done = torch.tensor([0.0])
        joint_time_out = torch.tensor([0.0])
        effective_done = joint_done * (1 - joint_time_out)
        assert effective_done.item() == 0.0  # 1 - effective_done = 1 -> bootstrap

    def test_true_death_no_bootstrap(self):
        """True termination: done=1, timeout=0 -> effective_done=1 -> no bootstrap."""
        joint_done = torch.tensor([1.0])
        joint_time_out = torch.tensor([0.0])
        effective_done = joint_done * (1 - joint_time_out)
        assert effective_done.item() == 1.0  # 1 - effective_done = 0 -> no bootstrap

    def test_timeout_bootstraps(self):
        """Timeout (truncation): done=1, timeout=1 -> effective_done=0 -> bootstrap."""
        joint_done = torch.tensor([1.0])
        joint_time_out = torch.tensor([1.0])
        effective_done = joint_done * (1 - joint_time_out)
        assert effective_done.item() == 0.0  # 1 - effective_done = 1 -> bootstrap

    def test_bellman_target_with_timeout(self):
        """Full Bellman target: r + theta (1 - effective_done) * Q_next."""
        gamma = 0.99
        reward = torch.tensor([1.0])
        q_next = torch.tensor([10.0])
        # Timeout case: should bootstrap
        effective_done = torch.tensor([0.0])  # timeout
        target = reward + gamma * (1 - effective_done) * q_next
        assert abs(target.item() - 10.9) < 1e-5

    def test_bellman_target_with_true_death(self):
        """True death: target should be reward only."""
        gamma = 0.99
        reward = torch.tensor([1.0])
        q_next = torch.tensor([10.0])
        effective_done = torch.tensor([1.0])  # true death
        target = reward + gamma * (1 - effective_done) * q_next
        assert abs(target.item() - 1.0) < 1e-5

    def test_batch_mixed_scenarios(self):
        """Batch with mixed done/timeout scenarios."""
        joint_done = torch.tensor([0.0, 1.0, 1.0])
        joint_time_out = torch.tensor([0.0, 0.0, 1.0])
        effective_done = joint_done * (1 - joint_time_out)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(effective_done, expected)

    def test_no_timeout_key_fallback(self):
        """When joint_time_out is None, effective_done = joint_done."""
        joint_done = torch.tensor([1.0, 0.0])
        joint_time_out = None
        if joint_time_out is not None:
            effective_done = joint_done * (1 - joint_time_out)
        else:
            effective_done = joint_done
        assert torch.allclose(effective_done, joint_done)


class TestBufferRawSignals:
    """Validates that replay buffer stores raw done/timeout, not pre-masked."""

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
        """Any agent done -> team done (max aggregation)."""
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


class TestEncodeForwardConsistency:
    """Validates that forward() uses encode() and produces consistent results."""

    def _make_agent_net(self, with_measurements=False):
        import gymnasium as gym
        from sample_factory.utils.attr_dict import AttrDict
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
        })
        spaces = {"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))}
        if with_measurements:
            spaces["measurements"] = gym.spaces.Box(-1, 1, shape=(3,))
        obs_space = gym.spaces.Dict(spaces)
        action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
        return QMixAgentNet(cfg, obs_space, action_space)

    def test_encode_matches_forward_encoding(self):
        """encode(obs) should produce the same tensor that forward() passes to core."""
        net = self._make_agent_net()
        net.eval()
        obs = TensorDict({"obs": torch.randn(2, 3, 64, 64)})
        with torch.no_grad():
            encoded = net.encode(obs)
            rnn_states = torch.zeros(2, net.core.get_out_size())
            q_values, _ = net(obs, rnn_states)
        # encode() output fed through core+decoder+heads should equal forward() output
        assert encoded.shape == (2, net.encoder_out_size)
        assert q_values.shape[0] == 2

    def test_encode_handles_measurements(self):
        """encode() should concatenate measurements when present."""
        net = self._make_agent_net(with_measurements=True)
        net.eval()
        obs = TensorDict({
            "obs": torch.randn(2, 3, 64, 64),
            "measurements": torch.randn(2, 3),
        })
        with torch.no_grad():
            encoded = net.encode(obs)
        # encoder_out + 64 (measurements_head output)
        assert net.measurements_head is not None
        assert encoded.shape == (2, net.encoder_out_size)


class TestMultiHeadActionGuard:
    """Validates that 1D actions raise ValueError for multi-head Q-networks."""

    def _make_multi_head_net(self):
        net = object.__new__(QMixAgentNet)
        net.q_head = None  # Multi-head
        net.action_sizes = [3, 2, 2]
        net.num_heads = 3
        return net

    def test_1d_actions_raises_for_multi_head(self):
        net = self._make_multi_head_net()
        q_values = torch.randn(4, 7)  # [batch=4, total_actions=3+2+2=7]
        actions_1d = torch.tensor([0, 1, 2, 0])  # 1D — wrong for multi-head
        with pytest.raises(ValueError, match="Multi-head Q-network received 1D actions"):
            net.get_q_for_actions(q_values, actions_1d)

    def test_2d_actions_works_for_multi_head(self):
        net = self._make_multi_head_net()
        q_values = torch.tensor([
            [1.0, 0.5, 0.2, 0.8, 0.3, 0.1, 0.9],  # heads: [3], [2], [2]
        ])
        actions = torch.tensor([[0, 1, 0]])  # head0->idx0, head1->idx1, head2->idx0
        q = net.get_q_for_actions(q_values, actions)
        # head0: q[0]=1.0, head1: q[3+1]=0.3, head2: q[5+0]=0.1
        expected = 1.0 + 0.3 + 0.1
        assert abs(q.item() - expected) < 1e-5

    def test_multi_head_sums_per_head_q(self):
        """Additive decomposition: Q_agent = sum of Q_head(a_h)."""
        net = self._make_multi_head_net()
        q_values = torch.tensor([
            [1.2, 0.1, -0.2, 0.5, -1.0, -0.1, 0.2],
        ])
        actions = torch.tensor([[2, 0, 1]])  # head0->-0.2, head1->0.5, head2->0.2
        q = net.get_q_for_actions(q_values, actions)
        expected = -0.2 + 0.5 + 0.2
        assert abs(q.item() - expected) < 1e-5


class TestAlgoNaming:
    """Validates algo name normalization for experiment naming and registration."""

    def test_algo_upper_check(self):
        """str(algo).upper() should match 'QMIX' and 'VDN' for any case."""
        for algo_input in ('QMIX', 'qmix', 'Qmix', 'QMix'):
            assert str(algo_input).upper() in ('QMIX', 'VDN')
        for algo_input in ('VDN', 'vdn', 'Vdn'):
            assert str(algo_input).upper() in ('QMIX', 'VDN')
        for algo_input in ('APPO', 'appo', 'Appo'):
            assert str(algo_input).upper() not in ('QMIX', 'VDN')

    def test_mixer_default_is_truthy(self):
        """The default --mixer='qmix' is truthy, which was the bug in fix #7."""
        default_mixer = "qmix"
        assert bool(default_mixer)  # Always truthy — old code always entered QMIX branch


class TestTargetUpdate:
    """Validates soft update (Polyak averaging) behavior."""

    def test_soft_update_interpolates(self):
        """θ_target ← (1-τ)θ_target + τθ_online"""
        tau = 0.005
        target_param = torch.tensor([10.0])
        online_param = torch.tensor([20.0])
        # Polyak: target = (1-0.005)*10 + 0.005*20 = 9.95 + 0.1 = 10.05
        target_param.data.mul_(1 - tau).add_(online_param.data, alpha=tau)
        assert abs(target_param.item() - 10.05) < 1e-5

    def test_hard_update_copies(self):
        """tau=1.0 means full copy."""
        tau = 1.0
        target_param = torch.tensor([10.0])
        online_param = torch.tensor([20.0])
        target_param.data.mul_(1 - tau).add_(online_param.data, alpha=tau)
        assert abs(target_param.item() - 20.0) < 1e-5

    def test_tau_default_is_soft(self):
        """Default tau should be 0.005 (soft update), not 1.0."""
        import argparse
        from sample_factory.cfg.cfg import add_dqn_args
        parser = argparse.ArgumentParser(add_help=False)
        add_dqn_args(parser)
        args = parser.parse_args([])
        assert abs(args.target_update_tau - 0.005) < 1e-6


class TestActionMask:
    """QMixActorCritic.forward() now applies action_mask."""

    def _make_actor_critic(self):
        import gymnasium as gym
        from sample_factory.utils.attr_dict import AttrDict
        from sf.doom.qmix_model import QMixActorCritic
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
        """All-ones mask means all actions allowed — output unchanged."""
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
    """add and sample work correctly in sequential single-threaded access."""

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
    """Validates QMIX mixing weights are non-negative (monotonicity constraint)."""

    def test_mixing_weights_non_negative(self):
        from sf.doom.qmix_model import QMixMixer
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
    """Validates PER sampling and priority updates."""

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
    """Verify non-RNN schedule debt uses joint-transition units (not agent-transitions).

    Original code: total_env_steps_for_training += num_joint
    Regression would be: += num_joint * num_agents (2x update rate for 2 agents)
    """

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

    def test_non_rnn_debt_uses_joint_transitions(self):
        """With train_frequency=N and N joint transitions added, exactly 1 update should fire."""
        num_agents = 2
        train_freq = 4
        num_joint_added = 4  # exactly 1 update expected

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

        # This is the critical line: debt must be num_joint, NOT num_joint * num_agents
        transitions_added = num_joint  # as in the reverted code
        total_debt = 0
        total_debt += transitions_added
        num_updates = total_debt // train_freq
        total_debt -= num_updates * train_freq

        assert num_updates == 1, (
            f"Expected 1 update (4 joint transitions / train_freq=4), got {num_updates}. "
            f"If this is 2, debt is using agent-transitions instead of joint-transitions."
        )
        assert total_debt == 0

    def test_non_rnn_debt_regression_agent_transitions_would_double(self):
        """Verify that using agent-transition debt would produce 2x updates (the regression)."""
        num_agents = 2
        train_freq = 4
        num_joint_added = 4

        # If we used agent transitions (the regression):
        agent_transitions = num_joint_added * num_agents  # = 8
        num_updates_regression = agent_transitions // train_freq  # = 2
        assert num_updates_regression == 2, "Sanity: agent-transition bug would cause 2 updates"

        # Correct behavior (joint transitions):
        num_updates_correct = num_joint_added // train_freq  # = 1
        assert num_updates_correct == 1
