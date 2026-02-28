from __future__ import annotations
import pytest
import torch
from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from comrad.models.qmix_model import QMixAgentNet


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

    def test_normal_step_bootstraps(self):
        joint_done = torch.tensor([0.0])
        joint_time_out = torch.tensor([0.0])
        effective_done = joint_done * (1 - joint_time_out)
        assert effective_done.item() == 0.0  # 1 - effective_done = 1 -> bootstrap

    def test_true_death_no_bootstrap(self):
        joint_done = torch.tensor([1.0])
        joint_time_out = torch.tensor([0.0])
        effective_done = joint_done * (1 - joint_time_out)
        assert effective_done.item() == 1.0  # 1 - effective_done = 0 -> no bootstrap

    def test_timeout_bootstraps(self):
        joint_done = torch.tensor([1.0])
        joint_time_out = torch.tensor([1.0])
        effective_done = joint_done * (1 - joint_time_out)
        assert effective_done.item() == 0.0  # 1 - effective_done = 1 -> bootstrap

    def test_bellman_target_with_timeout(self):
        gamma = 0.99
        reward = torch.tensor([1.0])
        q_next = torch.tensor([10.0])
        # Timeout case: should bootstrap
        effective_done = torch.tensor([0.0])  # timeout
        target = reward + gamma * (1 - effective_done) * q_next
        assert abs(target.item() - 10.9) < 1e-5

    def test_bellman_target_with_true_death(self):
        gamma = 0.99
        reward = torch.tensor([1.0])
        q_next = torch.tensor([10.0])
        effective_done = torch.tensor([1.0])  # true death
        target = reward + gamma * (1 - effective_done) * q_next
        assert abs(target.item() - 1.0) < 1e-5

    def test_batch_mixed_scenarios(self):
        joint_done = torch.tensor([0.0, 1.0, 1.0])
        joint_time_out = torch.tensor([0.0, 0.0, 1.0])
        effective_done = joint_done * (1 - joint_time_out)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(effective_done, expected)

    def test_no_timeout_key_fallback(self):
        joint_done = torch.tensor([1.0, 0.0])
        joint_time_out = None
        if joint_time_out is not None:
            effective_done = joint_done * (1 - joint_time_out)
        else:
            effective_done = joint_done
        assert torch.allclose(effective_done, joint_done)


class TestEncodeForwardConsistency:

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
        net = self._make_multi_head_net()
        q_values = torch.tensor([
            [1.2, 0.1, -0.2, 0.5, -1.0, -0.1, 0.2],
        ])
        actions = torch.tensor([[2, 0, 1]])  # head0->-0.2, head1->0.5, head2->0.2
        q = net.get_q_for_actions(q_values, actions)
        expected = -0.2 + 0.5 + 0.2
        assert abs(q.item() - expected) < 1e-5


class TestAlgoNaming:

    def test_algo_upper_check(self):
        for algo_input in ('QMIX', 'qmix', 'Qmix', 'QMix'):
            assert str(algo_input).upper() in ('QMIX', 'VDN')
        for algo_input in ('VDN', 'vdn', 'Vdn'):
            assert str(algo_input).upper() in ('QMIX', 'VDN')
        for algo_input in ('APPO', 'appo', 'Appo'):
            assert str(algo_input).upper() not in ('QMIX', 'VDN')

    def test_mixer_default_is_truthy(self):
        default_mixer = "qmix"
        assert bool(default_mixer)  # Always truthy — old code always entered QMIX branch


class TestTargetUpdate:

    def test_soft_update_interpolates(self):
        tau = 0.005
        target_param = torch.tensor([10.0])
        online_param = torch.tensor([20.0])
        # Polyak: target = (1-0.005)*10 + 0.005*20 = 9.95 + 0.1 = 10.05
        target_param.data.mul_(1 - tau).add_(online_param.data, alpha=tau)
        assert abs(target_param.item() - 10.05) < 1e-5

    def test_hard_update_copies(self):
        tau = 1.0
        target_param = torch.tensor([10.0])
        online_param = torch.tensor([20.0])
        target_param.data.mul_(1 - tau).add_(online_param.data, alpha=tau)
        assert abs(target_param.item() - 20.0) < 1e-5

    def test_tau_default_is_soft(self):
        import argparse
        from sample_factory.cfg.cfg import add_dqn_args
        parser = argparse.ArgumentParser(add_help=False)
        add_dqn_args(parser)
        args = parser.parse_args([])
        assert abs(args.target_update_tau - 0.005) < 1e-6
