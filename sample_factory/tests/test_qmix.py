from __future__ import annotations
import pytest
import torch
from sample_factory.algo.learning.learner_qmix import QMixLearner
from sample_factory.algo.utils.joint_replay_buffer import JointReplayBuffer
from sample_factory.algo.utils.tensor_dict import TensorDict
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
    # After masking dones with (1 - time_outs), agent 0 remains done at first joint step
    assert j["dones"][0, 0].item() == 1.0


def test_prepare_joint_transitions_done_masking_agent1(joint):
    j, *_ = joint
    # Agent 1 was timed out, so effective done should be 0
    assert j["dones"][0, 1].item() == 0.0


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
