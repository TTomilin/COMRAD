import gymnasium as gym
import numpy as np
import pytest
import torch

from sf.doom.happo_model import _group_by_env, remove_agentid
from sf.doom.wrappers.agent_id_wrapper import AgentIDWrapper

from .conftest import _make_obs_space


class TestRemoveAgentIdFromObsSpace:
    def test_removes_agent_id_key(self):
        obs_space = _make_obs_space(num_agents=2)
        result = remove_agentid(obs_space)
        assert "agent_id" not in result.spaces
        assert "obs" in result.spaces

    def test_preserves_other_keys(self):
        obs_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(0, 1, shape=(4,)),
                "agent_id": gym.spaces.Box(0, 1, shape=(2,)),
                "extra": gym.spaces.Box(0, 1, shape=(3,)),
            }
        )
        result = remove_agentid(obs_space)
        assert "obs" in result.spaces
        assert "extra" in result.spaces
        assert "agent_id" not in result.spaces


class TestAgentIDWrapper:
    def test_dict_obs_space_augmented(self):
        base_env = gym.make("CartPole-v1")
        env = AgentIDWrapper(base_env, agent_index=0, num_agents=2)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "agent_id" in env.observation_space.spaces
        assert env.observation_space["agent_id"].shape == (2,)
        base_env.close()

    def test_one_hot_correctness(self):
        base_env = gym.make("CartPole-v1")
        env = AgentIDWrapper(base_env, agent_index=1, num_agents=3)
        obs, _ = env.reset()
        assert "agent_id" in obs
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(obs["agent_id"], expected)
        base_env.close()

    def test_different_agent_indices(self):
        for agent_idx in range(3):
            base_env = gym.make("CartPole-v1")
            env = AgentIDWrapper(base_env, agent_index=agent_idx, num_agents=3)
            obs, _ = env.reset()
            expected = np.zeros(3, dtype=np.float32)
            expected[agent_idx] = 1.0
            np.testing.assert_array_equal(obs["agent_id"], expected)
            base_env.close()


class TestGroupByEnv:
    def test_basic_grouping(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        agent_idx = torch.tensor([0, 1, 0, 1])
        env_group_idx = torch.tensor([0, 0, 1, 1])
        n_agents = 2

        grouped = _group_by_env(features, agent_idx, env_group_idx, n_agents)

        assert grouped.shape == (2, 2, 2)
        torch.testing.assert_close(grouped[0, 0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(grouped[0, 1], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(grouped[1, 0], torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(grouped[1, 1], torch.tensor([7.0, 8.0]))

    def test_non_contiguous_agents(self):
        features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        agent_idx = torch.tensor([0, 0, 1, 1])
        env_group_idx = torch.tensor([0, 1, 0, 1])
        n_agents = 2

        grouped = _group_by_env(features, agent_idx, env_group_idx, n_agents)

        assert grouped.shape == (2, 2, 1)
        assert grouped[0, 0].item() == 1.0  # env0, agent0
        assert grouped[0, 1].item() == 3.0  # env0, agent1
        assert grouped[1, 0].item() == 2.0  # env1, agent0
        assert grouped[1, 1].item() == 4.0  # env1, agent1

    def test_three_agents(self):
        B = 6
        F = 4
        features = torch.randn(B, F)
        agent_idx = torch.tensor([0, 1, 2, 0, 1, 2])
        env_group_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        grouped = _group_by_env(features, agent_idx, env_group_idx, 3)

        assert grouped.shape == (2, 3, F)
        for i in range(B):
            torch.testing.assert_close(grouped[env_group_idx[i], agent_idx[i]], features[i])

    def test_duplicate_pair_raises(self):
        features = torch.tensor([[1.0], [2.0]])
        agent_idx = torch.tensor([0, 0])
        env_group_idx = torch.tensor([0, 0])

        with pytest.raises(AssertionError, match="Duplicate"):
            _group_by_env(features, agent_idx, env_group_idx, 2)
