"""Test that BatchedVectorEnvRunner._process_env_step extracts true_objective from non-vectorized infos"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner


def _make_stub_runner(num_agents: int = 3):
    runner = object.__new__(BatchedVectorEnvRunner)
    runner.cfg = SimpleNamespace(
        summaries_use_frameskip=False,
        reward_scale=1.0,
        reward_clip=float('inf'),
    )
    runner.env_info = SimpleNamespace(frameskip=1)
    runner.vec_env = SimpleNamespace(num_agents=num_agents)
    runner.policy_id = 0
    runner.curr_episode_reward = torch.zeros(num_agents)
    runner.curr_episode_len = torch.zeros(num_agents, dtype=torch.int64)
    runner.min_raw_rewards = torch.empty(num_agents).fill_(float('inf'))
    runner.max_raw_rewards = torch.empty(num_agents).fill_(float('-inf'))
    return runner


class TestTrueObjectiveExtraction:
    def test_true_objective_extracted_from_nonvectorized_infos(self):
        runner = _make_stub_runner(num_agents=2)

        runner.curr_episode_reward = torch.tensor([10.0, 20.0])
        runner.curr_episode_len = torch.tensor([100, 200])

        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([True, True])

        infos = [
            {"true_objective": 5.0, "episode_extra_stats": {"kills": 3}},
            {"true_objective": 8.0},
        ]

        reports = runner._process_env_step(rewards, dones, infos)
        assert len(reports) == 1
        stats = reports[0]["episodic"]

        assert "true_objective" in stats, "true_objective should be present in stats"
        np.testing.assert_array_almost_equal(stats["true_objective"], [5.0, 8.0])

    def test_true_objective_missing_when_infos_lacks_it(self):
        """No agent info contains true_objective"""
        runner = _make_stub_runner(num_agents=2)
        runner.curr_episode_reward = torch.tensor([10.0, 20.0])
        runner.curr_episode_len = torch.tensor([100, 200])

        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([True, True])
        infos = [{"episode_extra_stats": {"kills": 1}}, {}]

        reports = runner._process_env_step(rewards, dones, infos)
        stats = reports[0]["episodic"]
        assert "true_objective" not in stats

    def test_partial_true_objective(self):
        """Some finished agents have true_objective"""
        runner = _make_stub_runner(num_agents=3)
        runner.curr_episode_reward = torch.tensor([10.0, 20.0, 30.0])
        runner.curr_episode_len = torch.tensor([100, 200, 300])

        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([True, False, True]) # agents 0 and 2 done

        infos = [
            {"true_objective": 5.0},
            {}, # agent 1 not done, not checked
            {"true_objective": 15.0},
        ]

        reports = runner._process_env_step(rewards, dones, infos)
        stats = reports[0]["episodic"]
        assert "true_objective" in stats
        np.testing.assert_array_almost_equal(stats["true_objective"], [5.0, 15.0])
