import numpy as np
import pytest
import torch

from sample_factory.algo.learning.learner_happo import HAPPOLearner
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from sf.doom.happo_model import _group_by_env

from .conftest import _make_happo_model, _make_obs_batch


class TestComputeEnvGroupIdx:

    @pytest.fixture
    def learner_stub(self):
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        stub.cfg = AttrDict({"rollout": 4})
        return stub

    def test_basic_grouping(self, learner_stub):
        n_agents = 2
        n_envs = 2
        rollout = 4
        dataset_size = n_agents * n_envs * rollout  # 16

        # env_idx follows batched_sampling layout:
        # Agent 0 of env 0: rollout steps, Agent 1 of env 0: rollout steps, etc.
        env_idx = torch.tensor(
            [0] * rollout + [0] * rollout + [1] * rollout + [1] * rollout,
            dtype=torch.long,
        )
        agent_idx = torch.tensor(
            [0] * rollout + [1] * rollout + [0] * rollout + [1] * rollout,
            dtype=torch.long,
        )

        group_idx = learner_stub._compute_env_group_idx(agent_idx, env_idx, dataset_size)

        assert group_idx.shape == (dataset_size,)
        n_transitions = group_idx.max().item() + 1
        assert n_transitions == n_envs * rollout  # 8 unique (env, timestep) pairs

        counts = torch.bincount(group_idx)
        assert (counts == n_agents).all()

    def test_multi_window_same_env(self, learner_stub):
        n_agents = 2
        rollout = 4
        # 2 windows from env 0: 4 trajectories total (2 agents × 2 windows)
        n_traj = 4
        dataset_size = n_traj * rollout  # 16

        # Layout: [agent0_env0_w1, agent1_env0_w1, agent0_env0_w2, agent1_env0_w2]
        env_idx = torch.tensor(
            [0] * rollout + [0] * rollout + [0] * rollout + [0] * rollout,
            dtype=torch.long,
        )
        agent_idx = torch.tensor(
            [0] * rollout + [1] * rollout + [0] * rollout + [1] * rollout,
            dtype=torch.long,
        )

        group_idx = learner_stub._compute_env_group_idx(agent_idx, env_idx, dataset_size)

        assert group_idx.shape == (dataset_size,)
        # 2 windows × 4 timesteps = 8 unique transition groups
        n_transitions = group_idx.max().item() + 1
        assert n_transitions == 2 * rollout

        counts = torch.bincount(group_idx, minlength=n_transitions)
        assert (counts == n_agents).all()

        window1_groups = set(group_idx[:n_agents * rollout].tolist())
        window2_groups = set(group_idx[n_agents * rollout:].tolist())
        assert window1_groups.isdisjoint(window2_groups), (
            "Groups from different rollout windows should not overlap"
        )

    def test_wrong_agent_count_raises(self, learner_stub):
        # 3 samples with env_idx 0 at timestep 0, but n_agents=2
        dataset_size = 3
        env_idx = torch.tensor([0, 0, 0], dtype=torch.long)
        agent_idx = torch.tensor([0, 1, 0], dtype=torch.long)

        learner_stub.cfg.rollout = 1
        with pytest.raises(RuntimeError, match="wrong agent count"):
            learner_stub._compute_env_group_idx(agent_idx, env_idx, dataset_size)

    def test_mismatched_env_idx_in_group_raises(self, learner_stub):
        rollout = 4
        n_agents = 2
        # 2 trajectories but with different env_idx (agent0 from env 0, agent1 from env 1)
        dataset_size = n_agents * rollout  # 8

        env_idx = torch.tensor(
            [0] * rollout + [1] * rollout,  # mismatched!
            dtype=torch.long,
        )
        agent_idx = torch.tensor(
            [0] * rollout + [1] * rollout,
            dtype=torch.long,
        )

        with pytest.raises(RuntimeError, match="mismatched env_idx"):
            learner_stub._compute_env_group_idx(agent_idx, env_idx, dataset_size)


class TestGetAgentMinibatches:

    @pytest.fixture
    def learner_stub(self):
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        stub.cfg = AttrDict({"recurrence": 1})
        return stub

    def test_single_agent_indices(self, learner_stub):
        experience_size = 100
        agent_mask = torch.zeros(experience_size, dtype=torch.bool)
        # Agent 0 gets even indices, Agent 1 gets odd
        agent_mask[0::2] = True  # Agent 0

        minibatches = learner_stub._get_agent_minibatches(
            batch_size=experience_size, agent_mask=agent_mask, experience_size=experience_size
        )

        all_indices = np.concatenate(minibatches)
        assert len(all_indices) == 50  # Half the buffer
        assert np.all(all_indices % 2 == 0)  # All even indices

    def test_minibatch_sizes(self, learner_stub):
        experience_size = 40
        agent_mask = torch.zeros(experience_size, dtype=torch.bool)
        agent_mask[:20] = True  # 20 samples for this agent

        minibatches = learner_stub._get_agent_minibatches(
            batch_size=20, agent_mask=agent_mask, experience_size=experience_size
        )

        for mb in minibatches:
            assert len(mb) <= 20 // 2  # batch_size / n_agents = 10

    def test_rnn_recurrence_alignment(self, learner_stub):
        learner_stub.cfg.recurrence = 4
        learner_stub.cfg.rollout = 4  # rollout must be divisible by recurrence
        experience_size = 40
        agent_mask = torch.zeros(experience_size, dtype=torch.bool)
        agent_mask[:20] = True  # 20 samples (divisible by recurrence=4)

        minibatches = learner_stub._get_agent_minibatches(
            batch_size=40, agent_mask=agent_mask, experience_size=experience_size
        )

        for mb in minibatches:
            assert len(mb) % 4 == 0, f"Minibatch size {len(mb)} not aligned to recurrence=4"


class TestGetTransitionMinibatches:

    @pytest.fixture
    def learner_stub(self):
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        return stub

    def test_complete_transitions(self, learner_stub):
        n_agents = 2
        n_transitions = 4
        experience_size = n_agents * n_transitions

        # Interleaved layout: [agent0_t0, agent1_t0, agent0_t1, agent1_t1, ...]
        env_group_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        batch_size = n_agents * 2  # 2 transitions per batch

        minibatches = learner_stub._get_transition_minibatches(
            batch_size, experience_size, env_group_idx, n_transitions
        )

        for mb in minibatches:
            mb_groups = env_group_idx[mb]
            for g in mb_groups.unique():
                assert (mb_groups == g).sum() == n_agents

    def test_wrong_agent_count_raises(self, learner_stub):
        env_group_idx = torch.tensor([0, 0, 0, 1, 1])  # 3 agents in transition 0
        with pytest.raises(RuntimeError, match="has .* samples"):
            learner_stub._get_transition_minibatches(4, 5, env_group_idx, 2)


class TestGradientIsolation:

    def test_gradient_isolation_between_agents(self):
        num_agents = 2
        model = _make_happo_model(num_agents=num_agents)
        model.train()

        batch_size = 4
        obs = _make_obs_batch(batch_size, num_agents)
        # All samples are agent 0
        obs["agent_id"] = torch.zeros(batch_size, num_agents)
        obs["agent_id"][:, 0] = 1.0
        agent_idx = torch.zeros(batch_size, dtype=torch.long)

        head_out = model.forward_head(obs, agent_idx=agent_idx)
        core_out, _ = model.forward_core(
            head_out, torch.zeros(batch_size, get_rnn_size(model.cfg)), agent_idx=agent_idx
        )
        decoder_out = model.agent_decoders[0](core_out)
        logits, _ = model.agent_action_params[0](decoder_out, None)

        loss = logits.sum()
        loss.backward()

        for name, param in model.agent_encoders[1].named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"Agent 1 encoder param {name} has non-zero grad!"

        has_nonzero = False
        for param in model.agent_encoders[0].parameters():
            if param.grad is not None and (param.grad != 0).any():
                has_nonzero = True
                break
        assert has_nonzero, "Agent 0 encoder has all-zero gradients!"

    def test_critic_gradient_isolation_from_actor(self):
        num_agents = 2
        batch = num_agents * 2  # 2 transitions
        model = _make_happo_model(num_agents=num_agents)
        model.train()

        obs = _make_obs_batch(batch, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch) // num_agents

        obs_no_id = {k: v for k, v in obs.items() if k != "agent_id"}
        critic_enc_out = model.critic_encoders[0].get_out_size()
        critic_features = torch.zeros(batch, critic_enc_out)
        for i in range(num_agents):
            mask = agent_idx == i
            if mask.any():
                agent_obs = {k: v[mask] for k, v in obs_no_id.items()}
                critic_features[mask] = model.critic_encoders[i](agent_obs)

        grouped = _group_by_env(critic_features, agent_idx, env_group_idx, num_agents)
        critic_input = grouped.view(2, -1)
        joint_value = model.centralized_critic(critic_input)

        loss = joint_value.sum()
        loss.backward()

        for name, param in model.agent_encoders[0].named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"Actor param {name} has grad from critic loss!"
