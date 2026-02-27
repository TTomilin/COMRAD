import numpy as np
import pytest
import torch

from sample_factory.algo.learning.learner_happo import HAPPOLearner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from sf.doom.happo_model import make_happo_actor_critic

from .conftest import _make_happo_cfg, _make_obs_space, _make_action_space, _make_obs_batch


class TestCriticRNNTraining:

    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def _make_critic_rnn_stub(self, num_agents=2, rnn_type="gru"):
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, rnn_type=rnn_type,
                              rnn_size=32, happo_critic_rnn=True)
        cfg.recurrence = 4
        cfg.rollout = 4
        cfg.ppo_clip_value = 10.0
        cfg.value_loss_coeff = 0.5
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.train()

        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = num_agents
        stub.actor_critic = model
        stub.use_critic_rnn = getattr(cfg, 'happo_critic_rnn', False) and cfg.use_rnn
        return stub, model, cfg

    def test_critic_loss_with_rnn_runs(self):
        stub, model, cfg = self._make_critic_rnn_stub()
        recurrence = cfg.recurrence
        num_agents = cfg.num_agents
        batch_size = recurrence * num_agents

        obs = _make_obs_batch(batch_size, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch_size) // num_agents

        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
        })

        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)

        assert value_loss.requires_grad
        assert not torch.isnan(value_loss)
        assert not torch.isinf(value_loss)

    def test_gradients_flow_through_critic_cores(self):
        stub, model, cfg = self._make_critic_rnn_stub()
        recurrence = cfg.recurrence
        num_agents = cfg.num_agents
        batch_size = recurrence * num_agents

        obs = _make_obs_batch(batch_size, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch_size) // num_agents

        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
        })

        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)
        value_loss.backward()

        # critic_cores should have gradients
        for i, core in enumerate(model.critic_cores):
            has_grad = any(p.grad is not None and (p.grad != 0).any() for p in core.parameters())
            assert has_grad, f"Critic core {i} has no gradients after backward"

    def test_critic_rnn_gradients_isolated_from_actor(self):
        stub, model, cfg = self._make_critic_rnn_stub()
        recurrence = cfg.recurrence
        num_agents = cfg.num_agents
        batch_size = recurrence * num_agents

        obs = _make_obs_batch(batch_size, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch_size) // num_agents

        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
        })

        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)
        value_loss.backward()

        # Actor encoder/core/decoder should have zero or no gradients
        for name, param in model.agent_encoders[0].named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"Actor encoder param {name} has grad from critic RNN loss"

    def test_critic_projection_receives_gradients(self):
        # Use rnn_size != encoder output to force projection creation
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_type="gru",
                              rnn_size=32, happo_critic_rnn=True)
        cfg.recurrence = 4
        cfg.rollout = 4
        cfg.ppo_clip_value = 10.0
        cfg.value_loss_coeff = 0.5
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.train()

        # Projection should exist (enc_out=512 != rnn_size=32)
        assert model.critic_projection is not None, "Expected critic_projection to exist"

        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = 2
        stub.actor_critic = model

        batch_size = cfg.recurrence * 2
        obs = _make_obs_batch(batch_size, 2)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch_size) // 2
        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
        })

        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)
        value_loss.backward()

        has_grad = any(
            p.grad is not None and (p.grad != 0).any()
            for p in model.critic_projection.parameters()
        )
        assert has_grad, "critic_projection should receive gradients during critic training"

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_critic_rnn_both_types(self, rnn_type):
        stub, model, cfg = self._make_critic_rnn_stub(rnn_type=rnn_type)
        recurrence = cfg.recurrence
        num_agents = cfg.num_agents
        batch_size = recurrence * num_agents

        obs = _make_obs_batch(batch_size, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch_size) // num_agents

        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
        })

        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)
        assert not torch.isnan(value_loss)
        value_loss.backward()


class TestTransitionRNNMinibatches:

    @pytest.fixture
    def learner_stub(self):
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        stub.cfg = AttrDict({"recurrence": 4, "rollout": 4})
        return stub

    def test_minibatch_temporal_alignment(self, learner_stub):
        n_agents = 2
        n_envs = 2
        rollout = 4
        experience_size = n_agents * n_envs * rollout

        # Build data layout: agent0_env0(4 steps), agent1_env0(4 steps), agent0_env1(4 steps), agent1_env1(4 steps)
        env_idx = torch.tensor(
            [0] * rollout + [0] * rollout + [1] * rollout + [1] * rollout, dtype=torch.long
        )
        agent_idx = torch.tensor(
            [0] * rollout + [1] * rollout + [0] * rollout + [1] * rollout, dtype=torch.long
        )
        env_group_idx = learner_stub._compute_env_group_idx(agent_idx, env_idx, experience_size)
        n_transitions = env_group_idx.max().item() + 1

        minibatches = learner_stub._get_transition_rnn_minibatches(
            batch_size=experience_size, experience_size=experience_size,
            env_group_idx=env_group_idx, n_transitions=n_transitions,
            env_idx=env_idx, agent_idx=agent_idx,
        )

        # All samples should be covered
        all_indices = np.concatenate(minibatches)
        assert len(all_indices) == experience_size
        assert len(set(all_indices)) == experience_size

    def test_each_minibatch_has_all_agents(self, learner_stub):
        n_agents = 2
        rollout = 4
        experience_size = n_agents * rollout

        env_idx = torch.tensor([0] * rollout + [0] * rollout, dtype=torch.long)
        agent_idx = torch.tensor([0] * rollout + [1] * rollout, dtype=torch.long)
        env_group_idx = learner_stub._compute_env_group_idx(agent_idx, env_idx, experience_size)
        n_transitions = env_group_idx.max().item() + 1

        minibatches = learner_stub._get_transition_rnn_minibatches(
            batch_size=experience_size, experience_size=experience_size,
            env_group_idx=env_group_idx, n_transitions=n_transitions,
            env_idx=env_idx, agent_idx=agent_idx,
        )

        for mb in minibatches:
            mb_groups = env_group_idx[mb]
            for g in mb_groups.unique():
                assert (mb_groups == g).sum() == n_agents


class TestCriticRNNStoredStatesTraining:

    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def _make_stub(self, num_agents=2, rnn_type="gru"):
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, rnn_type=rnn_type,
                              rnn_size=32, happo_critic_rnn=True)
        cfg.recurrence = 4
        cfg.rollout = 4
        cfg.ppo_clip_value = 10.0
        cfg.ppo_clip_ratio = 0.2
        cfg.value_loss_coeff = 0.5
        cfg.exploration_loss_coeff = 0.0
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.train()

        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = num_agents
        stub.actor_critic = model
        stub.exploration_loss_func = lambda action_distr, valids, num_invalids: 0.0
        return stub, model, cfg

    def test_stored_states_affect_critic_output(self):
        stub, model, cfg = self._make_stub()
        num_agents = cfg.num_agents
        batch_size = cfg.recurrence * num_agents
        rnn_size = get_rnn_size(cfg)
        R = model.critic_rnn_state_size

        obs = _make_obs_batch(batch_size, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch_size) // num_agents
        base_returns = torch.randn(batch_size)

        # Run with zero critic states
        mb_zero = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": base_returns,
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": torch.zeros(batch_size, rnn_size),
        })
        loss_zero = stub._calculate_critic_loss(mb_zero, agent_idx, env_group_idx, num_invalids=0)

        # Run with non-zero critic states  (actor half = zeros, critic half = randn)
        rnn_with_critic = torch.zeros(batch_size, rnn_size)
        rnn_with_critic[:, R:] = torch.randn(batch_size, R) * 0.5
        mb_nonzero = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": base_returns,
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
            "rnn_states": rnn_with_critic,
        })
        loss_nonzero = stub._calculate_critic_loss(mb_nonzero, agent_idx, env_group_idx, num_invalids=0)

        # The losses should differ because the critic RNN states affect the critic output
        assert not torch.allclose(loss_zero.detach(), loss_nonzero.detach()), \
            "Critic loss should differ when using non-zero stored states vs zero states"

    def test_actor_policy_loss_slices_actor_half(self):
        stub, model, cfg = self._make_stub()
        num_agents = cfg.num_agents
        recurrence = cfg.recurrence
        # Single agent batch
        batch_size = recurrence
        rnn_size = get_rnn_size(cfg)

        obs = torch.randn(batch_size, 3, 64, 64)
        agent_id = torch.zeros(batch_size, num_agents)
        agent_id[:, 0] = 1.0  # All agent 0
        obs_dict = {"obs": obs, "agent_id": agent_id}

        model.eval()
        with torch.no_grad():
            # Get some action logits for the log_prob_actions
            head = model.forward_head(obs_dict, agent_idx=agent_id.argmax(dim=-1))
            core, _ = model.agent_cores[0](head, torch.zeros(batch_size, model.critic_rnn_state_size))
            dec = model.agent_decoders[0](core)
            logits, _ = model.agent_action_params[0](dec, None)
            dist = get_action_distribution(model.action_space, logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        model.train()
        mb = AttrDict({
            "normalized_obs": obs_dict,
            "actions": actions,
            "log_prob_actions": log_probs,
            "valids": torch.ones(batch_size, dtype=torch.bool),
            "dones_cpu": torch.zeros(batch_size, dtype=torch.bool),
            "rnn_states": torch.randn(batch_size, rnn_size),  # Full 2R size
        })
        M_full = torch.ones(batch_size)
        mb_indices = torch.arange(batch_size)

        # Should NOT crash — should correctly slice actor half
        policy_loss, exploration_loss = stub._calculate_agent_policy_loss(
            0, mb, M_full, mb_indices, num_invalids=0
        )
        assert not torch.isnan(policy_loss)
        assert policy_loss.requires_grad

    def test_evaluate_agent_log_probs_slices_actor_half(self):
        stub, model, cfg = self._make_stub()
        num_agents = cfg.num_agents
        recurrence = cfg.recurrence
        # Build a buffer-like structure for agent 0
        batch_size = recurrence * num_agents  # Need both agents for full buffer
        rnn_size = get_rnn_size(cfg)

        obs = torch.randn(batch_size, 3, 64, 64)
        agent_id = torch.zeros(batch_size, num_agents)
        for i in range(batch_size):
            agent_id[i, i % num_agents] = 1.0
        obs_dict = {"obs": obs, "agent_id": agent_id}

        # Generate actions
        model.eval()
        with torch.no_grad():
            result = model(obs_dict, torch.zeros(batch_size, rnn_size))
        actions = result["actions"]
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Build gpu_buffer-like dict
        gpu_buffer = {
            "normalized_obs": obs_dict,
            "actions": actions,
            "rnn_states": torch.randn(batch_size, rnn_size),  # Full 2R size
            "dones_cpu": torch.zeros(batch_size, dtype=torch.bool),
            "valids": torch.ones(batch_size, dtype=torch.bool),
        }

        # Agent 0 mask
        agent_mask = agent_id[:, 0].bool()

        # Should NOT crash — should correctly slice actor half
        log_probs = stub._evaluate_agent_log_probs(0, gpu_buffer, agent_mask)
        assert log_probs.shape == (agent_mask.sum(),)
        assert not torch.isnan(log_probs).any()

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_multi_layer_rnn_forward(self, rnn_type):
        num_agents = 2
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, rnn_type=rnn_type,
                              rnn_size=32, rnn_num_layers=2, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()

        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_size = get_rnn_size(cfg)
        rnn_states = torch.zeros(batch, rnn_size)

        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result['new_rnn_states'].shape == (batch, rnn_size)
        assert not torch.isnan(result['values']).any()
        # Verify both halves updated
        R = model.critic_rnn_state_size
        assert not torch.allclose(result['new_rnn_states'][:, :R], torch.zeros(batch, R))
        assert not torch.allclose(result['new_rnn_states'][:, R:], torch.zeros(batch, R))
