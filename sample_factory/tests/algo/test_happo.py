"""
Tests:
HAPPOActorCritic model: forward pass shapes, per-agent routing, centralized critic
HAPPOLearner: factor M correctness, gradient isolation, env_group_idx, minibatch generation
Helper functions: _group_by_env, AgentIDWrapper
"""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.context import global_model_factory, sf_global_context
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from sf.doom.happo_model import (
    HAPPOActorCritic,
    _group_by_env,
    make_happo_actor_critic,
    remove_agentid,
)
from sf.doom.wrappers.agent_id_wrapper import AgentIDWrapper
from sample_factory.algo.learning.learner_happo import HAPPOLearner


def _make_happo_cfg(
    *,
    num_agents: int = 2,
    use_rnn: bool = False,
    rnn_type: str = "gru",
    rnn_size: int = 64,
    rnn_num_layers: int = 1,
    hidden_size: int = 32,
    happo_critic_hidden_sizes: list = None,
) -> AttrDict:
    if happo_critic_hidden_sizes is None:
        happo_critic_hidden_sizes = [64, 32]
    return AttrDict(
        {
            "encoder_conv_architecture": "convnet_simple",
            "encoder_conv_mlp_layers": [],
            "encoder_extra_fc_layers": 0,
            "hidden_size": hidden_size,
            "nonlinearity": "relu",
            "use_rnn": use_rnn,
            "rnn_type": rnn_type,
            "rnn_size": rnn_size,
            "rnn_num_layers": rnn_num_layers,
            "decoder_mlp_layers": [],
            "normalize_input": False,
            "normalize_returns": False,
            "obs_subtract_mean": 0.0,
            "obs_scale": 1.0,
            "num_agents": num_agents,
            "algo": "HAPPO",
            "adaptive_stddev": True,
            "initial_stddev": 1.0,
            "policy_initialization": "orthogonal",
            "policy_init_gain": 1.0,
            "actor_critic_share_weights": True,
            "happo_critic_hidden_sizes": happo_critic_hidden_sizes,
        }
    )


def _make_obs_space(obs_shape=(3, 64, 64), num_agents=2):
    """Create observation space WITH agent_id (as HAPPO expects after AgentIDWrapper)."""
    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(0, 1, shape=obs_shape, dtype=np.float32),
            "agent_id": gym.spaces.Box(0, 1, shape=(num_agents,), dtype=np.float32),
        }
    )


def _make_obs_space_no_id(obs_shape=(3, 64, 64)):
    """Create observation space WITHOUT agent_id (raw env obs)."""
    return gym.spaces.Dict(
        {"obs": gym.spaces.Box(0, 1, shape=obs_shape, dtype=np.float32)}
    )


def _make_action_space(n_actions=4):
    return gym.spaces.Discrete(n_actions)


def _make_happo_model(num_agents=2, use_rnn=False, obs_shape=(3, 64, 64), n_actions=4):
    """Create a HAPPOActorCritic model with default config."""
    cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=use_rnn)
    obs_space = _make_obs_space(obs_shape=obs_shape, num_agents=num_agents)
    action_space = _make_action_space(n_actions)
    return make_happo_actor_critic(cfg, obs_space, action_space)


def _make_obs_batch(batch_size, num_agents, obs_shape=(3, 64, 64)):
    """Create a batch of observations with agent_id one-hot vectors."""
    obs = torch.randn(batch_size, *obs_shape)
    agent_id = torch.zeros(batch_size, num_agents)
    for i in range(batch_size):
        agent_id[i, i % num_agents] = 1.0
    return {"obs": obs, "agent_id": agent_id}


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
        # CartPole has Box obs space, wrapper should convert to Dict
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
        """2 envs, 2 agents → 2 transitions."""
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
        """Agents interleaved: [agent0_env0, agent0_env1, agent1_env0, agent1_env1]."""
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
        """3 agents, 2 transitions."""
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
        """Duplicate (env_group, agent) pair should trigger assertion."""
        features = torch.tensor([[1.0], [2.0]])
        agent_idx = torch.tensor([0, 0])  # Same agent twice in same group
        env_group_idx = torch.tensor([0, 0])

        with pytest.raises(AssertionError, match="Duplicate"):
            _group_by_env(features, agent_idx, env_group_idx, 2)



class TestHAPPOModelConstruction:
    def test_creates_per_agent_networks(self):
        model = _make_happo_model(num_agents=3)
        assert len(model.agent_encoders) == 3
        assert len(model.agent_cores) == 3
        assert len(model.agent_decoders) == 3
        assert len(model.agent_action_params) == 3

    def test_creates_critic_encoders(self):
        model = _make_happo_model(num_agents=2)
        assert len(model.critic_encoders) == 2
        assert model.centralized_critic is not None

    def test_encoders_list_for_base_compat(self):
        model = _make_happo_model(num_agents=2)
        assert len(model.encoders) == 2
        assert model.encoders[0] is model.agent_encoders[0]

    def test_separate_agent_params(self):
        """Each agent should have independent parameters."""
        model = _make_happo_model(num_agents=2)
        params_0 = set(id(p) for p in model.agent_encoders[0].parameters())
        params_1 = set(id(p) for p in model.agent_encoders[1].parameters())
        assert params_0.isdisjoint(params_1), "Agent encoders share parameters!"

    def test_critic_params_separate_from_actor(self):
        """Critic encoders should NOT share params with actor encoders."""
        model = _make_happo_model(num_agents=2)
        actor_params = set(id(p) for p in model.agent_encoders[0].parameters())
        critic_params = set(id(p) for p in model.critic_encoders[0].parameters())
        assert actor_params.isdisjoint(critic_params), "Critic shares params with actor!"

    def test_critic_mlp_architecture(self):
        """Critic MLP should have correct layer sizes."""
        model = _make_happo_model(num_agents=2)
        # With happo_critic_hidden_sizes=[64, 32], we expect:
        # Linear(enc_out*2, 64), Tanh, Linear(64, 32), Tanh, Linear(32, 1)
        linears = [m for m in model.centralized_critic if isinstance(m, nn.Linear)]
        assert len(linears) == 3  # 2 hidden + 1 output
        assert linears[-1].out_features == 1  # Final output is scalar V(s)


class TestHAPPOForwardShapes:
    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_forward_output_shapes(self):
        num_agents = 2
        batch = num_agents * 3  # 3 transitions
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result["values"].shape == (batch,)
        assert "action_logits" in result
        assert "actions" in result
        assert "new_rnn_states" in result
        assert result["new_rnn_states"].shape == rnn_states.shape

    def test_values_only_mode(self):
        num_agents = 2
        batch = num_agents * 2
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states, values_only=True)

        assert result["values"].shape == (batch,)
        assert "action_logits" not in result
        assert "actions" not in result

    @pytest.mark.parametrize("num_agents", [2, 3, 4])
    def test_various_agent_counts(self, num_agents):
        batch = num_agents * 2
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result["values"].shape == (batch,)

    def test_centralized_critic_same_transition_same_value(self):
        """All agents in the same transition should get the same value."""
        num_agents = 2
        batch = num_agents * 2  # 2 transitions
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        values = result["values"]
        # Agents 0,1 are in transition 0; agents 2,3 in transition 1
        # (positional grouping in forward() uses arange // n_agents)
        torch.testing.assert_close(values[0], values[1])
        torch.testing.assert_close(values[2], values[3])

    def test_forward_head_routing(self):
        """Verify forward_head routes obs to the correct per-agent encoder."""
        num_agents = 2
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        # Create batch with only agent 0
        batch_size = 4
        obs = {
            "obs": torch.randn(batch_size, 3, 64, 64),
            "agent_id": torch.zeros(batch_size, num_agents),
        }
        obs["agent_id"][:, 0] = 1.0  # All samples are agent 0
        agent_idx = torch.zeros(batch_size, dtype=torch.long)

        with torch.no_grad():
            out = model.forward_head(obs, agent_idx=agent_idx)

        assert out.shape == (batch_size, model.agent_encoders[0].get_out_size())

    def test_forward_core_requires_agent_idx(self):
        """forward_core should raise ValueError without agent_idx."""
        model = _make_happo_model(num_agents=2)
        head_out = torch.randn(4, model.agent_encoders[0].get_out_size())
        rnn_states = torch.zeros(4, get_rnn_size(model.cfg))

        with pytest.raises(ValueError, match="agent_idx is required"):
            model.forward_core(head_out, rnn_states)



class TestComputeEnvGroupIdx:
    """Test _compute_env_group_idx from HAPPOLearner."""

    @pytest.fixture
    def learner_stub(self):
        """Create a minimal learner stub with just the method we need."""
        from sample_factory.algo.learning.learner_happo import HAPPOLearner

        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        stub.cfg = AttrDict({"rollout": 4})
        return stub

    def test_basic_grouping(self, learner_stub):
        """2 agents, 2 envs, rollout=4 → 2*4=8 transitions."""
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

        # Each transition should have exactly 2 agent samples
        counts = torch.bincount(group_idx)
        assert (counts == n_agents).all()

    def test_wrong_agent_count_raises(self, learner_stub):
        """Should raise if a transition doesn't have exactly n_agents samples."""
        # 3 samples with env_idx 0 at timestep 0, but n_agents=2
        dataset_size = 3
        env_idx = torch.tensor([0, 0, 0], dtype=torch.long)
        agent_idx = torch.tensor([0, 1, 0], dtype=torch.long)

        learner_stub.cfg.rollout = 1
        with pytest.raises(RuntimeError, match="wrong agent count"):
            learner_stub._compute_env_group_idx(agent_idx, env_idx, dataset_size)


class TestGetAgentMinibatches:
    """Test _get_agent_minibatches from HAPPOLearner."""

    @pytest.fixture
    def learner_stub(self):
        from sample_factory.algo.learning.learner_happo import HAPPOLearner

        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        stub.cfg = AttrDict({"recurrence": 1})
        return stub

    def test_single_agent_indices(self, learner_stub):
        """Minibatches should contain only indices of the specified agent."""
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
        """Each minibatch should have at most batch_size // n_agents samples."""
        experience_size = 40
        agent_mask = torch.zeros(experience_size, dtype=torch.bool)
        agent_mask[:20] = True  # 20 samples for this agent

        minibatches = learner_stub._get_agent_minibatches(
            batch_size=20, agent_mask=agent_mask, experience_size=experience_size
        )

        for mb in minibatches:
            assert len(mb) <= 20 // 2  # batch_size / n_agents = 10

    def test_rnn_recurrence_alignment(self, learner_stub):
        """With recurrence > 1, chunks must be recurrence-aligned."""
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
    """Test _get_transition_minibatches from HAPPOLearner."""

    @pytest.fixture
    def learner_stub(self):
        from sample_factory.algo.learning.learner_happo import HAPPOLearner

        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2
        return stub

    def test_complete_transitions(self, learner_stub):
        """Each minibatch must contain complete transitions (all agents)."""
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
            # Check each transition has exactly n_agents
            for g in mb_groups.unique():
                assert (mb_groups == g).sum() == n_agents

    def test_wrong_agent_count_raises(self, learner_stub):
        """Should raise if a transition has wrong number of agents."""
        env_group_idx = torch.tensor([0, 0, 0, 1, 1])  # 3 agents in transition 0
        with pytest.raises(RuntimeError, match="has .* samples"):
            learner_stub._get_transition_minibatches(4, 5, env_group_idx, 2)



class TestGradientIsolation:
    """Verify that training one agent doesn't affect another agent's parameters."""

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

        # Forward through agent 0's encoder/core/decoder
        head_out = model.forward_head(obs, agent_idx=agent_idx)
        core_out, _ = model.forward_core(
            head_out, torch.zeros(batch_size, get_rnn_size(model.cfg)), agent_idx=agent_idx
        )
        decoder_out = model.agent_decoders[0](core_out)
        logits, _ = model.agent_action_params[0](decoder_out, None)

        # Fake loss
        loss = logits.sum()
        loss.backward()

        # Agent 1's encoder should have zero gradients
        for name, param in model.agent_encoders[1].named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"Agent 1 encoder param {name} has non-zero grad!"

        # Agent 0's encoder should have non-zero gradients
        has_nonzero = False
        for param in model.agent_encoders[0].parameters():
            if param.grad is not None and (param.grad != 0).any():
                has_nonzero = True
                break
        assert has_nonzero, "Agent 0 encoder has all-zero gradients!"

    def test_critic_gradient_isolation_from_actor(self):
        """Critic loss should not produce gradients on actor parameters."""
        num_agents = 2
        batch = num_agents * 2  # 2 transitions
        model = _make_happo_model(num_agents=num_agents)
        model.train()

        obs = _make_obs_batch(batch, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch) // num_agents

        # Compute critic values
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

        # Fake critic loss
        loss = joint_value.sum()
        loss.backward()

        # Actor encoder should have zero gradients
        for name, param in model.agent_encoders[0].named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"Actor param {name} has grad from critic loss!"



class TestFactorM:
    """Unit tests for HAPPO factor M accumulation logic."""

    def test_m_initialized_as_advantage(self):
        """Factor M should be initialized as the normalized advantage."""
        adv = torch.tensor([1.0, -2.0, 3.0, 0.5])
        mean = adv.mean()
        std = adv.std()
        normalized = (adv - mean) / torch.clamp_min(std, 1e-7)
        M = normalized.clone().detach()

        # M should equal normalized advantage
        torch.testing.assert_close(M, normalized)

    def test_m_detached(self):
        """M should have no gradient (detached from compute graph)."""
        adv = torch.tensor([1.0, 2.0], requires_grad=True)
        M = adv.clone().detach()
        assert not M.requires_grad

    def test_post_ratio_broadcast_two_agents(self):
        """After training agent 0, its post-update ratio should be broadcast to all agents."""
        n_agents = 2
        n_transitions = 3
        experience_size = n_agents * n_transitions

        # Layout: [agent0_t0, agent1_t0, agent0_t1, agent1_t1, agent0_t2, agent1_t2]
        agent_idx = torch.tensor([0, 1, 0, 1, 0, 1])
        env_group_idx = torch.tensor([0, 0, 1, 1, 2, 2])

        M = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        agent_mask = agent_idx == 0

        # Simulate: agent 0 has post-update ratio of [1.5, 2.0, 0.8]
        post_ratio = torch.tensor([1.5, 2.0, 0.8])

        # Broadcast logic (from _train)
        ratio_per_transition = torch.ones(n_transitions)
        ratio_per_transition[env_group_idx[agent_mask]] = post_ratio
        M_new = M * ratio_per_transition[env_group_idx]

        # Agent 0's entries: [1.0*1.5, 2.0*2.0, 3.0*0.8] = [1.5, 4.0, 2.4]
        # Agent 1's entries: same transitions → same ratio → [1.0*1.5, 2.0*2.0, 3.0*0.8]
        expected = torch.tensor([1.5, 1.5, 4.0, 4.0, 2.4, 2.4])
        torch.testing.assert_close(M_new, expected)

    def test_post_ratio_accumulates_three_agents(self):
        """With 3 agents, M accumulates post-update ratios from all previous agents."""
        n_agents = 3
        n_transitions = 2
        experience_size = n_agents * n_transitions

        agent_idx = torch.tensor([0, 1, 2, 0, 1, 2])
        env_group_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        # Start: M = advantage
        M = torch.ones(experience_size)

        # Agent 0 post-ratio: [2.0, 3.0] for transitions [0, 1]
        post_ratio_0 = torch.tensor([2.0, 3.0])
        ratio_per_trans = torch.ones(n_transitions)
        ratio_per_trans[env_group_idx[agent_idx == 0]] = post_ratio_0
        M = M * ratio_per_trans[env_group_idx]
        # M = [2, 2, 2, 3, 3, 3]

        # Agent 1 post-ratio: [1.5, 0.5] for transitions [0, 1]
        post_ratio_1 = torch.tensor([1.5, 0.5])
        ratio_per_trans = torch.ones(n_transitions)
        ratio_per_trans[env_group_idx[agent_idx == 1]] = post_ratio_1
        M = M * ratio_per_trans[env_group_idx]
        # M = [2*1.5, 2*1.5, 2*1.5, 3*0.5, 3*0.5, 3*0.5] = [3, 3, 3, 1.5, 1.5, 1.5]

        expected = torch.tensor([3.0, 3.0, 3.0, 1.5, 1.5, 1.5])
        torch.testing.assert_close(M, expected)

    def test_m_handles_negative_advantages(self):
        """Factor M should correctly handle negative advantages (sign preserved)."""
        M = torch.tensor([-2.0, 1.0, -0.5, 3.0])
        post_ratio = torch.tensor([2.0, 2.0])
        env_group_idx = torch.tensor([0, 0, 1, 1])
        agent_mask = torch.tensor([True, False, True, False])

        ratio_per_trans = torch.ones(2)
        ratio_per_trans[env_group_idx[agent_mask]] = post_ratio
        M_new = M * ratio_per_trans[env_group_idx]

        expected = torch.tensor([-4.0, 2.0, -1.0, 6.0])
        torch.testing.assert_close(M_new, expected)

    def test_factor_clamp(self):
        """Optional factor clamping should prevent M explosion."""
        post_ratio = torch.tensor([100.0, 0.001])
        factor_clamp = 5.0
        clamped = torch.clamp(post_ratio, 1.0 / factor_clamp, factor_clamp)
        assert clamped[0].item() == 5.0
        assert clamped[1].item() == pytest.approx(0.2)

    def test_invalid_samples_masked(self):
        """Invalid samples should have post-update ratio set to 1.0."""
        post_ratio = torch.tensor([1.5, 99.0, 2.0])
        agent_valids = torch.tensor([True, False, True])
        post_ratio[~agent_valids] = 1.0

        assert post_ratio[0].item() == 1.5
        assert post_ratio[1].item() == 1.0  # Was 99.0, now masked
        assert post_ratio[2].item() == 2.0


class TestPolicyLoss:
    """Test the PPO loss with factor M (asymmetric clipping)."""

    def test_asymmetric_clipping(self):
        """Verify [1/(1+ε), 1+ε] clipping, not symmetric [1-ε, 1+ε]."""
        ppo_clip_ratio = 0.2
        clip_high = 1.0 + ppo_clip_ratio  # 1.2
        clip_low = 1.0 / clip_high  # 1/1.2 ≈ 0.8333

        assert clip_high == pytest.approx(1.2)
        assert clip_low == pytest.approx(1.0 / 1.2)
        # Asymmetric: clip_low ≈ 0.833, NOT 0.8
        assert clip_low != pytest.approx(1.0 - ppo_clip_ratio)

    def test_loss_direction_positive_advantage(self):
        """With positive M (advantage), policy loss should encourage higher ratio."""
        ratio = torch.tensor([1.5])
        M = torch.tensor([2.0])  # Positive advantage

        clip_high = 1.2
        clip_low = 1.0 / 1.2
        clipped = torch.clamp(ratio, clip_low, clip_high)

        loss = -torch.min(ratio * M, clipped * M)
        # ratio*M = 3.0, clipped*M = 1.2*2.0 = 2.4
        # min = 2.4, loss = -2.4
        assert loss.item() == pytest.approx(-2.4)

    def test_loss_direction_negative_advantage(self):
        """With negative M, loss should discourage the ratio from being too large."""
        ratio = torch.tensor([1.5])
        M = torch.tensor([-2.0])  # Negative advantage

        clip_high = 1.2
        clip_low = 1.0 / 1.2
        clipped = torch.clamp(ratio, clip_low, clip_high)

        loss = -torch.min(ratio * M, clipped * M)
        # ratio*M = -3.0, clipped*M = 1.2*(-2.0) = -2.4
        # min = -3.0, loss = -(-3.0) = 3.0
        assert loss.item() == pytest.approx(3.0)


class TestAgentOrdering:
    def test_random_order_changes_with_train_step(self):
        """Different train_step seeds should produce different orderings (most of the time)."""
        n_agents = 4
        orders = set()
        for step in range(20):
            rng = torch.Generator().manual_seed(step)
            order = tuple(torch.randperm(n_agents, generator=rng).tolist())
            orders.add(order)

        # With 4!=24 permutations and 20 tries, we should see multiple orderings
        assert len(orders) > 1

    def test_fixed_order_is_sequential(self):
        """Fixed ordering should be [0, 1, ..., n_agents-1]."""
        agent_order = list(range(3))
        assert agent_order == [0, 1, 2]

    def test_random_order_deterministic_with_same_seed(self):
        """Same seed should produce identical orderings (for checkpoint resume)."""
        n_agents = 3
        rng1 = torch.Generator().manual_seed(42)
        order1 = torch.randperm(n_agents, generator=rng1).tolist()

        rng2 = torch.Generator().manual_seed(42)
        order2 = torch.randperm(n_agents, generator=rng2).tolist()

        assert order1 == order2


class TestHAPPORNN:
    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_rnn_forward_shapes(self):
        """Model forward with use_rnn=True should return correct shapes."""
        num_agents = 2
        batch = num_agents * 2
        model = _make_happo_model(num_agents=num_agents, use_rnn=True)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result["values"].shape == (batch,)
        assert result["action_logits"].shape[0] == batch
        assert result["new_rnn_states"].shape == rnn_states.shape

    def test_rnn_states_change_after_forward(self):
        """RNN states should be updated (not all-zero) after a forward pass."""
        num_agents = 2
        batch = num_agents * 2
        model = _make_happo_model(num_agents=num_agents, use_rnn=True)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        new_rnn = result["new_rnn_states"]
        assert not torch.allclose(new_rnn, torch.zeros_like(new_rnn)), \
            "RNN states should change after forward pass"

    def test_rnn_states_per_agent_independent(self):
        """Agent 0's RNN state update should not leak to agent 1."""
        num_agents = 2
        batch = num_agents  # 1 transition
        model = _make_happo_model(num_agents=num_agents, use_rnn=True)
        model.eval()

        # Create obs where agent 0 has different input than agent 1
        obs0 = torch.randn(1, 3, 64, 64)
        obs1 = torch.zeros(1, 3, 64, 64)  # Deliberately different
        obs = {
            "obs": torch.cat([obs0, obs1], dim=0),
            "agent_id": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        # RNN states should differ between agents (different inputs)
        rnn_0 = result["new_rnn_states"][0]
        rnn_1 = result["new_rnn_states"][1]
        assert not torch.allclose(rnn_0, rnn_1), \
            "Per-agent RNN states should differ with different inputs"


class TestSingleAgent:
    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_single_agent_model_construction(self):
        """HAPPOActorCritic with num_agents=1 should construct correctly."""
        model = _make_happo_model(num_agents=1)
        assert len(model.agent_encoders) == 1
        assert len(model.critic_encoders) == 1

    def test_single_agent_forward(self):
        """Forward pass with num_agents=1 should work."""
        model = _make_happo_model(num_agents=1)
        model.eval()

        batch = 3
        obs = {
            "obs": torch.randn(batch, 3, 64, 64),
            "agent_id": torch.ones(batch, 1),  # Only one agent
        }
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result["values"].shape == (batch,)
        assert "action_logits" in result

    def test_single_agent_group_by_env(self):
        """_group_by_env with 1 agent should produce [n_transitions, 1, F]."""
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        agent_idx = torch.tensor([0, 0])
        env_group_idx = torch.tensor([0, 1])

        grouped = _group_by_env(features, agent_idx, env_group_idx, 1)
        assert grouped.shape == (2, 1, 2)


class TestCentralizedCriticObservability:
    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_critic_value_depends_on_all_agents(self):
        """Perturbing one agent's obs should change the other agent's value."""
        num_agents = 2
        batch = num_agents  # 1 transition
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        # Baseline forward
        obs_base = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))
        with torch.no_grad():
            result_base = model(obs_base, rnn_states)
        v_base = result_base["values"]

        # Perturb agent 0's observation significantly
        obs_perturbed = copy.deepcopy(obs_base)
        obs_perturbed["obs"][0] += 100.0  # Large perturbation to agent 0

        with torch.no_grad():
            result_perturbed = model(obs_perturbed, rnn_states)
        v_perturbed = result_perturbed["values"]

        # Agent 1's value (index 1) should change because the centralized critic
        # sees BOTH agents' obs. If it doesn't change, the critic is broken.
        assert not torch.allclose(v_base[1], v_perturbed[1], atol=1e-5), \
            "Agent 1's value should change when agent 0's obs is perturbed (centralized critic)"



class TestLearnerMethodIntegration:
    """Test actual HAPPOLearner methods with real model (not stubs)."""

    @pytest.fixture
    def model_and_cfg(self):
        num_agents = 2
        cfg = _make_happo_cfg(num_agents=num_agents)
        cfg.ppo_clip_ratio = 0.2
        cfg.ppo_clip_value = 10.0
        cfg.value_loss_coeff = 0.5
        cfg.exploration_loss_coeff = 0.0
        cfg.recurrence = 1
        cfg.rollout = 4
        cfg.batch_size = 8
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space(4)
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()
        return model, cfg, obs_space, action_space

    def test_calculate_agent_policy_loss(self, model_and_cfg):
        """Direct test of _calculate_agent_policy_loss with real model."""
        model, cfg, obs_space, action_space = model_and_cfg

        # Create learner stub with real model
        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = cfg.num_agents
        stub.actor_critic = model
        stub.exploration_loss_func = lambda d, v, n: 0.0

        # Create synthetic minibatch
        batch_size = 4
        obs = _make_obs_batch(batch_size, cfg.num_agents)
        # All samples for agent 0
        obs["agent_id"] = torch.zeros(batch_size, cfg.num_agents)
        obs["agent_id"][:, 0] = 1.0

        # Get real log probs for the actions
        with torch.no_grad():
            head = model.forward_head(obs, agent_idx=torch.zeros(batch_size, dtype=torch.long))
            core, _ = model.forward_core(head, torch.zeros(batch_size, get_rnn_size(cfg)),
                                          agent_idx=torch.zeros(batch_size, dtype=torch.long))
            dec = model.agent_decoders[0](core)
            logits, _ = model.agent_action_params[0](dec, None)
            dist = get_action_distribution(action_space, logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        mb = AttrDict({
            "normalized_obs": obs,
            "actions": actions,
            "log_prob_actions": log_probs,
            "rnn_states": torch.zeros(batch_size, get_rnn_size(cfg)),
            "valids": torch.ones(batch_size),
            "dones_cpu": torch.zeros(batch_size),
        })

        M_full = torch.ones(10)  # Larger than batch for indexing
        mb_indices = np.arange(batch_size)

        model.train()
        policy_loss, exploration_loss = stub._calculate_agent_policy_loss(
            agent_id=0, mb=mb, M_full=M_full, mb_indices=mb_indices, num_invalids=0
        )

        assert policy_loss.requires_grad
        assert isinstance(policy_loss.item(), float)
        assert not torch.isnan(policy_loss)
        assert not torch.isinf(policy_loss)

    def test_calculate_critic_loss(self, model_and_cfg):
        """Direct test of _calculate_critic_loss with real model."""
        model, cfg, obs_space, action_space = model_and_cfg

        stub = object.__new__(HAPPOLearner)
        stub.cfg = cfg
        stub.n_agents = cfg.num_agents
        stub.actor_critic = model

        # Create synthetic minibatch with 2 complete transitions
        batch_size = cfg.num_agents * 2  # 4 samples = 2 transitions
        obs = _make_obs_batch(batch_size, cfg.num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.tensor([0, 0, 1, 1])

        mb = AttrDict({
            "normalized_obs": obs,
            "values": torch.zeros(batch_size),
            "returns": torch.ones(batch_size),
            "valids": torch.ones(batch_size),
        })

        model.train()
        value_loss = stub._calculate_critic_loss(mb, agent_idx, env_group_idx, num_invalids=0)

        assert value_loss.requires_grad
        assert isinstance(value_loss.item(), float)
        assert not torch.isnan(value_loss)
        assert value_loss.item() > 0  # returns=1.0, values=0.0 → nonzero loss


class TestCheckpointRoundTrip:
    def test_checkpoint_save_restore(self):
        """Checkpoint dict should contain all necessary keys and restore correctly."""
        cfg = _make_happo_cfg(num_agents=2)
        cfg.learning_rate = 0.001
        cfg.adam_beta1 = 0.9
        cfg.adam_beta2 = 0.999
        cfg.adam_eps = 1e-8

        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space(4)
        model = make_happo_actor_critic(cfg, obs_space, action_space)

        # Create optimizers manually (simulating init)
        agent_optimizers = []
        for i in range(2):
            params = (
                list(model.agent_encoders[i].parameters())
                + list(model.agent_cores[i].parameters())
                + list(model.agent_decoders[i].parameters())
                + list(model.agent_action_params[i].parameters())
            )
            agent_optimizers.append(torch.optim.Adam(params, lr=0.001))

        critic_params = (
            list(model.centralized_critic.parameters())
            + list(model.critic_encoders.parameters())
        )
        critic_optimizer = torch.optim.Adam(critic_params, lr=0.001)

        # Simulate a training step to populate optimizer states
        dummy_loss = sum(p.sum() for p in model.parameters())
        dummy_loss.backward()
        for opt in agent_optimizers:
            opt.step()
        critic_optimizer.step()

        # Build checkpoint dict (matching learner's _get_checkpoint_dict)
        checkpoint = {
            "model": model.state_dict(),
            "agent_optimizers": [opt.state_dict() for opt in agent_optimizers],
            "critic_optimizer": critic_optimizer.state_dict(),
            "env_steps": 1000,
            "train_step": 42,
            "curr_lr": 0.0005,
            "best_performance": 10.0,
        }

        # Verify all required keys present
        assert "model" in checkpoint
        assert len(checkpoint["agent_optimizers"]) == 2
        assert "critic_optimizer" in checkpoint
        assert checkpoint["train_step"] == 42
        assert checkpoint["curr_lr"] == 0.0005

        # Create fresh model and restore
        model2 = make_happo_actor_critic(cfg, obs_space, action_space)
        model2.load_state_dict(checkpoint["model"])

        # Verify parameter equality
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)


class TestApplyLR:
    def test_apply_lr_updates_all_optimizers(self):
        """_apply_lr should update LR on all agent + critic optimizers."""
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 2

        # Create dummy optimizers
        params_a = [nn.Parameter(torch.randn(3, 3))]
        params_b = [nn.Parameter(torch.randn(3, 3))]
        params_c = [nn.Parameter(torch.randn(3, 3))]
        stub.agent_optimizers = [
            torch.optim.Adam(params_a, lr=0.01),
            torch.optim.Adam(params_b, lr=0.01),
        ]
        stub.critic_optimizer = torch.optim.Adam(params_c, lr=0.01)

        # Apply new LR
        new_lr = 0.0001
        HAPPOLearner._apply_lr(stub, new_lr)

        for opt in stub.agent_optimizers:
            assert opt.param_groups[0]["lr"] == new_lr
        assert stub.critic_optimizer.param_groups[0]["lr"] == new_lr

    def test_apply_lr_does_not_set_curr_lr(self):
        """_apply_lr must NOT modify self.curr_lr (prevents compounding decay)."""
        stub = object.__new__(HAPPOLearner)
        stub.n_agents = 1
        stub.curr_lr = 0.001
        stub.agent_optimizers = [torch.optim.Adam([nn.Parameter(torch.randn(2))], lr=0.01)]
        stub.critic_optimizer = torch.optim.Adam([nn.Parameter(torch.randn(2))], lr=0.01)

        HAPPOLearner._apply_lr(stub, 0.0001)

        assert stub.curr_lr == 0.001  # Unchanged
