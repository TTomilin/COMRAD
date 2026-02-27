import copy

import pytest
import torch
import torch.nn as nn

from sample_factory.algo.utils.context import sf_global_context
from sample_factory.model.model_utils import get_rnn_size

from sf.doom.happo_model import _group_by_env, make_happo_actor_critic

from .conftest import (
    _make_happo_cfg,
    _make_happo_model,
    _make_obs_batch,
    _make_obs_space,
    _make_action_space,
)


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
        model = _make_happo_model(num_agents=2)
        params_0 = set(id(p) for p in model.agent_encoders[0].parameters())
        params_1 = set(id(p) for p in model.agent_encoders[1].parameters())
        assert params_0.isdisjoint(params_1), "Agent encoders share parameters!"

    def test_critic_params_separate_from_actor(self):
        model = _make_happo_model(num_agents=2)
        actor_params = set(id(p) for p in model.agent_encoders[0].parameters())
        critic_params = set(id(p) for p in model.critic_encoders[0].parameters())
        assert actor_params.isdisjoint(critic_params), "Critic shares params with actor!"

    def test_critic_mlp_architecture(self):
        model = _make_happo_model(num_agents=2)
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
        num_agents = 2
        batch = num_agents * 2  # 2 transitions
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        values = result["values"]
        torch.testing.assert_close(values[0], values[1])
        torch.testing.assert_close(values[2], values[3])

    def test_forward_head_routing(self):
        num_agents = 2
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

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
        model = _make_happo_model(num_agents=2)
        head_out = torch.randn(4, model.agent_encoders[0].get_out_size())
        rnn_states = torch.zeros(4, get_rnn_size(model.cfg))

        with pytest.raises(ValueError, match="agent_idx is required"):
            model.forward_core(head_out, rnn_states)


class TestHAPPORNN:
    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_rnn_forward_shapes(self):
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
        model = _make_happo_model(num_agents=1)
        assert len(model.agent_encoders) == 1
        assert len(model.critic_encoders) == 1

    def test_single_agent_forward(self):
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
        num_agents = 2
        batch = num_agents  # 1 transition
        model = _make_happo_model(num_agents=num_agents)
        model.eval()

        obs_base = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(model.cfg))
        with torch.no_grad():
            result_base = model(obs_base, rnn_states)
        v_base = result_base["values"]

        obs_perturbed = copy.deepcopy(obs_base)
        obs_perturbed["obs"][0] += 100.0  # Large perturbation to agent 0

        with torch.no_grad():
            result_perturbed = model(obs_perturbed, rnn_states)
        v_perturbed = result_perturbed["values"]

        assert not torch.allclose(v_base[1], v_perturbed[1], atol=1e-5), \
            "Agent 1's value should change when agent 0's obs is perturbed (centralized critic)"
