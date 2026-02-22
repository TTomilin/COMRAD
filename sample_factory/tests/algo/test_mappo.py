from __future__ import annotations

import gymnasium as gym
import pytest
import torch
from torch import nn

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.mappo_model import MAPPOActorCritic, make_mappo_actor_critic


def _make_cfg(
    *,
    algo: str = "MAPPO",
    num_agents: int = 2,
    use_rnn: bool = False,
    rnn_type: str = "gru",
    rnn_size: int = 64,
    rnn_num_layers: int = 1,
) -> AttrDict:
    return AttrDict(
        {
            "encoder_conv_architecture": "convnet_simple",
            "encoder_conv_mlp_layers": [],
            "encoder_extra_fc_layers": 0,
            "hidden_size": 32,
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
            "algo": algo,
            "adaptive_stddev": True,
            "initial_stddev": 1.0,
            "policy_initialization": "orthogonal",
            "policy_init_gain": 1.0,
            "actor_critic_share_weights": True,
        }
    )


def _make_spaces():
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))})
    action_space = gym.spaces.Discrete(4)
    return obs_space, action_space


def _make_mappo(*, algo: str = "MAPPO", num_agents: int = 2, use_rnn: bool = False, **kw):
    cfg = _make_cfg(algo=algo, num_agents=num_agents, use_rnn=use_rnn, **kw)
    obs_space, action_space = _make_spaces()
    return make_mappo_actor_critic(cfg, obs_space, action_space)


class TestMAPPOConstruction:
    """Validates that MAPPO and IPPO models are constructed correctly."""

    def test_mappo_has_centralized_critic(self):
        ac = _make_mappo(num_agents=2)
        assert hasattr(ac, "centralized_critic")
        assert ac.centralized_critic is not None
        assert ac.critic_linear is None

    def test_ippo_has_decentralized_critic(self):
        ac = _make_mappo(algo="APPO", num_agents=2)
        assert ac.critic_linear is not None
        assert not hasattr(ac, "centralized_critic") or ac.use_centralized_critic is False

    def test_mappo_centralized_critic_input_dim(self):
        """Centralized critic input = decoder_out_size * num_agents."""
        ac = _make_mappo(num_agents=3)
        decoder_out = ac.decoder.get_out_size()
        first_layer = ac.centralized_critic[0]
        assert first_layer.in_features == decoder_out * 3

    def test_mappo_centralized_critic_output_dim(self):
        """Centralized critic outputs one value per agent."""
        ac = _make_mappo(num_agents=3)
        last_layer = ac.centralized_critic[-1]
        assert last_layer.out_features == 3

    def test_mappo_n_agents_stored(self):
        ac = _make_mappo(num_agents=4)
        assert ac.n_agents == 4


class TestMAPPOForwardShapes:
    """Validates tensor shapes through the MAPPO forward pass."""

    def test_forward_output_keys(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4  # must be divisible by num_agents
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert "values" in result
        assert "action_logits" in result
        assert "actions" in result
        assert "new_rnn_states" in result

    def test_values_shape_matches_batch(self):
        """Values should be [batch_size * n_agents] = [batch_size]."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 6
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)

    def test_action_logits_shape(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        # Discrete(4) => 4 logits
        assert result["action_logits"].shape == (batch, 4)

    def test_actions_shape(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["actions"].shape == (batch,)

    @pytest.mark.parametrize("num_agents", [2, 3, 4])
    def test_various_agent_counts(self, num_agents):
        ac = _make_mappo(num_agents=num_agents)
        ac.eval()
        batch = num_agents * 2
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)
        assert result["action_logits"].shape == (batch, 4)



class TestCentralizedCritic:
    """Validates that MAPPO uses centralized critic while IPPO uses decentralized."""

    def test_mappo_values_depend_on_all_agents(self):
        """Changing one agent's observation should affect ALL agents' values in MAPPO."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        obs_a = {"obs": torch.randn(2, 3, 64, 64)}
        obs_b = {"obs": obs_a["obs"].clone()}
        obs_b["obs"][0] += 10.0  # Perturb agent 0's observation

        rnn_states = torch.zeros(2, get_rnn_size(ac.cfg))
        with torch.no_grad():
            val_a = ac(obs_a, rnn_states)["values"]
            val_b = ac(obs_b, rnn_states)["values"]

        # Agent 1's value should change too (centralized critic sees all agents)
        assert not torch.allclose(val_b[1:2], val_a[1:2], atol=1e-5), (
            "MAPPO centralized critic: agent 1's value should change when agent 0's obs changes"
        )

    def test_ippo_values_independent_per_agent(self):
        """In IPPO, each agent's value depends only on its own observation."""
        ac = _make_mappo(algo="APPO", num_agents=2)
        ac.eval()
        obs_a = {"obs": torch.randn(2, 3, 64, 64)}
        obs_b = {"obs": obs_a["obs"].clone()}
        obs_b["obs"][0] += 10.0  # Perturb agent 0

        rnn_states = torch.zeros(2, get_rnn_size(ac.cfg))
        with torch.no_grad():
            val_a = ac(obs_a, rnn_states)["values"]
            val_b = ac(obs_b, rnn_states)["values"]

        # Agent 1's value should NOT change (decentralized critic)
        assert torch.allclose(val_b[1:2], val_a[1:2], atol=1e-6), (
            "IPPO decentralized critic: agent 1's value should be unchanged when agent 0's obs changes"
        )

    def test_mappo_actions_still_decentralized(self):
        """Even in MAPPO, action logits depend only on the agent's own observation."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        obs_a = {"obs": torch.randn(2, 3, 64, 64)}
        obs_b = {"obs": obs_a["obs"].clone()}
        obs_b["obs"][0] += 10.0  # Perturb agent 0

        rnn_states = torch.zeros(2, get_rnn_size(ac.cfg))
        with torch.no_grad():
            logits_a = ac(obs_a, rnn_states)["action_logits"]
            logits_b = ac(obs_b, rnn_states)["action_logits"]

        # Agent 1's logits should stay the same (actions are decentralized)
        assert torch.allclose(logits_b[1:2], logits_a[1:2], atol=1e-6), (
            "MAPPO actions should be decentralized: agent 1's logits should not change"
        )

    def test_values_only_mode(self):
        """forward with values_only=True should only return values."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states, values_only=True)
        assert "values" in result
        assert "action_logits" not in result

class TestMAPPORNN:
    """Validates that MAPPO works with RNN (GRU) enabled."""

    def test_rnn_forward_shapes(self):
        ac = _make_mappo(num_agents=2, use_rnn=True)
        ac.eval()
        batch = 4
        rnn_size = get_rnn_size(ac.cfg)
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, rnn_size)
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)
        assert result["new_rnn_states"].shape == (batch, rnn_size)

    def test_rnn_states_change_after_forward(self):
        """RNN hidden states should be updated after a forward pass."""
        ac = _make_mappo(num_agents=2, use_rnn=True)
        ac.eval()
        batch = 2
        rnn_size = get_rnn_size(ac.cfg)
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, rnn_size)
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert not torch.allclose(result["new_rnn_states"], rnn_states), (
            "RNN states should change after forward pass"
        )

    def test_rnn_multi_layer(self):
        """Multi-layer GRU should work correctly."""
        ac = _make_mappo(num_agents=2, use_rnn=True, rnn_num_layers=2)
        ac.eval()
        batch = 4
        rnn_size = get_rnn_size(ac.cfg)
        assert rnn_size == 64 * 2  # rnn_size * rnn_num_layers
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, rnn_size)
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["new_rnn_states"].shape == (batch, rnn_size)

    def test_ippo_rnn_forward(self):
        """IPPO with RNN should also work (decentralized critic + RNN)."""
        ac = _make_mappo(algo="APPO", num_agents=2, use_rnn=True)
        ac.eval()
        batch = 4
        rnn_size = get_rnn_size(ac.cfg)
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, rnn_size)
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)
        assert result["new_rnn_states"].shape == (batch, rnn_size)


class TestMAPPOActionMask:
    """Validates action masking works correctly for MAPPO."""

    def test_mask_does_not_change_values(self):
        """Action mask should not affect value computation."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        mask = torch.ones(batch, 4)
        mask[:, 0] = 0.0  # Block action 0

        with torch.no_grad():
            result_no_mask = ac(obs, rnn_states, action_mask=None)
            result_mask = ac(obs, rnn_states, action_mask=mask)

        assert torch.allclose(result_no_mask["values"], result_mask["values"], atol=1e-6), (
            "Action mask should not affect value computation"
        )

    def test_mask_blocks_actions(self):
        """Masked actions should have zero probability in the distribution."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 2
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        mask = torch.ones(batch, 4)
        mask[:, 2] = 0.0  # Block action 2

        with torch.no_grad():
            _ = ac(obs, rnn_states, action_mask=mask)

        # The mask is applied inside the distribution, not on raw logits
        dist = ac.action_distribution()
        probs = dist.probs if hasattr(dist, "probs") else dist.distribution.probs
        assert (probs[:, 2] < 1e-6).all(), (
            "Masked action should have ~zero probability in the distribution"
        )

    def test_all_ones_mask_no_effect(self):
        """All-ones mask should produce identical output to no mask."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        mask = torch.ones(batch, 4)

        with torch.no_grad():
            out_none = ac(obs, rnn_states, action_mask=None)
            out_ones = ac(obs, rnn_states, action_mask=mask)

        assert torch.allclose(out_none["action_logits"], out_ones["action_logits"], atol=1e-6)


class TestMAPPOFactory:
    """Validates the make_mappo_actor_critic factory function."""

    def test_factory_returns_correct_type(self):
        ac = _make_mappo()
        assert isinstance(ac, MAPPOActorCritic)

    def test_factory_ippo_returns_correct_type(self):
        ac = _make_mappo(algo="APPO")
        assert isinstance(ac, MAPPOActorCritic)
        assert ac.use_centralized_critic is False

    def test_factory_handles_tuple_action_space(self):
        """Multi-head action space (Tuple of Discrete) should work."""
        cfg = _make_cfg(num_agents=2)
        obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, shape=(3, 64, 64))})
        action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
        ac = make_mappo_actor_critic(cfg, obs_space, action_space)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        # Tuple(Discrete(3), Discrete(2)) -> 5 logits
        assert result["action_logits"].shape == (batch, 5)


class TestMAPPOGradients:
    """Validates gradient flow through the MAPPO model."""

    def test_centralized_critic_receives_gradients(self):
        """Centralized critic parameters should receive gradients from value loss."""
        ac = _make_mappo(num_agents=2)
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        result = ac(obs, rnn_states)

        # Simulate value loss
        target_values = torch.zeros(batch)
        value_loss = (result["values"] - target_values).pow(2).mean()
        value_loss.backward()

        for name, param in ac.centralized_critic.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_encoder_receives_gradients_from_value_loss(self):
        """Encoder should get gradients through centralized critic path."""
        ac = _make_mappo(num_agents=2)
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        result = ac(obs, rnn_states)

        target_values = torch.zeros(batch)
        value_loss = (result["values"] - target_values).pow(2).mean()
        value_loss.backward()

        encoder_has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in ac.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should receive gradients from centralized critic"

    def test_encoder_receives_gradients_from_policy_loss(self):
        """Encoder should get gradients from action logits (policy loss)."""
        ac = _make_mappo(num_agents=2)
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        result = ac(obs, rnn_states, sample_actions=False)

        # Simulate policy loss via logits
        policy_loss = result["action_logits"].sum()
        policy_loss.backward()

        encoder_has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in ac.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should receive gradients from policy loss"


class TestMAPPODeterminism:
    """Validates that the model is deterministic in eval mode."""

    def test_same_input_same_output(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            r1 = ac(obs, rnn_states, sample_actions=False)
            r2 = ac(obs, rnn_states, sample_actions=False)
        assert torch.allclose(r1["values"], r2["values"])
        assert torch.allclose(r1["action_logits"], r2["action_logits"])

    def test_centralized_critic_deterministic(self):
        """Centralized critic with same input should give same values."""
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))

        with torch.no_grad():
            v1 = ac(obs, rnn_states, values_only=True)["values"]
            v2 = ac(obs, rnn_states, values_only=True)["values"]
        assert torch.allclose(v1, v2)


class TestMAPPOEdgeCases:
    """Tests edge cases and boundary conditions."""

    def test_single_agent_mappo_falls_back(self):
        """With num_agents=1, MAPPO should still work (though it's equivalent to IPPO)."""
        ac = _make_mappo(num_agents=1)
        ac.eval()
        batch = 3
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)

    def test_large_agent_count(self):
        """MAPPO should handle larger agent counts."""
        ac = _make_mappo(num_agents=8)
        ac.eval()
        batch = 16
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)
        # Check centralized critic input dimension is correct
        first_layer = ac.centralized_critic[0]
        assert first_layer.in_features == ac.decoder.get_out_size() * 8

    def test_batch_size_equals_num_agents(self):
        """Minimum valid batch: exactly num_agents samples."""
        ac = _make_mappo(num_agents=3)
        ac.eval()
        batch = 3
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)


class TestForwardTail:
    """Tests forward_tail specifically, bypassing encoder/core."""

    def test_mappo_forward_tail_centralized_values(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        core_out_size = ac.core.get_out_size()
        # Simulate core output for 2 agents (batch=1 team)
        core_output = torch.randn(2, core_out_size)
        with torch.no_grad():
            result = ac.forward_tail(core_output, values_only=False, sample_actions=False)
        assert result["values"].shape == (2,)
        assert result["action_logits"].shape == (2, 4)

    def test_ippo_forward_tail_decentralized_values(self):
        ac = _make_mappo(algo="APPO", num_agents=2)
        ac.eval()
        core_out_size = ac.core.get_out_size()
        core_output = torch.randn(4, core_out_size)
        with torch.no_grad():
            result = ac.forward_tail(core_output, values_only=False, sample_actions=False)
        assert result["values"].shape == (4,)

    def test_values_only_skips_action_logits(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        core_output = torch.randn(4, ac.core.get_out_size())
        with torch.no_grad():
            result = ac.forward_tail(core_output, values_only=True, sample_actions=False)
        assert "values" in result
        assert "action_logits" not in result


class TestCentralizedCriticMLP:
    """Validates the architecture of the centralized critic MLP."""

    def test_mlp_layer_count(self):
        """Centralized critic should have 5 layers (3 Linear + 2 Tanh)."""
        ac = _make_mappo(num_agents=2)
        layers = list(ac.centralized_critic.children())
        assert len(layers) == 5
        assert isinstance(layers[0], nn.Linear)
        assert isinstance(layers[1], nn.Tanh)
        assert isinstance(layers[2], nn.Linear)
        assert isinstance(layers[3], nn.Tanh)
        assert isinstance(layers[4], nn.Linear)

    def test_mlp_hidden_sizes(self):
        ac = _make_mappo(num_agents=2)
        layers = [l for l in ac.centralized_critic.children() if isinstance(l, nn.Linear)]
        assert layers[0].out_features == 512
        assert layers[1].out_features == 256
        assert layers[2].out_features == 2  # num_agents

    def test_mlp_adapts_to_agent_count(self):
        """Output dimension should match num_agents."""
        for n in [2, 3, 5]:
            ac = _make_mappo(num_agents=n)
            last_linear = [l for l in ac.centralized_critic.children() if isinstance(l, nn.Linear)][-1]
            assert last_linear.out_features == n


class TestAlgoFlag:
    """Validates that the algo flag correctly gates centralized vs decentralized."""

    def test_mappo_enables_centralized(self):
        ac = _make_mappo()
        assert ac.use_centralized_critic is True

    def test_appo_disables_centralized(self):
        ac = _make_mappo(algo="APPO")
        assert ac.use_centralized_critic is False

    def test_algo_default_is_appo(self):
        """When algo is not in cfg, should default to APPO (decentralized)."""
        cfg = _make_cfg(algo="APPO")
        del cfg["algo"]
        obs_space, action_space = _make_spaces()
        ac = make_mappo_actor_critic(cfg, obs_space, action_space)
        assert ac.use_centralized_critic is False
