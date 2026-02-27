import torch
from torch import nn

from sample_factory.model.model_utils import get_rnn_size

from .conftest import _make_mappo


class TestCentralizedCritic:

    def test_mappo_values_depend_on_all_agents(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        obs_a = {"obs": torch.randn(2, 3, 64, 64)}
        obs_b = {"obs": obs_a["obs"].clone()}
        obs_b["obs"][0] += 10.0

        rnn_states = torch.zeros(2, get_rnn_size(ac.cfg))
        with torch.no_grad():
            val_a = ac(obs_a, rnn_states)["values"]
            val_b = ac(obs_b, rnn_states)["values"]

        assert not torch.allclose(val_b[1:2], val_a[1:2], atol=1e-5), (
            "MAPPO centralized critic: agent 1's value should change when agent 0's obs changes"
        )

    def test_ippo_values_independent_per_agent(self):
        ac = _make_mappo(algo="APPO", num_agents=2)
        ac.eval()
        obs_a = {"obs": torch.randn(2, 3, 64, 64)}
        obs_b = {"obs": obs_a["obs"].clone()}
        obs_b["obs"][0] += 10.0

        rnn_states = torch.zeros(2, get_rnn_size(ac.cfg))
        with torch.no_grad():
            val_a = ac(obs_a, rnn_states)["values"]
            val_b = ac(obs_b, rnn_states)["values"]

        assert torch.allclose(val_b[1:2], val_a[1:2], atol=1e-6), (
            "IPPO decentralized critic: agent 1's value should be unchanged when agent 0's obs changes"
        )

    def test_mappo_actions_still_decentralized(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        obs_a = {"obs": torch.randn(2, 3, 64, 64)}
        obs_b = {"obs": obs_a["obs"].clone()}
        obs_b["obs"][0] += 10.0

        rnn_states = torch.zeros(2, get_rnn_size(ac.cfg))
        with torch.no_grad():
            logits_a = ac(obs_a, rnn_states)["action_logits"]
            logits_b = ac(obs_b, rnn_states)["action_logits"]

        assert torch.allclose(logits_b[1:2], logits_a[1:2], atol=1e-6), (
            "MAPPO actions should be decentralized: agent 1's logits should not change"
        )

    def test_values_only_mode(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states, values_only=True)
        assert "values" in result
        assert "action_logits" not in result


class TestMAPPOGradients:

    def test_centralized_critic_receives_gradients(self):
        ac = _make_mappo(num_agents=2)
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        result = ac(obs, rnn_states)

        target_values = torch.zeros(batch)
        value_loss = (result["values"] - target_values).pow(2).mean()
        value_loss.backward()

        for name, param in ac.centralized_critic.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_encoder_receives_gradients_from_value_loss(self):
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
        ac = _make_mappo(num_agents=2)
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        result = ac(obs, rnn_states, sample_actions=False)

        policy_loss = result["action_logits"].sum()
        policy_loss.backward()

        encoder_has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in ac.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should receive gradients from policy loss"


class TestForwardTail:

    def test_mappo_forward_tail_centralized_values(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        core_out_size = ac.core.get_out_size()
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

    def test_mlp_layer_count(self):
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
        assert layers[2].out_features == 2

    def test_mlp_adapts_to_agent_count(self):
        for n in [2, 3, 5]:
            ac = _make_mappo(num_agents=n)
            last_linear = [l for l in ac.centralized_critic.children() if isinstance(l, nn.Linear)][-1]
            assert last_linear.out_features == n
