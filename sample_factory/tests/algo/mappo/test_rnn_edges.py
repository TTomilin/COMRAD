import torch

from sample_factory.model.model_utils import get_rnn_size

from .conftest import _make_mappo


class TestMAPPORNN:

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
        ac = _make_mappo(num_agents=2, use_rnn=True, rnn_num_layers=2)
        ac.eval()
        batch = 4
        rnn_size = get_rnn_size(ac.cfg)
        assert rnn_size == 64 * 2
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, rnn_size)
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["new_rnn_states"].shape == (batch, rnn_size)

    def test_ippo_rnn_forward(self):
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

    def test_mask_does_not_change_values(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        mask = torch.ones(batch, 4)
        mask[:, 0] = 0.0

        with torch.no_grad():
            result_no_mask = ac(obs, rnn_states, action_mask=None)
            result_mask = ac(obs, rnn_states, action_mask=mask)

        assert torch.allclose(result_no_mask["values"], result_mask["values"], atol=1e-6), (
            "Action mask should not affect value computation"
        )

    def test_mask_blocks_actions(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 2
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        mask = torch.ones(batch, 4)
        mask[:, 2] = 0.0

        with torch.no_grad():
            _ = ac(obs, rnn_states, action_mask=mask)

        dist = ac.action_distribution()
        probs = dist.probs if hasattr(dist, "probs") else dist.distribution.probs
        assert (probs[:, 2] < 1e-6).all(), (
            "Masked action should have ~zero probability in the distribution"
        )

    def test_all_ones_mask_no_effect(self):
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


class TestMAPPODeterminism:

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

    def test_single_agent_mappo_falls_back(self):
        ac = _make_mappo(num_agents=1)
        ac.eval()
        batch = 3
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)

    def test_large_agent_count(self):
        ac = _make_mappo(num_agents=8)
        ac.eval()
        batch = 16
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)
        first_layer = ac.centralized_critic[0]
        assert first_layer.in_features == ac.decoder.get_out_size() * 8

    def test_batch_size_equals_num_agents(self):
        ac = _make_mappo(num_agents=3)
        ac.eval()
        batch = 3
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert result["values"].shape == (batch,)
