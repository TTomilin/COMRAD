import gymnasium as gym
import pytest
import torch

from sample_factory.model.model_utils import get_rnn_size
from comrad.models.mappo_model import MAPPOActorCritic, make_mappo_actor_critic

from .conftest import _make_cfg, _make_mappo, _make_spaces


class TestMAPPOConstruction:

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
        ac = _make_mappo(num_agents=3)
        decoder_out = ac.decoder.get_out_size()
        first_layer = ac.centralized_critic[0]
        assert first_layer.in_features == decoder_out * 3

    def test_mappo_centralized_critic_output_dim(self):
        ac = _make_mappo(num_agents=3)
        last_layer = ac.centralized_critic[-1]
        assert last_layer.out_features == 3

    def test_mappo_n_agents_stored(self):
        ac = _make_mappo(num_agents=4)
        assert ac.n_agents == 4


class TestMAPPOForwardShapes:

    def test_forward_output_keys(self):
        ac = _make_mappo(num_agents=2)
        ac.eval()
        batch = 4
        obs = {"obs": torch.randn(batch, 3, 64, 64)}
        rnn_states = torch.zeros(batch, get_rnn_size(ac.cfg))
        with torch.no_grad():
            result = ac(obs, rnn_states)
        assert "values" in result
        assert "action_logits" in result
        assert "actions" in result
        assert "new_rnn_states" in result

    def test_values_shape_matches_batch(self):
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


class TestMAPPOFactory:

    def test_factory_returns_correct_type(self):
        ac = _make_mappo()
        assert isinstance(ac, MAPPOActorCritic)

    def test_factory_ippo_returns_correct_type(self):
        ac = _make_mappo(algo="APPO")
        assert isinstance(ac, MAPPOActorCritic)
        assert ac.use_centralized_critic is False

    def test_factory_handles_tuple_action_space(self):
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
        assert result["action_logits"].shape == (batch, 5)


class TestAlgoFlag:

    def test_mappo_enables_centralized(self):
        ac = _make_mappo()
        assert ac.use_centralized_critic is True

    def test_appo_disables_centralized(self):
        ac = _make_mappo(algo="APPO")
        assert ac.use_centralized_critic is False

    def test_algo_default_is_appo(self):
        cfg = _make_cfg(algo="APPO")
        del cfg["algo"]
        obs_space, action_space = _make_spaces()
        ac = make_mappo_actor_critic(cfg, obs_space, action_space)
        assert ac.use_centralized_critic is False
