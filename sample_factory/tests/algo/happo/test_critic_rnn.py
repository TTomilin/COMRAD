import numpy as np
import pytest
import torch
import torch.nn as nn

from sample_factory.algo.utils.context import sf_global_context
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.happo_model import make_happo_actor_critic
from sample_factory.algo.learning.learner_happo import HAPPOLearner

from .conftest import (
    _make_happo_cfg,
    _make_obs_space,
    _make_action_space,
    _make_happo_model,
    _make_obs_batch,
)


class TestCriticRNNConstruction:

    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_critic_cores_created_when_enabled(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        assert model.critic_cores is not None
        assert len(model.critic_cores) == 2

    def test_critic_cores_none_by_default(self):
        model = _make_happo_model(num_agents=2, use_rnn=True)
        assert model.critic_cores is None

    def test_critic_cores_none_without_rnn(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=False, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        assert model.critic_cores is None

    def test_critic_mlp_input_size_with_rnn(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_size=64, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        linears = [m for m in model.centralized_critic if isinstance(m, nn.Linear)]
        assert linears[0].in_features == 64 * 2  # rnn_size * n_agents

    def test_critic_rnn_runs_at_inference(self):
        num_agents = 2
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, rnn_size=64, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()
        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(cfg))
        with torch.no_grad():
            result = model(obs, rnn_states)
        assert result['values'].shape == (batch,)
        assert result['new_rnn_states'].shape == rnn_states.shape

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_critic_cores_rnn_type(self, rnn_type):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_type=rnn_type, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        assert model.critic_cores is not None
        # The core should have rnn_size output
        assert model.critic_cores[0].get_out_size() == cfg.rnn_size


class TestCriticRNNForward:

    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_forward_shapes_with_critic_rnn(self):
        num_agents = 2
        batch = num_agents * 2
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result["values"].shape == (batch,)
        assert result["action_logits"].shape[0] == batch
        assert result["new_rnn_states"].shape == rnn_states.shape

    def test_values_only_with_critic_rnn(self):
        num_agents = 2
        batch = num_agents * 2
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(cfg))

        with torch.no_grad():
            result = model(obs, rnn_states, values_only=True)

        assert result["values"].shape == (batch,)
        assert "action_logits" not in result

    def test_same_transition_same_value_with_critic_rnn(self):
        num_agents = 2
        batch = num_agents * 2
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)
        model.eval()

        obs = _make_obs_batch(batch, num_agents)
        rnn_states = torch.zeros(batch, get_rnn_size(cfg))

        with torch.no_grad():
            result = model(obs, rnn_states)

        values = result["values"]
        torch.testing.assert_close(values[0], values[1])
        torch.testing.assert_close(values[2], values[3])


class TestGetCriticParams:

    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def test_includes_cores_when_present(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)

        stub = object.__new__(HAPPOLearner)
        stub.actor_critic = model

        params = stub._get_critic_params()
        param_ids = set(id(p) for p in params)

        # Must include critic_cores params
        for core in model.critic_cores:
            for p in core.parameters():
                assert id(p) in param_ids, "critic_cores params missing from _get_critic_params"

    def test_excludes_cores_when_absent(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, happo_critic_rnn=False)
        obs_space = _make_obs_space(num_agents=2)
        action_space = _make_action_space()
        model = make_happo_actor_critic(cfg, obs_space, action_space)

        stub = object.__new__(HAPPOLearner)
        stub.actor_critic = model

        params = stub._get_critic_params()
        param_ids = set(id(p) for p in params)

        # Should include encoders and MLP
        for p in model.centralized_critic.parameters():
            assert id(p) in param_ids
        for p in model.critic_encoders.parameters():
            assert id(p) in param_ids


class TestGetRnnSizeDoubling:

    def test_no_change_without_critic_rnn(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_size=64, rnn_num_layers=1,
                              rnn_type="gru", happo_critic_rnn=False)
        assert get_rnn_size(cfg) == 64

    def test_doubles_with_critic_rnn_gru(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_size=64, rnn_num_layers=1,
                              rnn_type="gru", happo_critic_rnn=True)
        assert get_rnn_size(cfg) == 128

    def test_doubles_with_critic_rnn_lstm(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_size=64, rnn_num_layers=1,
                              rnn_type="lstm", happo_critic_rnn=True)
        assert get_rnn_size(cfg) == 256

    def test_doubles_with_multi_layer_rnn(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_size=64, rnn_num_layers=2,
                              rnn_type="gru", happo_critic_rnn=True)
        assert get_rnn_size(cfg) == 256

    def test_no_doubling_non_happo_algo(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=True, rnn_size=64, happo_critic_rnn=True)
        cfg.algo = "APPO"  # Override to non-HAPPO
        assert get_rnn_size(cfg) == 64

    def test_no_doubling_without_rnn(self):
        cfg = _make_happo_cfg(num_agents=2, use_rnn=False, happo_critic_rnn=True)
        assert get_rnn_size(cfg) == 1  # No RNN at all


class TestCriticRNNRollout:

    @pytest.fixture(autouse=True)
    def setup_context(self):
        sf_global_context()
        yield

    def _make_model(self, num_agents=2, rnn_type="gru", rnn_size=64):
        cfg = _make_happo_cfg(num_agents=num_agents, use_rnn=True, rnn_type=rnn_type,
                              rnn_size=rnn_size, happo_critic_rnn=True)
        obs_space = _make_obs_space(num_agents=num_agents)
        action_space = _make_action_space()
        return make_happo_actor_critic(cfg, obs_space, action_space), cfg

    def test_use_critic_rnn_attribute(self):
        model, _ = self._make_model()
        assert model.use_critic_rnn is True

    def test_critic_rnn_state_size_gru(self):
        model, cfg = self._make_model(rnn_type="gru", rnn_size=64)
        assert model.critic_rnn_state_size == 64

    def test_critic_rnn_state_size_lstm(self):
        model, cfg = self._make_model(rnn_type="lstm", rnn_size=64)
        assert model.critic_rnn_state_size == 128

    def test_forward_core_receives_actor_only(self):
        model, cfg = self._make_model()
        num_agents = cfg.num_agents
        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_size = get_rnn_size(cfg)
        R = model.critic_rnn_state_size

        # Create non-zero rnn_states to verify correct splitting
        rnn_states = torch.randn(batch, rnn_size)
        actor_rnn = rnn_states[:, :R]

        head_out = model.forward_head(obs, agent_idx=obs["agent_id"].argmax(dim=-1))
        core_out, new_actor_rnn = model.forward_core(head_out, actor_rnn,
                                                      agent_idx=obs["agent_id"].argmax(dim=-1))
        # Actor core output should have core out size, not 2*core
        assert core_out.shape == (batch, model.agent_cores[0].get_out_size())
        # new_actor_rnn should be actor-sized (R, not 2R)
        assert new_actor_rnn.shape == (batch, R)

    def test_forward_roundtrip_shapes(self):
        model, cfg = self._make_model()
        num_agents = cfg.num_agents
        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_size = get_rnn_size(cfg)
        rnn_states = torch.zeros(batch, rnn_size)

        model.eval()
        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result['new_rnn_states'].shape == (batch, rnn_size)

    def test_critic_rnn_states_updated_after_forward(self):
        model, cfg = self._make_model()
        num_agents = cfg.num_agents
        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_size = get_rnn_size(cfg)
        R = model.critic_rnn_state_size
        rnn_states = torch.zeros(batch, rnn_size)

        model.eval()
        with torch.no_grad():
            result = model(obs, rnn_states)

        critic_half = result['new_rnn_states'][:, R:]
        assert not torch.allclose(critic_half, torch.zeros_like(critic_half)), \
            "Critic RNN states should be non-zero after forward (critic cores should have processed them)"

    def test_actor_rnn_states_updated_after_forward(self):
        model, cfg = self._make_model()
        num_agents = cfg.num_agents
        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_size = get_rnn_size(cfg)
        R = model.critic_rnn_state_size
        rnn_states = torch.zeros(batch, rnn_size)

        model.eval()
        with torch.no_grad():
            result = model(obs, rnn_states)

        actor_half = result['new_rnn_states'][:, :R]
        assert not torch.allclose(actor_half, torch.zeros_like(actor_half)), \
            "Actor RNN states should be non-zero after forward"

    def test_forward_tail_raises_without_critic_states(self):
        model, cfg = self._make_model()
        num_agents = cfg.num_agents
        batch = num_agents * 2
        core_output = torch.randn(batch, model.agent_cores[0].get_out_size())
        obs = _make_obs_batch(batch, num_agents)
        agent_idx = obs["agent_id"].argmax(dim=-1)
        env_group_idx = torch.arange(batch) // num_agents

        with pytest.raises(ValueError, match="use_critic_rnn=True but critic_rnn_states is None"):
            model.forward_tail(
                core_output,
                agent_idx=agent_idx,
                env_group_idx=env_group_idx,
                normalized_obs_dict=obs,
                values_only=True,
                critic_rnn_states=None,
            )

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_roundtrip_both_rnn_types(self, rnn_type):
        model, cfg = self._make_model(rnn_type=rnn_type)
        num_agents = cfg.num_agents
        batch = num_agents * 2
        obs = _make_obs_batch(batch, num_agents)
        rnn_size = get_rnn_size(cfg)
        rnn_states = torch.zeros(batch, rnn_size)

        model.eval()
        with torch.no_grad():
            result = model(obs, rnn_states)

        assert result['new_rnn_states'].shape == (batch, rnn_size)
        assert not torch.isnan(result['values']).any()
