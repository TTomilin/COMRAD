import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from comrad.models.qplex_mixer import (
    DMAQ_SI_Weight,
    DMAQer,
    Qatten_Weight,
    DMAQ_QattenMixer,
)
from comrad.models.qmix_model import make_mixer, VDNMixer, QMixMixer


def _default_cfg(**overrides):
    defaults = dict(
        qplex_embed_dim=32,
        qplex_hypernet_embed=64,
        qplex_adv_hypernet_layers=1,
        qplex_adv_hypernet_embed=64,
        qplex_num_kernel=4,
        qplex_n_head=4,
        qplex_attend_reg_coef=0.001,
        qplex_is_minus_one=True,
        qplex_weighted_head=False,
        qplex_nonlinear=False,
        qplex_state_bias=True,
        qmix_embed_dim=32,
        qmix_hypernet_hidden=64,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# DMAQ_SI_Weight
class TestDMAQ_SI_Weight:
    def test_output_shape(self):
        n_agents, state_dim, n_actions = 3, 64, 18
        si = DMAQ_SI_Weight(n_agents, state_dim, n_actions, num_kernel=4, adv_hypernet_layers=1)
        B = 8
        states = torch.randn(B, state_dim)
        actions = torch.randn(B, n_agents * n_actions)
        out = si(states, actions)
        assert out.shape == (B, n_agents)

    def test_output_non_negative(self):
        """SI weights should be non-negative (product of abs and sigmoids)."""
        n_agents, state_dim, n_actions = 2, 32, 10
        si = DMAQ_SI_Weight(n_agents, state_dim, n_actions, num_kernel=4, adv_hypernet_layers=2)
        states = torch.randn(16, state_dim)
        actions = torch.randn(16, n_agents * n_actions)
        out = si(states, actions)
        assert (out >= 0).all()

    @pytest.mark.parametrize("layers", [1, 2, 3])
    def test_hypernet_layers(self, layers):
        n_agents, state_dim, n_actions = 2, 32, 10
        si = DMAQ_SI_Weight(n_agents, state_dim, n_actions, num_kernel=2, adv_hypernet_layers=layers)
        states = torch.randn(4, state_dim)
        actions = torch.randn(4, n_agents * n_actions)
        out = si(states, actions)
        assert out.shape == (4, n_agents)


# DMAQer
class TestDMAQer:
    def _make(self, n_agents=3, state_dim=64, n_actions=18, **cfg_kw):
        cfg = _default_cfg(qplex_weighted_head=True, qplex_adv_hypernet_layers=3, **cfg_kw)
        return DMAQer(cfg, n_agents, state_dim, n_actions)

    def test_v_tot_shape(self):
        mixer = self._make()
        B = 8
        agent_qs = torch.randn(B, 3)
        states = torch.randn(B, 64)
        v_tot, regs = mixer(agent_qs, states, is_v=True)
        assert v_tot.shape == (B, 1, 1)
        assert regs == []

    def test_a_tot_shape(self):
        mixer = self._make()
        B = 8
        agent_qs = torch.randn(B, 3)
        states = torch.randn(B, 64)
        actions = torch.randn(B, 3 * 18)
        max_q_i = torch.randn(B, 3)
        a_tot, regs = mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)
        assert a_tot.shape == (B, 1, 1)
        assert regs == []

    def test_gradient_flow_through_si_weights(self):
        """Gradients should flow through SI weights, not through advantage values (detached)."""
        mixer = self._make(n_agents=2, state_dim=32, n_actions=10)
        agent_qs = torch.randn(4, 2, requires_grad=True)
        states = torch.randn(4, 32)
        actions = torch.randn(4, 2 * 10)
        max_q_i = torch.randn(4, 2)

        # V_tot path: gradient should flow through agent_qs (weighted sum)
        v_tot, _ = mixer(agent_qs, states, is_v=True)
        v_loss = v_tot.squeeze().sum()
        v_loss.backward(retain_graph=True)
        assert agent_qs.grad is not None

        agent_qs.grad = None
        # A_tot path: adv_q = (agent_qs - max_q_i).detach(), so no grad through agent_qs
        # This is correct per the reference: gradients only flow through SI weights
        a_tot, _ = mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)
        a_loss = a_tot.squeeze().sum()
        a_loss.backward()
        # agent_qs should NOT have gradient from A_tot (advantage detached)
        assert agent_qs.grad is None


# Qatten_Weight
class TestQattenWeight:
    def test_output_shapes(self):
        n_agents, state_dim, n_actions, unit_dim = 3, 96, 18, 32
        qw = Qatten_Weight(n_agents, state_dim, n_actions, unit_dim, n_head=4)
        B = 8
        agent_qs = torch.randn(B, n_agents)
        states = torch.randn(B, state_dim)
        head_attend, v, attend_mag_regs, head_entropies = qw(agent_qs, states)
        assert head_attend.shape == (B, n_agents)
        assert v.shape == (B, 1)
        assert isinstance(attend_mag_regs, (float, torch.Tensor))
        assert len(head_entropies) == 4

    def test_attention_weights_sum_to_one_per_head(self):
        """Softmax attention weights should sum to ~1 per head before aggregation."""
        n_agents, state_dim, n_actions, unit_dim = 2, 64, 10, 32
        qw = Qatten_Weight(n_agents, state_dim, n_actions, unit_dim, n_head=2, weighted_head=False)
        B = 4
        agent_qs = torch.randn(B, n_agents)
        states = torch.randn(B, state_dim)
        head_attend, v, _, _ = qw(agent_qs, states)
        # Without weighted_head, the head_attend weights should be close to n_head (each head's softmax sums to 1)
        # sum across agents should be approximately n_head
        sums = head_attend.sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, 2.0), atol=0.01)


# DMAQ_QattenMixer
class TestDMAQ_QattenMixer:
    def _make(self, n_agents=3, state_dim=96, n_actions=18, unit_dim=32, **cfg_kw):
        cfg = _default_cfg(**cfg_kw)
        return DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

    def test_v_tot_shape(self):
        mixer = self._make()
        B = 8
        v_tot, regs = mixer(torch.randn(B, 3), torch.randn(B, 96), is_v=True)
        assert v_tot.shape == (B, 1, 1)
        assert len(regs) == 1  # attend_mag_regs

    def test_a_tot_shape(self):
        mixer = self._make()
        B = 8
        a_tot, regs = mixer(
            torch.randn(B, 3), torch.randn(B, 96),
            actions=torch.randn(B, 3 * 18), max_q_i=torch.randn(B, 3), is_v=False
        )
        assert a_tot.shape == (B, 1, 1)
        assert len(regs) == 1

    def test_head_entropies_stored(self):
        mixer = self._make()
        B = 4
        mixer(torch.randn(B, 3), torch.randn(B, 96), is_v=True)
        assert hasattr(mixer, '_last_head_entropies')
        assert len(mixer._last_head_entropies) == 4  # n_head=4

    def test_regularization_gradient_flows(self):
        """Attention regularization should be differentiable."""
        mixer = self._make(n_agents=2, state_dim=64, n_actions=10, unit_dim=32)
        agent_qs = torch.randn(4, 2)
        states = torch.randn(4, 64, requires_grad=True)
        v_tot, regs = mixer(agent_qs, states, is_v=True)
        loss = v_tot.squeeze().sum() + sum(regs)
        loss.backward()
        assert states.grad is not None


class TestMakeMixer:
    def test_vdn(self):
        cfg = _default_cfg(mixer='vdn')
        m = make_mixer(cfg, num_agents=3, state_dim=96)
        assert isinstance(m, VDNMixer)

    def test_qmix(self):
        cfg = _default_cfg(mixer='qmix')
        m = make_mixer(cfg, num_agents=3, state_dim=96)
        assert isinstance(m, QMixMixer)

    def test_dmaq(self):
        cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=3)
        m = make_mixer(cfg, num_agents=3, state_dim=96, action_dim=18)
        assert isinstance(m, DMAQer)

    def test_dmaq_qatten(self):
        cfg = _default_cfg(mixer='dmaq_qatten')
        m = make_mixer(cfg, num_agents=3, state_dim=96, action_dim=18, unit_dim=32)
        assert isinstance(m, DMAQ_QattenMixer)

    def test_invalid_mixer(self):
        cfg = _default_cfg(mixer='invalid')
        with pytest.raises(ValueError, match="Wrong mixer type"):
            make_mixer(cfg, num_agents=3, state_dim=96)


class TestDuplexDueling:
    """Test the full V_tot + A_tot decomposition produces valid Q_tot."""

    @pytest.mark.parametrize("mixer_type,MixerClass", [
        ("dmaq", DMAQer),
        ("dmaq_qatten", DMAQ_QattenMixer),
    ])
    def test_q_tot_decomposition(self, mixer_type, MixerClass):
        n_agents, state_dim, n_actions, unit_dim = 2, 64, 10, 32
        if mixer_type == "dmaq":
            cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=3)
            mixer = DMAQer(cfg, n_agents, state_dim, n_actions)
        else:
            cfg = _default_cfg(mixer='dmaq_qatten')
            mixer = DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

        B = 8
        agent_qs = torch.randn(B, n_agents)
        states = torch.randn(B, state_dim)
        max_q_i = torch.randn(B, n_agents)
        onehot_actions = torch.randn(B, n_agents * n_actions)

        v_tot, v_regs = mixer(max_q_i, states, is_v=True)
        a_tot, a_regs = mixer(agent_qs, states, actions=onehot_actions, max_q_i=max_q_i, is_v=False)

        q_tot = v_tot.squeeze() + a_tot.squeeze()
        assert q_tot.shape == (B,)
        assert torch.isfinite(q_tot).all()

    def test_advantage_zero_at_greedy(self):
        """When agent_qs == max_q_i, advantage should be zero."""
        n_agents, state_dim, n_actions = 2, 64, 10
        cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=3)
        mixer = DMAQer(cfg, n_agents, state_dim, n_actions)

        B = 4
        agent_qs = torch.randn(B, n_agents)
        states = torch.randn(B, state_dim)
        # When agent takes greedy action: agent_qs == max_q_i
        max_q_i = agent_qs.clone()
        onehot_actions = torch.randn(B, n_agents * n_actions)

        a_tot, _ = mixer(agent_qs, states, actions=onehot_actions, max_q_i=max_q_i, is_v=False)
        # adv_q = (agent_qs - max_q_i).detach() = 0, so A_tot = 0
        assert torch.allclose(a_tot.squeeze(), torch.zeros(B), atol=1e-6)


class TestReturnTypes:
    """All mixers must return types compatible with the learner."""

    def test_vdn_returns_scalar(self):
        mixer = VDNMixer(3)
        out = mixer(torch.randn(8, 3))
        assert out.shape == (8,)

    def test_qmix_returns_scalar(self):
        mixer = QMixMixer(3, 64)
        out = mixer(torch.randn(8, 3), torch.randn(8, 64))
        assert out.shape == (8,)

    def test_dmaq_returns_tuple(self):
        cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=3)
        mixer = DMAQer(cfg, 3, 64, 18)
        out, regs = mixer(torch.randn(8, 3), torch.randn(8, 64), is_v=True)
        assert out.shape == (8, 1, 1)
        assert isinstance(regs, list)

    def test_qatten_returns_tuple(self):
        cfg = _default_cfg(mixer='dmaq_qatten')
        mixer = DMAQ_QattenMixer(cfg, 3, 96, 18, 32)
        out, regs = mixer(torch.randn(8, 3), torch.randn(8, 96), is_v=True)
        assert out.shape == (8, 1, 1)
        assert isinstance(regs, list)
        assert len(regs) == 1
