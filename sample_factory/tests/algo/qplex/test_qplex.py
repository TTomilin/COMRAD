import pytest
import torch
from types import SimpleNamespace

from comrad.models.qplex_mixer import (
    DMAQ_SI_Weight,
    DMAQer,
    Qatten_Weight,
    DMAQ_QattenMixer,
)
from comrad.models.qmix_model import make_mixer, VDNMixer, QMixMixer
from sample_factory.algo.learning.learner_qmix import QMixLearner


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
        for param in mixer.parameters():
            param.grad = None

        # A_tot path: adv_q = (agent_qs - max_q_i).detach(), so no grad through agent_qs
        # This is correct per the reference: gradients only flow through SI weights
        a_tot, _ = mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)
        a_loss = a_tot.squeeze().sum()
        a_loss.backward()
        # agent_qs should NOT have gradient from A_tot (advantage detached)
        assert agent_qs.grad is None

        weighted_head_has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in mixer.named_parameters()
            if name.startswith("hyper_w_final") or name.startswith("V")
        )
        si_has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in mixer.named_parameters()
            if name.startswith("si_weight")
        )
        assert not weighted_head_has_grad
        assert si_has_grad


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

    def test_v_tot_shape_batch_size_one(self):
        mixer = self._make(n_agents=2, state_dim=64, n_actions=10, unit_dim=32)
        v_tot, regs = mixer(torch.randn(1, 2), torch.randn(1, 64), is_v=True)
        assert v_tot.shape == (1, 1, 1)
        assert len(regs) == 1

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
        assert all(not entropy.requires_grad for entropy in mixer._last_head_entropies)
        assert all(entropy.device.type == "cpu" for entropy in mixer._last_head_entropies)

    def test_regularization_gradient_flows(self):
        """Attention regularization should be differentiable."""
        mixer = self._make(n_agents=2, state_dim=64, n_actions=10, unit_dim=32)
        agent_qs = torch.randn(4, 2)
        states = torch.randn(4, 64, requires_grad=True)
        v_tot, regs = mixer(agent_qs, states, is_v=True)
        loss = v_tot.squeeze().sum() + sum(regs)
        loss.backward()
        assert states.grad is not None

    def test_a_tot_grad_flows_only_through_si_weights(self):
        mixer = self._make(n_agents=2, state_dim=64, n_actions=10, unit_dim=32)
        agent_qs = torch.randn(4, 2, requires_grad=True)
        states = torch.randn(4, 64)
        actions = torch.randn(4, 2 * 10)
        max_q_i = torch.randn(4, 2)

        a_tot, _ = mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)
        a_tot.squeeze().sum().backward()

        assert agent_qs.grad is None
        attention_has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in mixer.named_parameters()
            if name.startswith("attention_weight")
        )
        si_has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in mixer.named_parameters()
            if name.startswith("si_weight")
        )
        assert not attention_has_grad
        assert si_has_grad


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


class TestQMixLearnerQPLEXHelpers:
    def test_compute_qplex_q_tot_returns_regs_without_persistent_graph_state(self):
        learner = object.__new__(QMixLearner)
        learner.agent_net = SimpleNamespace(action_sizes=[3, 2])

        cfg = _default_cfg(mixer='dmaq_qatten')
        mixer = DMAQ_QattenMixer(cfg, 2, 64, 5, 32)

        agent_qs = torch.randn(4, 2, requires_grad=True)
        state = torch.randn(4, 64)
        q_logits = torch.randn(4, 2, 5, requires_grad=True)
        actions = torch.tensor(
            [
                [[0, 0], [1, 1]],
                [[2, 1], [0, 0]],
                [[1, 0], [2, 1]],
                [[0, 1], [1, 0]],
            ],
            dtype=torch.long,
        )

        q_tot, regs = QMixLearner._compute_qplex_q_tot(learner, agent_qs, state, q_logits, actions, mixer)

        assert q_tot.shape == (4,)
        assert len(regs) == 1
        assert not hasattr(learner, '_last_qplex_regs')


class TestQPlexGradientFlow:
    """
    V_tot must receive agent_qs (live gradient), not max_q_i (detached)
    Passing max_q_i zeroes out Q-net gradient entirely because max_q_i is detached in _compute_qplex_q_tot. The reference (dmaq_qatten_learner.py) passes chosen_action_qvals
    """

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_v_tot_receives_agent_qs_with_gradient(self, mixer_type):
        """Q-net parameters must receive gradient from V_tot via agent_qs."""
        n_agents, n_actions, state_dim, unit_dim = 2, 5, 64, 32

        if mixer_type == "dmaq":
            cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=1)
            mixer = DMAQer(cfg, n_agents, state_dim, n_actions)
        else:
            cfg = _default_cfg(mixer='dmaq_qatten')
            mixer = DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

        learner = object.__new__(QMixLearner)
        learner.agent_net = SimpleNamespace(action_sizes=[n_actions])

        # agent_qs with gradient (simulating Q-net output)
        agent_qs = torch.randn(4, n_agents, requires_grad=True)
        state = torch.randn(4, state_dim)
        q_logits = torch.randn(4, n_agents, n_actions)
        actions = torch.randint(0, n_actions, (4, n_agents, 1))

        q_tot, regs = QMixLearner._compute_qplex_q_tot(
            learner, agent_qs, state, q_logits, actions, mixer
        )

        loss = q_tot.sum()
        loss.backward()

        # agent_qs MUST have gradient from V_tot (the weighted sum path)
        assert agent_qs.grad is not None, (
            "agent_qs has no gradient -- V_tot is not propagating through Q-values"
        )
        assert agent_qs.grad.abs().sum() > 0, (
            "agent_qs gradient is all zeros -- V_tot weighting has no effect on Q-net"
        )

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_passing_max_q_i_to_v_tot_would_zero_qnet_gradient(self, mixer_type):
        """if max_q_i (detached) were passed to V_tot instead of agent_qs, the Q-net would receive zero gradient"""
        n_agents, n_actions, state_dim, unit_dim = 2, 5, 64, 32

        if mixer_type == "dmaq":
            cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=1)
            mixer = DMAQer(cfg, n_agents, state_dim, n_actions)
        else:
            cfg = _default_cfg(mixer='dmaq_qatten')
            mixer = DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

        agent_qs = torch.randn(4, n_agents, requires_grad=True)
        state = torch.randn(4, state_dim)
        max_q_i = agent_qs.detach()  # THIS IS THE BUG: detached
        onehot_actions = torch.randn(4, n_agents * n_actions)

        # Simulate the broken path: pass detached max_q_i to V_tot
        v_tot, _ = mixer(max_q_i, state, is_v=True)
        a_tot, _ = mixer(agent_qs, state, actions=onehot_actions, max_q_i=max_q_i, is_v=False)
        q_tot = v_tot.squeeze() + a_tot.squeeze()

        q_tot.sum().backward()

        # agent_qs should have NO gradient because:
        # V_tot received detached max_q_i (no grad)
        # A_tot detaches (agent_qs - max_q_i) internally
        assert agent_qs.grad is None, (
            "Expected no gradient when max_q_i is passed to V_tot, but gradient flowed. Test logic error"
        )

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_mixer_params_receive_gradient_from_both_paths(self, mixer_type):
        """Both V_tot (attention/hyper weights) and A_tot (SI weights) mixer
        parameters should receive gradient through _compute_qplex_q_tot"""
        n_agents, n_actions, state_dim, unit_dim = 2, 5, 64, 32

        if mixer_type == "dmaq":
            cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=1)
            mixer = DMAQer(cfg, n_agents, state_dim, n_actions)
        else:
            cfg = _default_cfg(mixer='dmaq_qatten')
            mixer = DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

        learner = object.__new__(QMixLearner)
        learner.agent_net = SimpleNamespace(action_sizes=[n_actions])

        agent_qs = torch.randn(4, n_agents, requires_grad=True)
        state = torch.randn(4, state_dim)
        q_logits = torch.randn(4, n_agents, n_actions)
        actions = torch.randint(0, n_actions, (4, n_agents, 1))

        q_tot, regs = QMixLearner._compute_qplex_q_tot(
            learner, agent_qs, state, q_logits, actions, mixer
        )
        loss = q_tot.sum() + sum(regs)
        loss.backward()

        si_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in mixer.named_parameters()
            if "si_weight" in n
        )
        assert si_has_grad, "SI weight parameters have no gradient from A_tot"

        if mixer_type == "dmaq":
            vtot_param_prefix = "hyper_w_final"
        else:
            vtot_param_prefix = "attention_weight"
        vtot_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in mixer.named_parameters()
            if n.startswith(vtot_param_prefix)
        )
        assert vtot_has_grad, f"V_tot parameters ({vtot_param_prefix}.*) have no gradient"


class TestGradAccumEquivalence:
    """Gradient accumulation must produce equivalent gradients to full-batch."""

    @staticmethod
    def _make_qplex_learner(n_agents=2, n_actions=3, rnn_size=4, enc_dim=4):
        """Build a minimal learner with trainable agent net + QPLEX mixer."""
        from torch import nn
        import copy

        # state_dim for mixer = n_agents * enc_dim (encoder outputs are concatenated)
        state_dim = n_agents * enc_dim

        class _TrainableAgentStub(nn.Module):
            def __init__(self, rnn_size, n_actions, enc_dim):
                super().__init__()
                self._rnn_size = rnn_size
                self.num_actions = n_actions
                self.enc_dim = enc_dim
                self.encoder_out_size = enc_dim
                self.action_sizes = [n_actions]
                self.fc = nn.Linear(rnn_size, n_actions, bias=False)
                self.enc_fc = nn.Linear(rnn_size, enc_dim, bias=False)

            def get_rnn_size(self):
                return self._rnn_size

            def encode(self, obs):
                # stub: produce enc_dim features from obs by projecting rnn-sized zeros
                # We need deterministic output based on obs content
                if isinstance(obs, dict) and "obs" in obs:
                    val = obs["obs"]
                else:
                    val = obs if torch.is_tensor(obs) else list(obs.values())[0]
                batch_size = val.shape[0]
                # Use a simple deterministic transform matching the rnn_size -> enc_dim path
                dummy_rnn = torch.zeros(batch_size, self._rnn_size, device=val.device)
                return self.enc_fc(dummy_rnn)

            def forward_head(self, encoded, rnn_states):
                """Core -> decoder -> Q-head on pre-encoded features."""
                new_rnn = rnn_states + 0.1
                q_values = self.fc(new_rnn)
                return q_values, new_rnn

            def forward_decomposed(self, obs, rnn_states):
                encoded = self.encode(obs)
                q_values, new_rnn = self.forward_head(encoded, rnn_states)
                return q_values, new_rnn, encoded

            def get_q_for_actions(self, q, a):
                return q.gather(-1, a.unsqueeze(-1)).squeeze(-1) if a.dim() == 1 else q[:, 0]

            def flatten_rnn_parameters(self):
                pass

        learner = object.__new__(QMixLearner)
        learner.num_agents = n_agents
        learner._is_qplex = True
        learner.use_rnn = True
        learner.obs_normalizer = None
        learner.cfg = SimpleNamespace(
            gamma=0.99, double_dqn=True, q_value_clamp=100.0, use_huber_loss=True,
            qplex_grad_accum_mini_bs=2,  # force accumulation with mini_bs=2
            max_grad_norm=0,  # disable clipping for comparison
        )

        learner.agent_net = _TrainableAgentStub(rnn_size, n_actions, enc_dim)
        learner.target_agent_net = copy.deepcopy(learner.agent_net)
        for p in learner.target_agent_net.parameters():
            p.requires_grad = False

        cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=1)
        learner.mixer = DMAQer(cfg, n_agents, state_dim, n_actions)
        learner.target_mixer = copy.deepcopy(learner.mixer)
        for p in learner.target_mixer.parameters():
            p.requires_grad = False

        return learner

    @staticmethod
    def _make_batch(bs=4, rollout=3, n_agents=2, rnn_size=4):
        from sample_factory.algo.utils.tensor_dict import TensorDict
        torch.manual_seed(42)
        return TensorDict({
            'obs': TensorDict({'obs': torch.randn(bs, rollout + 1, n_agents, 1)}),
            'actions': torch.randint(0, 3, (bs, rollout, n_agents), dtype=torch.long),
            'rewards': torch.randn(bs, rollout, n_agents),
            'dones': torch.zeros(bs, rollout, n_agents),
            'time_outs': torch.zeros(bs, rollout, n_agents),
            'rnn_states': torch.randn(bs, n_agents, rnn_size),
        })

    def test_grad_accum_matches_full_batch(self):
        """Accumulated gradients over mini-batches must equal full-batch gradients."""
        import copy
        import math

        learner = self._make_qplex_learner()
        batch = self._make_batch(bs=4)  # will be split into 2 chunks of 2

        # Full batch single forward + backward
        learner_full = copy.deepcopy(learner)
        loss_full, td_full = QMixLearner._calculate_qmix_loss_sequential(learner_full, batch)
        loss_full.backward()
        full_grads = {n: p.grad.clone() for n, p in
                      list(learner_full.agent_net.named_parameters()) +
                      list(learner_full.mixer.named_parameters())
                      if p.grad is not None}

        # 2 chunks of 2
        learner_accum = copy.deepcopy(learner)
        total_bs = 4
        mini_bs = 2
        num_chunks = math.ceil(total_bs / mini_bs)

        # zero grads
        for p in list(learner_accum.agent_net.parameters()) + list(learner_accum.mixer.parameters()):
            if p.grad is not None:
                p.grad.zero_()

        for chunk_idx in range(num_chunks):
            start = chunk_idx * mini_bs
            end = min(start + mini_bs, total_bs)
            chunk_frac = (end - start) / total_bs

            chunk_batch = QMixLearner._slice_batch(batch, start, end)
            loss_chunk, _ = QMixLearner._calculate_qmix_loss_sequential(learner_accum, chunk_batch)
            (loss_chunk * chunk_frac).backward()

        accum_grads = {n: p.grad.clone() for n, p in
                       list(learner_accum.agent_net.named_parameters()) +
                       list(learner_accum.mixer.named_parameters())
                       if p.grad is not None}

        # Compare gradients
        assert set(full_grads.keys()) == set(accum_grads.keys()), \
            f"Mismatch in parameter names: {full_grads.keys()} vs {accum_grads.keys()}"
        for name in full_grads:
            torch.testing.assert_close(
                full_grads[name], accum_grads[name],
                atol=1e-5, rtol=1e-4,
                msg=f"Gradient mismatch for {name}",
            )


class TestQPlexDoubleDqnTargets:
    def test_double_dqn_target_matches_selected_action_qtot(self):
        """Under Double DQN the target mixer must evaluate the online-greedy action
        with the full V_tot + A_tot, not V_tot alone (which assumes target-greedy)."""
        torch.manual_seed(0)

        learner = object.__new__(QMixLearner)
        learner.agent_net = SimpleNamespace(action_sizes=[3])

        cfg = _default_cfg(mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=1)
        mixer = DMAQer(cfg, n_agents=2, state_dim=64, n_actions=3)

        state = torch.randn(1, 64)
        q_online_next = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])
        q_target_next = torch.tensor([[[1.0, 4.0, 0.0], [3.0, 2.0, 1.0]]])

        best_actions = q_online_next.argmax(dim=-1, keepdim=True)
        greedy_target_actions = q_target_next.argmax(dim=-1, keepdim=True)
        assert not torch.equal(best_actions, greedy_target_actions)

        target_agent_qs = q_target_next.gather(2, best_actions).squeeze(-1)

        v_only_target = QMixLearner._compute_qplex_target_v_tot(
            learner, target_agent_qs, state, mixer
        )
        full_target, _ = QMixLearner._compute_qplex_q_tot(
            learner, target_agent_qs, state, q_target_next, best_actions, mixer
        )

        # V_tot-only and full Q_tot MUST differ when online-greedy != target-greedy
        assert not torch.allclose(v_only_target, full_target, atol=1e-6), \
            "V_tot-only should differ from full Q_tot under Double DQN action disagreement"


class TestQPlexSoftmaxAttentionBottleneck:
    """
    With only 2 agents, the softmax in Qatten_Weight produces complementary
    weights (w1, w2) that always sum to 1 per head.  After summing n_head
    heads the total per-batch-element is always exactly n_head, severely
    constraining V_tot relative to QMIX's unconstrained hypernetwork.
    """

    def test_attention_weights_sum_to_n_head_for_2_agents(self):
        """Confirm that with 2 agents and 4 heads, w_1 + w_2 == 4 exactly."""
        n_agents, state_dim, n_actions, unit_dim = 2, 64, 14, 32
        n_head = 4
        qw = Qatten_Weight(
            n_agents, state_dim, n_actions, unit_dim, n_head=n_head, weighted_head=False,
        )
        B = 32
        agent_qs = torch.randn(B, n_agents)
        states = torch.randn(B, state_dim)
        head_attend, _, _, _ = qw(agent_qs, states)
        agent_weight_sums = head_attend.sum(dim=-1)  # [B]
        assert torch.allclose(
            agent_weight_sums,
            torch.full_like(agent_weight_sums, float(n_head)),
            atol=1e-5,
        ), (
            f"Expected weight sums == {n_head}, got {agent_weight_sums.tolist()}"
        )

    def test_vtot_gradient_nearly_uniform_for_2_agents(self):
        """
        V_tot gradient to Q_i (= attention weight w_i) must be nearly uniform
        when there are only 2 agents because softmax with 2 elements has only
        1 degree of freedom per head.  This demonstrates the representational
        bottleneck that limits QPLEX convergence relative to QMIX.
        """
        n_agents, state_dim, n_actions, unit_dim = 2, 64, 14, 32
        cfg = _default_cfg(mixer='dmaq_qatten', qplex_weighted_head=False)
        mixer = DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

        B = 64
        agent_qs = torch.randn(B, n_agents, requires_grad=True)
        states = torch.randn(B, state_dim)
        vtot, _ = mixer(agent_qs, states, is_v=True)
        vtot.squeeze().sum().backward()

        grad = agent_qs.grad  # [B, 2]
        per_agent_mean = grad.abs().mean(dim=0)  # [2]
        ratio = per_agent_mean.max() / per_agent_mean.min()
        # With 2 agents the gradient ratio is very close to 1
        assert ratio < 1.15, (
            f"Expected near-uniform gradient ratio for 2 agents, got {ratio:.4f}"
        )

    def test_vtot_gradient_std_lower_than_qmix(self):
        """
        QPLEX V_tot gradient to agent Q-values should have significantly lower
        variance than QMIX gradient, because QPLEX softmax constrains the
        weight sum while QMIX weights are unconstrained.
        """
        from comrad.models.qmix_model import QMixMixer
        n_agents, state_dim = 2, 64
        n_actions, unit_dim = 14, 32
        B = 128

        torch.manual_seed(123)

        # QMIX gradient
        qmix = QMixMixer(n_agents, state_dim)
        q_qmix = torch.randn(B, n_agents, requires_grad=True)
        s_qmix = torch.randn(B, state_dim)
        qmix(q_qmix, s_qmix).sum().backward()
        qmix_grad_std = q_qmix.grad.std().item()

        # QPLEX V_tot gradient
        cfg = _default_cfg(mixer='dmaq_qatten', qplex_weighted_head=False)
        mixer = DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)
        q_qplex = torch.randn(B, n_agents, requires_grad=True)
        s_qplex = torch.randn(B, state_dim)
        vtot, _ = mixer(q_qplex, s_qplex, is_v=True)
        vtot.squeeze().sum().backward()
        qplex_grad_std = q_qplex.grad.std().item()

        # QPLEX gradient std should be much lower (more uniform = less diverse)
        assert qplex_grad_std < qmix_grad_std, (
            f"Expected QPLEX grad std ({qplex_grad_std:.4f}) < "
            f"QMIX grad std ({qmix_grad_std:.4f})"
        )


class TestCompoundActionOnehotSemantic:
    """
    The compound action one-hot (_build_compound_onehot) produces a multi-hot
    vector (multiple 1s per agent), not a true one-hot.  The QPLEX reference
    code assumes single-action one-hot encoding.  These tests verify the shape
    is correct and document the semantic difference.
    """

    def test_compound_onehot_shape(self):
        """Shape must be [B, N * total_actions]."""
        learner = object.__new__(QMixLearner)
        learner.agent_net = SimpleNamespace(action_sizes=[3, 3, 3, 2, 3])
        n_agents, total_actions = 2, 14
        B = 8
        actions = torch.randint(0, 2, (B, n_agents, 5))
        # clamp per head
        for h, sz in enumerate([3, 3, 3, 2, 3]):
            actions[:, :, h] = actions[:, :, h] % sz
        onehot = QMixLearner._build_compound_onehot(learner, actions)
        assert onehot.shape == (B, n_agents * total_actions)

    def test_compound_onehot_is_multihot(self):
        """Each agent's segment should have exactly num_heads=5 non-zero entries."""
        learner = object.__new__(QMixLearner)
        action_sizes = [3, 3, 3, 2, 3]
        learner.agent_net = SimpleNamespace(action_sizes=action_sizes)
        n_agents = 2
        total_actions = sum(action_sizes)
        num_heads = len(action_sizes)
        B = 4
        actions = torch.zeros(B, n_agents, num_heads, dtype=torch.long)
        for h, sz in enumerate(action_sizes):
            actions[:, :, h] = torch.randint(0, sz, (B, n_agents))

        onehot = QMixLearner._build_compound_onehot(learner, actions)
        # Reshape to [B, N, total_actions] to check per-agent
        per_agent = onehot.view(B, n_agents, total_actions)
        for b in range(B):
            for a in range(n_agents):
                nnz = (per_agent[b, a] > 0).sum().item()
                assert nnz == num_heads, (
                    f"Expected {num_heads} non-zero entries per agent, got {nnz} "
                    f"(batch={b}, agent={a})"
                )

    def test_single_head_is_true_onehot(self):
        """With a single action head, compound onehot should be a true one-hot."""
        learner = object.__new__(QMixLearner)
        learner.agent_net = SimpleNamespace(action_sizes=[5])
        n_agents = 2
        B = 4
        actions = torch.randint(0, 5, (B, n_agents, 1))
        onehot = QMixLearner._build_compound_onehot(learner, actions)
        per_agent = onehot.view(B, n_agents, 5)
        for b in range(B):
            for a in range(n_agents):
                nnz = (per_agent[b, a] > 0).sum().item()
                assert nnz == 1, f"Single head should be true one-hot, got {nnz} active"


class TestSIWeightCapacityCompoundActions:
    """
    With compound actions and adv_hypernet_layers=1 (the qatten default),
    the SI weight network is a single linear layer mapping from
    (state_dim + action_dim) -> n_agents.  This may lack capacity for
    the 14-dim multi-hot action representation from 5 heads.
    """

    def test_1layer_si_distinguishes_different_compound_actions(self):
        """
        A 1-layer SI weight network should produce different weights for
        different compound actions (same state).  If it cannot, the advantage
        decomposition is action-blind.
        """
        n_agents, state_dim, n_actions = 2, 64, 14
        si = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions, num_kernel=4, adv_hypernet_layers=1,
        )
        B = 1
        states = torch.randn(B, state_dim)
        # Two distinct action vectors
        actions_a = torch.zeros(B, n_agents * n_actions)
        actions_a[0, 0] = 1; actions_a[0, 3] = 1; actions_a[0, 6] = 1  # head 0-2 for agent 0
        actions_a[0, 14] = 1; actions_a[0, 17] = 1; actions_a[0, 20] = 1  # head 0-2 for agent 1

        actions_b = torch.zeros(B, n_agents * n_actions)
        actions_b[0, 2] = 1; actions_b[0, 5] = 1; actions_b[0, 8] = 1
        actions_b[0, 16] = 1; actions_b[0, 19] = 1; actions_b[0, 22] = 1

        w_a = si(states, actions_a)
        w_b = si(states, actions_b)
        # Weights should differ for different actions
        assert not torch.allclose(w_a, w_b, atol=1e-6), (
            "SI weight produced identical outputs for different compound actions"
        )

    def test_3layer_si_has_more_capacity(self):
        """3-layer SI weight should have strictly more parameters than 1-layer."""
        n_agents, state_dim, n_actions = 2, 64, 14
        si_1 = DMAQ_SI_Weight(n_agents, state_dim, n_actions, num_kernel=4, adv_hypernet_layers=1)
        si_3 = DMAQ_SI_Weight(n_agents, state_dim, n_actions, num_kernel=4, adv_hypernet_layers=3)
        params_1 = sum(p.numel() for p in si_1.parameters())
        params_3 = sum(p.numel() for p in si_3.parameters())
        assert params_3 > params_1 * 5, (
            f"3-layer ({params_3}) should have >> 1-layer ({params_1}) params"
        )


class TestCompoundMaxQDecomposition:
    """
    Verify that the compound max Q decomposition is correct:
    max_{a1,...,aH} sum_h Q_h(s, a_h) == sum_h max_{a_h} Q_h(s, a_h)
    This holds because the per-head Q values are independent (separate linear
    heads on a shared encoder).
    """

    def test_max_q_i_equals_sum_of_per_head_maxes(self):
        """_compute_max_q_i should equal brute-force compound max."""
        learner = object.__new__(QMixLearner)
        action_sizes = [3, 3, 3, 2, 3]
        learner.agent_net = SimpleNamespace(action_sizes=action_sizes)

        B, n_agents = 4, 2
        total_actions = sum(action_sizes)
        q_logits = torch.randn(B, n_agents, total_actions)

        # _compute_max_q_i (sum of per-head maxes)
        max_q = QMixLearner._compute_max_q_i(learner, q_logits)

        # brute force over all 162 combinations
        import itertools
        brute_max = torch.full((B, n_agents), float('-inf'))
        all_combos = list(itertools.product(*[range(s) for s in action_sizes]))
        for combo in all_combos:
            q_val = torch.zeros(B, n_agents)
            offset = 0
            for h, (a, sz) in enumerate(zip(combo, action_sizes)):
                q_val += q_logits[:, :, offset + a]
                offset += sz
            brute_max = torch.max(brute_max, q_val)

        assert torch.allclose(max_q, brute_max, atol=1e-5), (
            f"max_q_i mismatch:\n  per-head: {max_q}\n  brute: {brute_max}"
        )
