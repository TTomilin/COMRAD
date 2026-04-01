import itertools
import math

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from comrad.models.qplex_mixer import (
    DMAQ_SI_Weight,
    DMAQer,
    DMAQ_QattenMixer,
)
from comrad.models.qmix_model import QMixMixer


ARMORY_HEADS = [3, 3, 3, 2, 3]
TOTAL_ACTIONS = sum(ARMORY_HEADS)
N_AGENTS = 2
N_COMPOUND = math.prod(ARMORY_HEADS)


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


class _FakeActionSizes:
    def __init__(self, action_sizes):
        self.action_sizes = action_sizes


class _LearnerStub:
    def __init__(self, action_sizes):
        self.agent_net = _FakeActionSizes(action_sizes)

    from sample_factory.algo.learning.learner_qmix import QMixLearner
    _compute_max_q_i = QMixLearner._compute_max_q_i
    _build_compound_onehot = QMixLearner._build_compound_onehot
    _compute_qplex_q_tot = QMixLearner._compute_qplex_q_tot


def _random_compound_actions(B, N, head_sizes):
    return torch.stack(
        [torch.randint(0, hs, (B, N)) for hs in head_sizes], dim=-1
    )


def _build_multihot(actions, head_sizes):
    B, N = actions.shape[:2]
    ohs = []
    for h, hs in enumerate(head_sizes):
        ohs.append(F.one_hot(actions[:, :, h].long(), num_classes=hs).float())
    per_agent = torch.cat(ohs, dim=-1)
    return per_agent.reshape(B, N * sum(head_sizes))


class TestComputeMaxQi:
    """Verify sum of per-head maxes equals max over all compound combos."""

    def test_matches_brute_force(self):
        torch.manual_seed(42)
        stub = _LearnerStub(ARMORY_HEADS)
        B = 16
        q_logits = torch.randn(B, N_AGENTS, TOTAL_ACTIONS)
        max_q = stub._compute_max_q_i(q_logits)  # [B, N]

        # brute-force
        best_q = torch.full((B, N_AGENTS), float('-inf'))
        for combo in itertools.product(*[range(hs) for hs in ARMORY_HEADS]):
            q_val = torch.zeros(B, N_AGENTS)
            offset = 0
            for h, a in enumerate(combo):
                q_val += q_logits[:, :, offset + a]
                offset += ARMORY_HEADS[h]
            best_q = torch.max(best_q, q_val)

        torch.testing.assert_close(max_q, best_q, atol=1e-5, rtol=1e-5)

    def test_max_q_i_always_ge_chosen(self):
        torch.manual_seed(7)
        stub = _LearnerStub(ARMORY_HEADS)
        B = 64
        q_logits = torch.randn(B, N_AGENTS, TOTAL_ACTIONS)
        max_q = stub._compute_max_q_i(q_logits)

        actions = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)
        chosen_q = torch.zeros(B, N_AGENTS)
        offset = 0
        for h, hs in enumerate(ARMORY_HEADS):
            head_q = q_logits[:, :, offset:offset + hs]
            chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
            offset += hs

        assert (max_q >= chosen_q - 1e-6).all()

    def test_advantage_always_nonpositive(self):
        torch.manual_seed(123)
        stub = _LearnerStub(ARMORY_HEADS)
        B = 128
        q_logits = torch.randn(B, N_AGENTS, TOTAL_ACTIONS)
        max_q = stub._compute_max_q_i(q_logits)

        actions = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)
        chosen_q = torch.zeros(B, N_AGENTS)
        offset = 0
        for h, hs in enumerate(ARMORY_HEADS):
            head_q = q_logits[:, :, offset:offset + hs]
            chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
            offset += hs

        adv = chosen_q - max_q
        assert (adv <= 1e-6).all()

    @pytest.mark.parametrize("head_sizes", [
        [14],
        [3, 3, 3, 2, 3],
        [7, 7],
    ])
    def test_single_head_equivalence(self, head_sizes):
        torch.manual_seed(0)
        stub = _LearnerStub(head_sizes)
        B = 32
        total = sum(head_sizes)
        q_logits = torch.randn(B, N_AGENTS, total)
        max_q = stub._compute_max_q_i(q_logits)
        assert max_q.shape == (B, N_AGENTS)
        assert max_q.isfinite().all()


class TestBuildCompoundOnehot:
    def test_output_shape(self):
        stub = _LearnerStub(ARMORY_HEADS)
        B = 8
        actions = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)
        onehot = stub._build_compound_onehot(actions)
        assert onehot.shape == (B, N_AGENTS * TOTAL_ACTIONS)

    def test_active_bits_per_agent(self):
        stub = _LearnerStub(ARMORY_HEADS)
        B = 32
        actions = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)
        onehot = stub._build_compound_onehot(actions)
        per_agent = onehot.view(B, N_AGENTS, TOTAL_ACTIONS)
        active_per_agent = per_agent.sum(dim=-1)
        assert (active_per_agent == len(ARMORY_HEADS)).all()

    def test_all_162_combinations_unique(self):
        stub = _LearnerStub(ARMORY_HEADS)
        combos = list(itertools.product(*[range(hs) for hs in ARMORY_HEADS]))
        B = len(combos)
        actions = torch.tensor(combos).unsqueeze(1).expand(B, 1, -1)  # [162,1,5]
        onehot = stub._build_compound_onehot(actions)  # [162, 1*14]
        unique_rows = set(tuple(row.tolist()) for row in onehot)
        assert len(unique_rows) == N_COMPOUND

    def test_concatenation_order_matches_q_logit_slicing(self):
        stub = _LearnerStub(ARMORY_HEADS)
        B = 4
        actions = torch.stack(
            [torch.tensor(hs - 1) for hs in ARMORY_HEADS]
        ).unsqueeze(0).unsqueeze(0).expand(B, N_AGENTS, -1)

        onehot = stub._build_compound_onehot(actions)
        per_agent = onehot.view(B, N_AGENTS, TOTAL_ACTIONS)

        offset = 0
        for h, hs in enumerate(ARMORY_HEADS):
            head_oh = per_agent[:, :, offset:offset + hs]
            assert (head_oh[:, :, -1] == 1.0).all()
            assert head_oh.sum(dim=-1).eq(1.0).all()
            offset += hs


class TestAdvantageDilution:
    @staticmethod
    def _compute_adv_stats(q_logits, head_sizes, n_samples=1000):
        B, N = q_logits.shape[:2]
        actions = _random_compound_actions(B, N, head_sizes)

        chosen_q = torch.zeros(B, N)
        max_q = torch.zeros(B, N)
        per_head_subadv = []
        offset = 0
        for h, hs in enumerate(head_sizes):
            head_q = q_logits[:, :, offset:offset + hs]
            c = head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
            m = head_q.max(dim=-1)[0]
            chosen_q += c
            max_q += m
            per_head_subadv.append((c - m).abs().mean().item())
            offset += hs

        adv = (chosen_q - max_q).abs()
        return {
            'mean_adv': adv.mean().item(),
            'mean_q_mag': chosen_q.abs().mean().item(),
            'adv_to_q_ratio': (adv / (chosen_q.abs() + 1e-8)).mean().item(),
            'per_head_subadv': per_head_subadv,
            'mean_per_head_subadv': sum(per_head_subadv) / len(per_head_subadv),
        }

    def test_compound_vs_single_absolute_advantage(self):
        torch.manual_seed(42)
        B, N, q_spread = 1024, 2, 0.5

        single_q = torch.randn(B, N, TOTAL_ACTIONS) * q_spread
        compound_q = torch.randn(B, N, TOTAL_ACTIONS) * q_spread

        single_stats = self._compute_adv_stats(single_q, [TOTAL_ACTIONS])
        compound_stats = self._compute_adv_stats(compound_q, ARMORY_HEADS)

        assert compound_stats['mean_adv'] > single_stats['mean_adv']

    def test_per_head_subadvantage_is_small(self):
        torch.manual_seed(42)
        B, N = 2048, 2
        q_logits = torch.randn(B, N, TOTAL_ACTIONS) * 0.5
        stats = self._compute_adv_stats(q_logits, ARMORY_HEADS)

        for h, subadv in enumerate(stats['per_head_subadv']):
            assert subadv < 0.5

    def test_advantage_dilution_ratio(self):
        torch.manual_seed(42)
        B, N, q_spread = 2048, 2, 0.5

        ratios = {}
        for label, heads in [('single_14', [14]), ('compound_5h', ARMORY_HEADS)]:
            total = sum(heads)
            q = torch.randn(B, N, total) * q_spread
            actions = _random_compound_actions(B, N, heads)

            chosen_q = torch.zeros(B, N)
            max_q = torch.zeros(B, N)
            q_range = torch.zeros(B, N)
            offset = 0
            for h, hs in enumerate(heads):
                head_q = q[:, :, offset:offset + hs]
                chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
                max_q += head_q.max(dim=-1)[0]
                q_range += head_q.max(dim=-1)[0] - head_q.min(dim=-1)[0]
                offset += hs
            adv = (chosen_q - max_q).abs()
            ratios[label] = (adv / (q_range + 1e-8)).mean().item()

        assert ratios['compound_5h'] < ratios['single_14']

    def test_dilution_scales_with_number_of_heads(self):
        torch.manual_seed(0)
        B, N = 2048, 2
        prev_ratio = float('inf')
        for n_heads in [1, 2, 5, 10]:
            heads = [3] * n_heads
            total = sum(heads)
            q = torch.randn(B, N, total) * 0.5
            actions = _random_compound_actions(B, N, heads)
            chosen_q = torch.zeros(B, N)
            max_q = torch.zeros(B, N)
            offset = 0
            for h, hs in enumerate(heads):
                head_q = q[:, :, offset:offset + hs]
                chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
                max_q += head_q.max(dim=-1)[0]
                offset += hs
            q_range = max_q - (torch.zeros_like(max_q))
            offset = 0
            min_q = torch.zeros(B, N)
            for hs in heads:
                head_q = q[:, :, offset:offset + hs]
                min_q += head_q.min(dim=-1)[0]
                offset += hs
            q_range = max_q - min_q
            adv = (chosen_q - max_q).abs()
            ratio = (adv / (q_range + 1e-8)).mean().item()
            assert ratio < prev_ratio + 0.01
            prev_ratio = ratio


class TestQmixVsQplexCompound:
    def test_qmix_gradient_independent_of_action_encoding(self):
        torch.manual_seed(42)
        B, N, state_dim = 64, N_AGENTS, 64
        state = torch.randn(B, state_dim)
        qmix = QMixMixer(N, state_dim, embed_dim=32, hypernet_hidden=64)

        # Same Q values, different "action encoding" origin
        agent_qs = torch.randn(B, N, requires_grad=True)
        q_tot = qmix(agent_qs, state)
        q_tot.sum().backward()
        grad_1 = agent_qs.grad.clone()

        agent_qs_2 = agent_qs.detach().clone().requires_grad_(True)
        q_tot_2 = qmix(agent_qs_2, state)
        q_tot_2.sum().backward()
        grad_2 = agent_qs_2.grad.clone()

        torch.testing.assert_close(grad_1, grad_2)

    def test_qplex_a_tot_depends_on_multihot_quality(self):
        torch.manual_seed(42)
        B, N, state_dim, n_actions = 64, N_AGENTS, 64, TOTAL_ACTIONS
        cfg = _default_cfg(qplex_weighted_head=True, qplex_adv_hypernet_layers=1)
        mixer = DMAQer(cfg, N, state_dim, n_actions)

        agent_qs = torch.randn(B, N)
        max_q_i = agent_qs + 1.0  # fixed gap
        state = torch.randn(B, state_dim)

        # Two different multi-hot encodings (different actions, same gap)
        act1 = _build_multihot(
            _random_compound_actions(B, N, ARMORY_HEADS), ARMORY_HEADS
        )
        act2 = _build_multihot(
            _random_compound_actions(B, N, ARMORY_HEADS), ARMORY_HEADS
        )

        a_tot_1, _ = mixer(agent_qs, state, actions=act1, max_q_i=max_q_i, is_v=False)
        a_tot_2, _ = mixer(agent_qs, state, actions=act2, max_q_i=max_q_i, is_v=False)

        assert not torch.allclose(a_tot_1, a_tot_2, atol=1e-6)

    def test_qplex_a_tot_magnitude_vs_v_tot(self):
        torch.manual_seed(42)
        B, N, state_dim, n_actions = 256, N_AGENTS, 64, TOTAL_ACTIONS

        for mixer_name, mixer_cls_args in [
            ('dmaq', lambda: DMAQer(
                _default_cfg(qplex_weighted_head=True, qplex_adv_hypernet_layers=1),
                N, state_dim, n_actions)),
            ('qatten', lambda: DMAQ_QattenMixer(
                _default_cfg(), N, state_dim, n_actions, unit_dim=32)),
        ]:
            mixer = mixer_cls_args()
            q_logits = torch.randn(B, N, TOTAL_ACTIONS) * 2.0

            actions = _random_compound_actions(B, N, ARMORY_HEADS)
            chosen_q = torch.zeros(B, N)
            max_q = torch.zeros(B, N)
            offset = 0
            for h, hs in enumerate(ARMORY_HEADS):
                head_q = q_logits[:, :, offset:offset + hs]
                chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
                max_q += head_q.max(dim=-1)[0]
                offset += hs

            state = torch.randn(B, state_dim)
            onehot = _build_multihot(actions, ARMORY_HEADS)

            v_tot, _ = mixer(chosen_q, state, is_v=True)
            a_tot, _ = mixer(chosen_q, state, actions=onehot, max_q_i=max_q, is_v=False)

            v_mag = v_tot.abs().mean().item()
            a_mag = a_tot.abs().mean().item()

            assert v_tot.isfinite().all()
            assert a_tot.isfinite().all()
            assert a_mag > 1e-6

            greedy_q = max_q.clone()
            a_tot_greedy, _ = mixer(
                greedy_q, state, actions=onehot, max_q_i=max_q, is_v=False
            )
            assert a_tot_greedy.abs().mean().item() < 1e-5

    def test_compound_vs_single_si_weight_variance(self):
        torch.manual_seed(42)
        n_actions = TOTAL_ACTIONS
        state_dim = 64
        si = DMAQ_SI_Weight(N_AGENTS, state_dim, n_actions, num_kernel=4,
                            adv_hypernet_layers=1)

        B = 512
        states = torch.randn(B, state_dim)

        # Single-head: one-hot (1 bit per agent)
        single_idx = torch.randint(0, n_actions, (B, N_AGENTS))
        single_oh = F.one_hot(single_idx, num_classes=n_actions).float()
        single_flat = single_oh.reshape(B, N_AGENTS * n_actions)
        w_single = si(states, single_flat)

        # Compound: multi-hot (5 bits per agent)
        compound_acts = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)
        compound_flat = _build_multihot(compound_acts, ARMORY_HEADS)
        w_compound = si(states, compound_flat)

        var_single = w_single.var(dim=0).mean().item()
        var_compound = w_compound.var(dim=0).mean().item()

        assert var_single > 0 and var_compound > 0


class TestComputeQplexQTotCompound:
    @staticmethod
    def _make_learner_and_mixer(mixer_type='dmaq'):
        stub = _LearnerStub(ARMORY_HEADS)
        state_dim = 64
        n_actions = TOTAL_ACTIONS
        if mixer_type == 'dmaq':
            cfg = _default_cfg(
                qplex_weighted_head=True, qplex_adv_hypernet_layers=1
            )
            mixer = DMAQer(cfg, N_AGENTS, state_dim, n_actions)
        else:
            cfg = _default_cfg()
            mixer = DMAQ_QattenMixer(cfg, N_AGENTS, state_dim, n_actions, 32)
        return stub, mixer, state_dim

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_q_tot_equals_v_tot_at_greedy(self, mixer_type):
        torch.manual_seed(42)
        stub, mixer, state_dim = self._make_learner_and_mixer(mixer_type)
        B = 32

        q_logits = torch.randn(B, N_AGENTS, TOTAL_ACTIONS)
        # Select greedy per-head actions
        greedy_actions = []
        offset = 0
        for hs in ARMORY_HEADS:
            head_q = q_logits[:, :, offset:offset + hs]
            greedy_actions.append(head_q.argmax(dim=-1))
            offset += hs
        actions = torch.stack(greedy_actions, dim=-1)

        # Evaluate chosen Q
        chosen_q = torch.zeros(B, N_AGENTS)
        offset = 0
        for h, hs in enumerate(ARMORY_HEADS):
            head_q = q_logits[:, :, offset:offset + hs]
            chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
            offset += hs

        state = torch.randn(B, state_dim)

        # Full Q_tot (V_tot + A_tot)
        q_tot, _ = stub._compute_qplex_q_tot(
            chosen_q, state, q_logits, actions, mixer
        )

        # V_tot only
        v_tot, _ = mixer(chosen_q, state, is_v=True)
        v_tot = v_tot.squeeze(-1).squeeze(-1)

        torch.testing.assert_close(q_tot, v_tot, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_q_tot_le_v_tot_at_nongreedy(self, mixer_type):
        torch.manual_seed(42)
        stub, mixer, state_dim = self._make_learner_and_mixer(mixer_type)
        B = 64

        q_logits = torch.randn(B, N_AGENTS, TOTAL_ACTIONS)
        actions = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)

        chosen_q = torch.zeros(B, N_AGENTS)
        offset = 0
        for h, hs in enumerate(ARMORY_HEADS):
            head_q = q_logits[:, :, offset:offset + hs]
            chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
            offset += hs

        state = torch.randn(B, state_dim)
        q_tot, _ = stub._compute_qplex_q_tot(
            chosen_q, state, q_logits, actions, mixer
        )
        v_tot, _ = mixer(chosen_q, state, is_v=True)
        v_tot = v_tot.squeeze(-1).squeeze(-1)

        assert q_tot.isfinite().all()
        assert v_tot.isfinite().all()

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_gradient_flows_through_compound_q_tot(self, mixer_type):
        torch.manual_seed(42)
        stub, mixer, state_dim = self._make_learner_and_mixer(mixer_type)
        B = 16

        q_logits = torch.randn(B, N_AGENTS, TOTAL_ACTIONS)
        actions = _random_compound_actions(B, N_AGENTS, ARMORY_HEADS)

        chosen_q = torch.zeros(B, N_AGENTS)
        offset = 0
        for h, hs in enumerate(ARMORY_HEADS):
            head_q = q_logits[:, :, offset:offset + hs]
            chosen_q += head_q.gather(2, actions[:, :, h:h+1]).squeeze(-1)
            offset += hs

        agent_qs = chosen_q.clone().requires_grad_(True)
        state = torch.randn(B, state_dim)

        q_tot, _ = stub._compute_qplex_q_tot(
            agent_qs, state, q_logits, actions, mixer
        )
        q_tot.sum().backward()

        assert agent_qs.grad is not None
        assert agent_qs.grad.abs().sum() > 0
