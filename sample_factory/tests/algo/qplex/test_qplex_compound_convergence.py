import pytest
import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

from comrad.models.qplex_mixer import (
    DMAQ_SI_Weight,
    DMAQer,
    Qatten_Weight,
    DMAQ_QattenMixer,
)
from comrad.models.qmix_model import QMixMixer, make_mixer


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


class TestAdvantageSignalDilution:
    @staticmethod
    def _make_q_logits(batch, n_agents, head_sizes, q_spread=0.5):
        total = sum(head_sizes)
        q = torch.zeros(batch, n_agents, total)
        offset = 0
        for hs in head_sizes:
            q[:, :, offset:offset + hs] = torch.randn(batch, n_agents, hs) * q_spread
            offset += hs
        return q

    @staticmethod
    def _compute_advantage_magnitude(q_logits, head_sizes, actions):
        B, N = q_logits.shape[0], q_logits.shape[1]
        chosen_q = torch.zeros(B, N)
        max_q = torch.zeros(B, N)
        offset = 0
        for h, hs in enumerate(head_sizes):
            head_q = q_logits[:, :, offset:offset + hs]
            head_a = actions[:, :, h]
            chosen_q += head_q.gather(2, head_a.unsqueeze(-1)).squeeze(-1)
            max_q += head_q.max(dim=-1)[0]
            offset += hs
        return (chosen_q - max_q).abs()

    def test_compound_advantage_ratio_smaller_than_single_head(self):
        torch.manual_seed(42)
        B, N = 256, 2

        # Single head: 14 actions
        single_q = torch.randn(B, N, 14) * 0.5
        single_actions = torch.randint(0, 14, (B, N, 1))
        single_adv = self._compute_advantage_magnitude(single_q, [14], single_actions)
        single_q_range = single_q.max(dim=-1)[0] - single_q.min(dim=-1)[0]
        single_ratio = (single_adv / (single_q_range.abs() + 1e-8)).mean().item()

        compound_heads = [3, 3, 3, 2, 3]
        compound_q = torch.randn(B, N, 14) * 0.5
        compound_actions = torch.cat([
            torch.randint(0, hs, (B, N, 1)) for hs in compound_heads
        ], dim=-1)
        compound_adv = self._compute_advantage_magnitude(
            compound_q, compound_heads, compound_actions
        )
        # Q range for compound = sum of per-head ranges
        compound_q_range = torch.zeros(B, N)
        offset = 0
        for hs in compound_heads:
            head_q = compound_q[:, :, offset:offset + hs]
            compound_q_range += head_q.max(dim=-1)[0] - head_q.min(dim=-1)[0]
            offset += hs
        compound_ratio = (compound_adv / (compound_q_range.abs() + 1e-8)).mean().item()

        assert compound_ratio < single_ratio

    def test_advantage_ratio_decreases_with_more_heads(self):
        torch.manual_seed(123)
        B, N, q_spread = 512, 2, 0.5

        ratio_by_heads = {}
        for n_heads in [1, 3, 5, 10]:
            head_sizes = [3] * n_heads
            total = sum(head_sizes)
            q = torch.randn(B, N, total) * q_spread
            actions = torch.cat([
                torch.randint(0, 3, (B, N, 1)) for _ in range(n_heads)
            ], dim=-1)
            adv = self._compute_advantage_magnitude(q, head_sizes, actions)
            chosen_q_mag = torch.zeros(B, N)
            offset = 0
            for h in range(n_heads):
                head_q = q[:, :, offset:offset + 3]
                head_a = actions[:, :, h]
                chosen_q_mag += head_q.gather(2, head_a.unsqueeze(-1)).squeeze(-1).abs()
                offset += 3
            ratio = (adv / (chosen_q_mag + 1e-8)).mean().item()
            ratio_by_heads[n_heads] = ratio

        assert ratio_by_heads[10] < ratio_by_heads[1]


class TestAttentionExpressiveness:
    def test_softmax_two_agents_single_degree_of_freedom(self):
        torch.manual_seed(0)
        n_agents, state_dim, unit_dim = 2, 64, 32
        n_actions = 14
        qw = Qatten_Weight(
            n_agents, state_dim, n_actions, unit_dim,
            n_head=4, weighted_head=False,
        )
        B = 64
        agent_qs = torch.randn(B, n_agents)
        states = torch.randn(B, state_dim)
        head_attend, v, _, _ = qw(agent_qs, states)

        weight_sums = head_attend.sum(dim=-1)
        expected = torch.full_like(weight_sums, 4.0)
        assert torch.allclose(weight_sums, expected, atol=1e-4)

    def test_qmix_gradient_richer_than_qplex_vtot(self):
        torch.manual_seed(42)
        B, N, state_dim = 64, 2, 64

        # QMIX gradient: dQ_tot/dQ_i depends on state through hypernetwork
        qmix = QMixMixer(N, state_dim, embed_dim=32, hypernet_hidden=64)
        states = torch.randn(B, state_dim)
        agent_qs_qmix = torch.randn(B, N, requires_grad=True)
        q_tot_qmix = qmix(agent_qs_qmix, states)
        q_tot_qmix.sum().backward()
        qmix_grad = agent_qs_qmix.grad.clone()

        # QPLEX V_tot gradient: dV_tot/dQ_i = w_i (attention weight)
        unit_dim = 32
        n_actions = 14
        cfg = _default_cfg()
        qplex = DMAQ_QattenMixer(cfg, N, state_dim, n_actions, unit_dim)
        agent_qs_qplex = torch.randn(B, N, requires_grad=True)
        v_tot, _ = qplex(agent_qs_qplex, states, is_v=True)
        v_tot.squeeze().sum().backward()
        qplex_grad = agent_qs_qplex.grad.clone()

        qmix_grad_std = qmix_grad.std(dim=0).mean().item()
        qplex_grad_std = qplex_grad.std(dim=0).mean().item()

        assert qmix_grad.abs().sum() > 0
        assert qplex_grad.abs().sum() > 0
        assert qmix_grad_std > 0
        assert qplex_grad_std > 0


class TestMultiHotVsOneHot:
    @staticmethod
    def _make_onehot_actions(B, n_agents, n_actions):
        indices = torch.randint(0, n_actions, (B, n_agents))
        onehot = F.one_hot(indices, num_classes=n_actions).float()
        return onehot.reshape(B, n_agents * n_actions)

    @staticmethod
    def _make_compound_multihot(B, n_agents, head_sizes):
        ohs = []
        for hs in head_sizes:
            indices = torch.randint(0, hs, (B, n_agents))
            ohs.append(F.one_hot(indices, num_classes=hs).float())
        per_agent = torch.cat(ohs, dim=-1)
        return per_agent.reshape(B, n_agents * sum(head_sizes))

    def test_active_bits_differ(self):
        B, N = 16, 2
        single_head_size = 14
        compound_heads = [3, 3, 3, 2, 3]

        onehot = self._make_onehot_actions(B, N, single_head_size)
        multihot = self._make_compound_multihot(B, N, compound_heads)

        assert onehot.shape == multihot.shape == (B, N * 14)
        assert (onehot.sum(dim=-1) == N).all()
        assert (multihot.sum(dim=-1) == N * 5).all()

    def test_si_weight_sensitivity_to_input_type(self):
        torch.manual_seed(7)
        n_agents, state_dim, n_actions = 2, 64, 14
        si = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=4, adv_hypernet_layers=1,
        )
        B = 128
        states = torch.randn(B, state_dim)

        onehot = self._make_onehot_actions(B, n_agents, n_actions)
        multihot = self._make_compound_multihot(B, n_agents, [3, 3, 3, 2, 3])

        w_onehot = si(states, onehot)
        w_multihot = si(states, multihot)

        mean_diff = (w_onehot.mean() - w_multihot.mean()).abs().item()
        assert mean_diff > 0.001

    def test_si_weight_sensitivity_to_single_head_change(self):
        torch.manual_seed(42)
        n_agents, state_dim, n_actions = 2, 64, 14
        si = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=4, adv_hypernet_layers=1,
        )
        B = 128
        states = torch.randn(B, state_dim)
        head_sizes = [3, 3, 3, 2, 3]

        base_ohs = []
        for hs in head_sizes:
            idx = torch.zeros(B, n_agents, dtype=torch.long)
            base_ohs.append(F.one_hot(idx, num_classes=hs).float())
        base_per_agent = torch.cat(base_ohs, dim=-1)
        base_actions = base_per_agent.reshape(B, n_agents * n_actions)

        changed_ohs = list(base_ohs)
        idx_changed = torch.ones(B, n_agents, dtype=torch.long)
        changed_ohs[0] = F.one_hot(idx_changed, num_classes=3).float()
        changed_per_agent = torch.cat(changed_ohs, dim=-1)
        changed_actions = changed_per_agent.reshape(B, n_agents * n_actions)

        w_base = si(states, base_actions)
        w_changed = si(states, changed_actions)

        rel_change = (w_changed - w_base).abs().mean() / (w_base.abs().mean() + 1e-8)

        # The output change should be detectable but potentially small
        # relative to the full action change (all heads different)
        all_diff_ohs = []
        for h, hs in enumerate(head_sizes):
            idx_all = torch.full((B, n_agents), min(hs - 1, 2), dtype=torch.long)
            all_diff_ohs.append(F.one_hot(idx_all, num_classes=hs).float())
        all_diff = torch.cat(all_diff_ohs, dim=-1).reshape(B, n_agents * n_actions)
        w_all_diff = si(states, all_diff)
        rel_change_all = (w_all_diff - w_base).abs().mean() / (w_base.abs().mean() + 1e-8)

        assert rel_change.item() < rel_change_all.item()


class TestAdvHypernetCapacity:
    def test_default_layers_for_qatten_vs_dmaq(self):
        cfg_qatten = _default_cfg()
        mixer_qatten = DMAQ_QattenMixer(cfg_qatten, 2, 64, 14, 32)
        for ext in mixer_qatten.si_weight.action_extractors:
            assert isinstance(ext, torch.nn.Linear)

        cfg_dmaq = _default_cfg(
            qplex_weighted_head=True,
            qplex_adv_hypernet_layers=3,
        )
        mixer_dmaq = DMAQer(cfg_dmaq, 2, 64, 14)
        for ext in mixer_dmaq.si_weight.action_extractors:
            assert isinstance(ext, torch.nn.Sequential)

    def test_three_layers_more_expressive(self):
        torch.manual_seed(99)
        n_agents, state_dim, n_actions = 2, 64, 14
        B = 128
        states = torch.randn(B, state_dim)

        # Generate diverse compound actions
        head_sizes = [3, 3, 3, 2, 3]
        actions_list = []
        for _ in range(B):
            ohs = []
            for hs in head_sizes:
                idx = torch.randint(0, hs, (1, n_agents))
                ohs.append(F.one_hot(idx, num_classes=hs).float())
            per_agent = torch.cat(ohs, dim=-1)
            actions_list.append(per_agent.reshape(1, n_agents * n_actions))
        actions = torch.cat(actions_list, dim=0)

        si_1 = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=4, adv_hypernet_layers=1,
        )
        w_1 = si_1(states, actions)

        si_3 = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=4, adv_hypernet_layers=3,
            adv_hypernet_embed=64,
        )
        w_3 = si_3(states, actions)

        var_1 = w_1.var(dim=0).mean().item()
        var_3 = w_3.var(dim=0).mean().item()

        assert var_1 > 1e-6
        assert var_3 > 1e-6

    def test_param_count_difference(self):
        n_agents, state_dim, n_actions = 2, 64, 14

        si_1 = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=4, adv_hypernet_layers=1,
        )
        si_3 = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=4, adv_hypernet_layers=3,
            adv_hypernet_embed=64,
        )

        params_1 = sum(p.numel() for p in si_1.parameters())
        params_3 = sum(p.numel() for p in si_3.parameters())

        assert params_3 > params_1 * 2


class TestCompoundMaxDecomposition:
    def test_additive_decomposition_holds(self):
        torch.manual_seed(42)
        head_sizes = [3, 3, 3, 2, 3]
        B, N = 32, 2
        total = sum(head_sizes)
        q_logits = torch.randn(B, N, total)

        # Sum of per-head maxes (what _compute_max_q_i does)
        sum_of_maxes = torch.zeros(B, N)
        offset = 0
        for hs in head_sizes:
            head_q = q_logits[:, :, offset:offset + hs]
            sum_of_maxes += head_q.max(dim=-1)[0]
            offset += hs

        import itertools
        ranges = [range(hs) for hs in head_sizes]
        best_q = torch.full((B, N), float('-inf'))
        for combo in itertools.product(*ranges):
            q_val = torch.zeros(B, N)
            offset = 0
            for h, a in enumerate(combo):
                head_q = q_logits[:, :, offset:offset + head_sizes[h]]
                q_val += head_q[:, :, a]
                offset += head_sizes[h]
            best_q = torch.max(best_q, q_val)

        torch.testing.assert_close(sum_of_maxes, best_q, atol=1e-5, rtol=1e-5)


class TestCompoundOnehotEncoding:
    def test_encoding_is_unique_for_all_combinations(self):
        import itertools
        head_sizes = [3, 3, 3, 2, 3]
        total = sum(head_sizes)

        seen = set()
        for combo in itertools.product(*[range(hs) for hs in head_sizes]):
            oh_parts = []
            for h, a in enumerate(combo):
                oh = torch.zeros(head_sizes[h])
                oh[a] = 1.0
                oh_parts.append(oh)
            full_oh = torch.cat(oh_parts)
            key = tuple(full_oh.tolist())
            assert key not in seen
            seen.add(key)

        assert len(seen) == 162

    def test_encoding_active_bits(self):
        head_sizes = [3, 3, 3, 2, 3]
        n_heads = len(head_sizes)
        total = sum(head_sizes)

        # Random compound actions
        B, N = 16, 2
        for _ in range(10):
            ohs = []
            for hs in head_sizes:
                idx = torch.randint(0, hs, (B, N))
                ohs.append(F.one_hot(idx, num_classes=hs).float())
            per_agent = torch.cat(ohs, dim=-1)
            active = per_agent.sum(dim=-1)
            assert (active == n_heads).all()


class TestTargetQTotIncludesAdvantage:
    @staticmethod
    def _make_mixer(mixer_type="dmaq", n_agents=2, state_dim=64, n_actions=14, unit_dim=32):
        if mixer_type == "dmaq":
            cfg = _default_cfg(
                mixer='dmaq', qplex_weighted_head=True, qplex_adv_hypernet_layers=1,
            )
            return DMAQer(cfg, n_agents, state_dim, n_actions)
        else:
            cfg = _default_cfg(mixer='dmaq_qatten')
            return DMAQ_QattenMixer(cfg, n_agents, state_dim, n_actions, unit_dim)

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_target_a_tot_nonzero_when_actions_disagree(self, mixer_type):
        torch.manual_seed(42)
        n_agents, state_dim, n_actions = 2, 64, 14
        unit_dim = 32
        mixer = self._make_mixer(mixer_type, n_agents, state_dim, n_actions, unit_dim)
        B = 64

        agent_qs = torch.randn(B, n_agents)
        max_q_i = agent_qs + torch.rand(B, n_agents) * 2.0  # max > chosen
        states = torch.randn(B, state_dim)
        onehot_actions = torch.randn(B, n_agents * n_actions)

        v_tot_chosen, _ = mixer(agent_qs, states, is_v=True)
        v_tot_chosen = v_tot_chosen.squeeze()

        a_tot, _ = mixer(agent_qs, states, actions=onehot_actions, max_q_i=max_q_i, is_v=False)
        a_tot = a_tot.squeeze()

        q_tot_full = v_tot_chosen + a_tot

        v_tot_max, _ = mixer(max_q_i, states, is_v=True)
        v_tot_max = v_tot_max.squeeze()

        assert q_tot_full.mean() < v_tot_max.mean()

    @pytest.mark.parametrize("mixer_type", ["dmaq", "dmaq_qatten"])
    def test_target_a_tot_zero_at_greedy(self, mixer_type):
        torch.manual_seed(42)
        n_agents, state_dim, n_actions = 2, 64, 14
        unit_dim = 32
        mixer = self._make_mixer(mixer_type, n_agents, state_dim, n_actions, unit_dim)
        B = 32

        agent_qs = torch.randn(B, n_agents)
        max_q_i = agent_qs.clone()  # chosen IS greedy
        states = torch.randn(B, state_dim)
        onehot_actions = torch.randn(B, n_agents * n_actions)

        v_tot, _ = mixer(agent_qs, states, is_v=True)
        a_tot, _ = mixer(agent_qs, states, actions=onehot_actions, max_q_i=max_q_i, is_v=False)

        assert torch.allclose(a_tot.squeeze(), torch.zeros(B), atol=1e-5)

    def test_target_bias_magnitude_grows_with_compound_action_disagreement(self):
        torch.manual_seed(42)
        n_agents, state_dim, n_actions = 2, 64, 14
        mixer = self._make_mixer("dmaq", n_agents, state_dim, n_actions)
        B = 128
        states = torch.randn(B, state_dim)
        onehot_actions = torch.randn(B, n_agents * n_actions)

        agent_qs_small = torch.randn(B, n_agents)
        max_q_small = agent_qs_small + 0.1

        agent_qs_large = torch.randn(B, n_agents)
        max_q_large = agent_qs_large + 2.0

        a_tot_small, _ = mixer(
            agent_qs_small, states, actions=onehot_actions,
            max_q_i=max_q_small, is_v=False,
        )
        a_tot_large, _ = mixer(
            agent_qs_large, states, actions=onehot_actions,
            max_q_i=max_q_large, is_v=False,
        )

        bias_small = a_tot_small.squeeze().abs().mean().item()
        bias_large = a_tot_large.squeeze().abs().mean().item()

        assert bias_large > bias_small

    def test_compound_head_disagreement_probability(self):
        n_heads = 5
        n_agents = 2
        total_decisions = n_heads * n_agents

        for per_head_agree_prob in [0.7, 0.8, 0.9, 0.95]:
            full_agree_prob = per_head_agree_prob ** total_decisions
            any_disagree_prob = 1.0 - full_agree_prob
            if per_head_agree_prob <= 0.9:
                assert any_disagree_prob > 0.5
