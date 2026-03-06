from types import SimpleNamespace

import torch

from comrad.models.qplex_mixer import DMAQ_QattenMixer


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
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestQPlexRegressions:
    def test_dmaq_qatten_forward_supports_batch_size_one(self):
        mixer = DMAQ_QattenMixer(_default_cfg(), n_agents=2, state_dim=64, n_actions=5, unit_dim=32)
        q_tot, regs = mixer(torch.randn(1, 2), torch.randn(1, 64), is_v=True)
        assert q_tot.shape == (1, 1, 1)
        assert len(regs) == 1
