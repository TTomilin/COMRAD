import pytest

from sample_factory.cfg.arguments import preprocess_cfg
from sample_factory.utils.attr_dict import AttrDict


class _DummyEnvInfo:
    def __init__(self):
        self.obs_space = None
        self.num_agents = 2


def _qplex_cfg(**overrides):
    cfg = AttrDict(
        {
            'algo': 'QPLEX',
            'recurrence': -1,
            'normalize_returns': False,
            'qmix_buffer_batch_size': 32,
            'qmix_sequence_batch_size': 8,
            'qmix_log_interval': 100,
            'mixer': 'dmaq_qatten',
            'qplex_weighted_head': False,
            'qplex_adv_hypernet_layers': 1,
            'use_rnn': True,
            'rnn_type': 'gru',
            'rollout': 8,
            'per': False,
            'actor_critic_share_weights': True,
            'num_agents': 2,
            'cli_args': {},
            'replay_buffer_size': 10000,
            'learning_starts': 1000,
            'qplex_grad_accum_mini_bs': 16,
            'num_envs_per_worker': 2,
            'worker_num_splits': 1,
            'with_vtrace': False,
            'async_rl': True,
            'serial_mode': False,
            'num_policies': 1,
            'batched_sampling': True,
            'batch_size': 256,
            'num_batches_per_epoch': 1,
            'num_workers': 2,
        }
    )
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize('mini_bs', [0, -1])
def test_qplex_grad_accum_mini_bs_must_be_positive(mini_bs):
    cfg = _qplex_cfg(qplex_grad_accum_mini_bs=mini_bs)

    with pytest.raises(ValueError, match='qplex_grad_accum_mini_bs > 0'):
        preprocess_cfg(cfg, _DummyEnvInfo())
