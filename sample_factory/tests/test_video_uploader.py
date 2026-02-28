from __future__ import annotations

from types import SimpleNamespace

import torch

from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
from sample_factory.algo.utils.misc import EPISODIC
from sample_factory.utils.attr_dict import AttrDict
from comrad.utils.video_uploader import upload_video


class _DummyRunner:
    def __init__(self):
        self.policy_msg_handlers = {}
        self.env_steps = {0: 0}


def test_video_uploader_skips_missing_episode_extra_stats(monkeypatch):
    runner = _DummyRunner()
    cfg = SimpleNamespace(wandb_video_fps=35)
    upload_video(runner, cfg)
    handler = runner.policy_msg_handlers[EPISODIC][0]

    logged = {'count': 0}

    def _fake_log(*args, **kwargs):
        logged['count'] += 1

    monkeypatch.setattr('wandb.log', _fake_log)

    handler(runner, {EPISODIC: {}}, 0)
    handler(runner, {EPISODIC: {'episode_extra_stats': None}}, 0)

    assert logged['count'] == 0


def test_video_uploader_uploads_when_payload_present(tmp_path, monkeypatch):
    runner = _DummyRunner()
    cfg = SimpleNamespace(wandb_video_fps=35)
    upload_video(runner, cfg)
    handler = runner.policy_msg_handlers[EPISODIC][0]

    payload_path = tmp_path / 'vid_payload.npz'
    frames = torch.randint(0, 255, (4, 3, 8, 8), dtype=torch.uint8).numpy()

    import numpy as np

    np.savez_compressed(payload_path, frames=frames)

    logged = {'count': 0}

    def _fake_video(frames_arg, fps, format):
        assert frames_arg.shape == (4, 3, 8, 8)
        assert fps == 12
        assert format == 'mp4'
        return {'ok': True}

    def _fake_log(data, step=None):
        assert any(k.startswith('videos/p_00_ep_00003') for k in data.keys())
        logged['count'] += 1

    monkeypatch.setattr('wandb.Video', _fake_video)
    monkeypatch.setattr('wandb.log', _fake_log)

    msg = {
        EPISODIC: {
            'episode_extra_stats': {
                'wandb_video': {
                    'path': str(payload_path),
                    'fps': 12,
                    'episode': 3,
                }
            }
        }
    }

    handler(runner, msg, 0)

    assert logged['count'] == 1
    assert not payload_path.exists()


def test_batched_sampling_propagates_episode_extra_stats():
    runner = object.__new__(BatchedVectorEnvRunner)
    runner.curr_episode_reward = torch.tensor([1.0, 2.0], dtype=torch.float32)
    runner.curr_episode_len = torch.tensor([10, 10], dtype=torch.int32)
    runner.min_raw_rewards = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    runner.max_raw_rewards = torch.tensor([3.0, 4.0], dtype=torch.float32)
    runner.vec_env = SimpleNamespace(num_agents=2)
    runner.policy_id = 0
    runner.cfg = AttrDict({'summaries_use_frameskip': False})
    runner.env_info = SimpleNamespace(frameskip=1)

    rewards = torch.tensor([0.0, 0.0], dtype=torch.float32)
    dones = torch.tensor([True, True], dtype=torch.bool)
    infos = [
        {'episode_extra_stats': {'wandb_video': {'path': '/tmp/fake.npz', 'episode': 9, 'fps': 35}}},
        {},
    ]

    reports = BatchedVectorEnvRunner._process_env_step(runner, rewards, dones, infos)

    assert len(reports) == 1
    episodic = reports[0][EPISODIC]
    assert 'episode_extra_stats' in episodic
    assert episodic['episode_extra_stats']['wandb_video']['episode'] == 9
