from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

from comrad.wrappers.video_recorder import VideoLoggerWrapper


class _AutoResetDoneEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 4, 4), dtype=np.uint8)
        self.worker_index = 2
        self.vector_index = 7

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.full((3, 4, 4), 11, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        obs = np.full((3, 4, 4), 99, dtype=np.uint8)
        info = {
            "reset_info": {},
            "true_objective": 44.0,
            "HEALTH": 96.0,
            "USER1": 73.0,
            "USER2": 5.0,
            "USER3": 1.0,
        }
        return obs, 0.0, True, False, info


def test_video_logger_records_diagnostic_metadata(tmp_path):
    env = VideoLoggerWrapper(_AutoResetDoneEnv(), record_every=1, fps=8, output_dir=str(tmp_path))

    obs, _ = env.reset()
    assert int(obs[0, 0, 0]) == 11

    _, _, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False

    payload = info["episode_extra_stats"]["wandb_video"]
    assert payload["episode"] == 1
    assert payload["worker_index"] == 2
    assert payload["vector_index"] == 7
    assert payload["recorded_frames"] == 1
    assert payload["true_objective"] == 44.0
    assert payload["reset_boundary"] is True

    with np.load(payload["path"]) as data:
        frames = data["frames"]
    assert frames.shape == (1, 3, 4, 4)
    assert np.all(frames[0] == 11)

    with open(payload["meta_path"], "r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)
    assert meta["recorded_frames"] == 1
    assert meta["HEALTH"] == 96.0
    assert meta["USER1"] == 73.0
    assert meta["USER2"] == 5.0
    assert meta["USER3"] == 1.0
    assert meta["terminated"] is True
    assert meta["truncated"] is False


class _MultiAgentDoneEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        self.worker_index = 4
        self.vector_index = 9

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = [
            np.full((3, 2, 2), 10, dtype=np.uint8),
            np.full((3, 2, 2), 20, dtype=np.uint8),
        ]
        return obs, [{}, {}]

    def step(self, action):
        obs = [
            np.full((3, 2, 2), 30, dtype=np.uint8),
            np.full((3, 2, 2), 40, dtype=np.uint8),
        ]
        info = [
            {
                "HEALTH": 88.0,
                "episode_time_tics": 212,
                "episode_timeout_tics": 5250,
                "player_dead": False,
                "reset_info": {},
            },
            {
                "HEALTH": 0.0,
                "episode_time_tics": 212,
                "episode_timeout_tics": 5250,
                "player_dead": True,
                "reset_info": {},
            },
        ]
        return obs, [0.0, 0.0], [True, True], [False, False], info


def test_video_logger_multiagent_metadata_uses_agent_lists(tmp_path):
    env = VideoLoggerWrapper(
        _MultiAgentDoneEnv(),
        record_every=1,
        fps=8,
        is_multi=True,
        output_dir=str(tmp_path),
    )

    env.reset()
    _, _, _, _, info = env.step([0, 0])

    payload = info[0]["episode_extra_stats"]["wandb_video"]
    assert payload["terminated"] is True
    assert payload["truncated"] is False
    assert payload["agent_HEALTH"] == [88.0, 0.0]
    assert payload["agent_episode_time_tics"] == [212, 212]
    assert payload["agent_player_dead"] == [False, True]
    assert payload["worker_index"] == 4
    assert payload["vector_index"] == 9
