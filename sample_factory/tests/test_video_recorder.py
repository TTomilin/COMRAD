from __future__ import annotations

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from sample_factory.utils.attr_dict import AttrDict

from comrad.envs.multiagent.doom_multiagent_wrapper import MultiAgentEnv, retry_doom
from comrad.utils.doom_utils import make_doom_multiplayer_env
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


class _MultiAgentPartialDoneEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        self._step_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        obs = [
            np.full((3, 2, 2), 10, dtype=np.uint8),
            np.full((3, 2, 2), 20, dtype=np.uint8),
        ]
        return obs, [{}, {}]

    def step(self, action):
        self._step_idx += 1
        obs = [
            np.full((3, 2, 2), 30 + self._step_idx, dtype=np.uint8),
            np.full((3, 2, 2), 40 + self._step_idx, dtype=np.uint8),
        ]
        if self._step_idx == 1:
            info = [{"HEALTH": 0.0}, {"HEALTH": 88.0}]
            return obs, [0.0, 0.0], [True, False], [False, False], info

        info = [{"HEALTH": 90.0}, {"HEALTH": 0.0}]
        return obs, [0.0, 0.0], [False, True], [False, False], info


def test_video_logger_multiagent_any_mode_saves_partial_boundaries(tmp_path):
    env = VideoLoggerWrapper(
        _MultiAgentPartialDoneEnv(),
        record_every=1,
        fps=8,
        is_multi=True,
        done_mode="any",
        output_dir=str(tmp_path),
    )

    env.reset()

    _, _, _, _, info1 = env.step([0, 0])
    payload1 = info1[0]["episode_extra_stats"]["wandb_video"]
    assert payload1["episode"] == 1
    assert payload1["done_mode"] == "any"
    assert payload1["reset_boundary"] is False
    assert payload1["agent_terminated"] == [True, False]

    _, _, _, _, info2 = env.step([0, 0])
    payload2 = info2[0]["episode_extra_stats"]["wandb_video"]
    assert payload2["episode"] == 2
    assert payload2["done_mode"] == "any"
    assert payload2["agent_terminated"] == [False, True]


class _HiddenResetBoundaryEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        self._step_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        obs = np.full((3, 2, 2), 11, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._step_idx += 1
        if self._step_idx == 1:
            obs = np.full((3, 2, 2), 12, dtype=np.uint8)
            return obs, 0.0, False, False, {}
        if self._step_idx == 2:
            obs = np.full((3, 2, 2), 21, dtype=np.uint8)
            return obs, 0.0, False, False, {"reset_info": {"from_crash_retry": True}, "_hidden_reset_count": 1}

        obs = np.full((3, 2, 2), 31, dtype=np.uint8)
        return obs, 0.0, True, False, {"true_objective": 7.0}


def test_video_logger_hidden_reset_boundary_advances_episode_counter(tmp_path):
    env = VideoLoggerWrapper(
        _HiddenResetBoundaryEnv(),
        record_every=2,
        fps=8,
        output_dir=str(tmp_path),
    )

    env.reset()

    _, _, _, _, info1 = env.step(0)
    assert "episode_extra_stats" not in info1

    _, _, terminated2, truncated2, info2 = env.step(0)
    assert terminated2 is False
    assert truncated2 is False
    assert "episode_extra_stats" not in info2

    _, _, terminated3, truncated3, info3 = env.step(0)
    assert terminated3 is True
    assert truncated3 is False

    payload = info3["episode_extra_stats"]["wandb_video"]
    assert payload["episode"] == 2
    assert payload["recorded_frames"] == 2

    with np.load(payload["path"]) as data:
        frames = data["frames"]
    assert frames.shape[0] == 2
    assert np.all(frames[0] == 21)
    assert np.all(frames[1] == 31)


class _MultipleHiddenResetBoundaryEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        self._step_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        return np.full((3, 2, 2), 5, dtype=np.uint8), {}

    def step(self, action):
        self._step_idx += 1
        if self._step_idx == 1:
            return np.full((3, 2, 2), 9, dtype=np.uint8), 0.0, False, False, {}
        if self._step_idx == 2:
            info = {"reset_info": {"from_retry": True}, "_hidden_reset_count": 2}
            return np.full((3, 2, 2), 17, dtype=np.uint8), 0.0, False, False, info

        return np.full((3, 2, 2), 33, dtype=np.uint8), 0.0, True, False, {"true_objective": 9.0}


def test_video_logger_counts_multiple_hidden_resets_toward_record_every(tmp_path):
    env = VideoLoggerWrapper(
        _MultipleHiddenResetBoundaryEnv(),
        record_every=3,
        fps=8,
        output_dir=str(tmp_path),
    )

    env.reset()
    env.step(0)
    env.step(0)
    _, _, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    payload = info["episode_extra_stats"]["wandb_video"]
    assert payload["episode"] == 3
    assert payload["recorded_frames"] == 2


class _TerminalHiddenResetEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        self._step_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        return np.full((3, 2, 2), 3, dtype=np.uint8), {}

    def step(self, action):
        self._step_idx += 1
        if self._step_idx == 1:
            return np.full((3, 2, 2), 7, dtype=np.uint8), 0.0, False, False, {}
        if self._step_idx == 2:
            info = {"reset_info": {"from_retry": True}, "_hidden_reset_count": 1}
            return np.full((3, 2, 2), 13, dtype=np.uint8), 0.0, True, False, info

        return np.full((3, 2, 2), 21, dtype=np.uint8), 0.0, True, False, {"true_objective": 11.0}


def test_video_logger_terminal_hidden_reset_does_not_block_next_episode_count(tmp_path):
    env = VideoLoggerWrapper(
        _TerminalHiddenResetEnv(),
        record_every=3,
        fps=8,
        output_dir=str(tmp_path),
    )

    env.reset()
    env.step(0)
    _, _, terminated2, truncated2, info2 = env.step(0)

    assert terminated2 is True
    assert truncated2 is False
    assert "episode_extra_stats" not in info2

    _, _, terminated3, truncated3, info3 = env.step(0)
    assert terminated3 is True
    assert truncated3 is False

    payload = info3["episode_extra_stats"]["wandb_video"]
    assert payload["episode"] == 3


def test_make_doom_multiplayer_env_uses_any_done_mode_for_on_policy_only(monkeypatch):
    import comrad.envs.multiagent.doom_multiagent_wrapper as doom_multiagent_wrapper
    import comrad.utils.doom_utils as doom_utils

    captured = []

    class _DummyMultiAgentEnv:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _CapturingVideoLoggerWrapper:
        def __init__(self, env, **kwargs):
            captured.append(kwargs["done_mode"])
            self.env = env
            self.kwargs = kwargs

    monkeypatch.setattr(doom_multiagent_wrapper, "MultiAgentEnv", _DummyMultiAgentEnv)
    monkeypatch.setattr(doom_utils, "VideoLoggerWrapper", _CapturingVideoLoggerWrapper)

    doom_spec = SimpleNamespace(
        num_agents=2,
        num_bots=0,
        shared_reward_alpha=0.0,
        shared_reward_scalarisation="sum",
    )
    cfg_base = {
        "env_frameskip": 4,
        "num_bots": -1,
        "num_agents": 2,
        "num_humans": 0,
        "wandb_record_every": 10,
        "wandb_video_fps": 35,
        "with_wandb": True,
        "shared_reward_alpha": None,
        "shared_reward_scalarisation": None,
        "experiment": "video-test",
        "train_dir": "/tmp/video-test",
    }

    make_doom_multiplayer_env(doom_spec, cfg=AttrDict({**cfg_base, "algo": "MAPPO"}), env_config=None)
    make_doom_multiplayer_env(doom_spec, cfg=AttrDict({**cfg_base, "algo": "QMIX"}), env_config=None)

    assert captured == ["any", "all"]


def test_retry_doom_stashes_reset_info_after_hidden_reset():
    class _FakeRetryEnv:
        def __init__(self):
            self.initialized = True
            self.closed = 0
            self.reset_calls = 0
            self.calls = 0
            self._pending_reset_infos = None

        def close(self):
            self.closed += 1

        def reset(self):
            self.reset_calls += 1
            return ["obs"], [{"restored": True}]

        @retry_doom(exception_class=RuntimeError, num_attempts=2, sleep_time=0, should_reset=True)
        def step(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return "ok"

    env = _FakeRetryEnv()

    result = env.step()

    assert result == "ok"
    assert env.closed == 1
    assert env.reset_calls == 1
    assert env._pending_reset_infos == [{"restored": True}]


def test_multiagent_env_attaches_pending_reset_info_without_overwriting_real_terminal_reset():
    env = object.__new__(MultiAgentEnv)
    env._pending_reset_infos = [{"hidden": 1}, {"hidden": 2}]
    env._pending_reset_count = 2

    infos = [{"value": 1}, {"value": 2, "reset_info": {"terminal": True}}]
    env._attach_pending_reset_infos(infos)

    assert infos[0]["reset_info"] == {"hidden": 1}
    assert infos[0]["_hidden_reset_count"] == 2
    assert infos[1]["reset_info"] == {"terminal": True}
    assert infos[1]["_hidden_reset_count"] == 2
    assert env._pending_reset_infos is None
    assert env._pending_reset_count == 0
